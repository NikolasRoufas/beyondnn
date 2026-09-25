"""Deterministic tiny reference models for BeyondNN's own tests (M1.3).

These are controlled test subjects, not useful models: they are never trained,
use only PyTorch, run on CPU in milliseconds, and are built so that specific
edge cases exist on purpose.

=================  ==========================================================
Model              Edge cases it exists to exercise
=================  ==========================================================
``TinyMLP``        one module instance (``shared``) called twice per forward;
                   a ``torch.nn.functional`` op invisible to module hooks;
                   nested module (``shared.linear``); keyword-only ``scale``.
``TinyCNN``        Conv2d; BatchNorm buffers (train/eval differ); functional
                   ReLU; pooling modules; nested blocks; tuple output with
                   ``return_features=True``.
``TinyTransformer`` embeddings; hand-written attention returning a tuple;
                   LayerNorm; MLP; residual paths; ``lm_head.weight`` tied to
                   ``token_embedding.weight`` (same Parameter); non-persistent
                   buffers (``position_ids``, ``causal_mask``) used in forward;
                   integer token inputs; dict output with ``return_dict=True``.
=================  ==========================================================

Construction with ``seed`` is deterministic and does not change the caller's
global RNG state: parameters are initialised inside an isolated fork of the CPU
generator (other device generators are never touched).

Pinned module paths (owned by these models, stable for tests):

* TinyMLP: ``projection``, ``shared``, ``shared.linear``, ``head``
* TinyCNN: ``stem``, ``stem.conv``, ``stem.norm``, ``pool``, ``block``,
  ``block.conv``, ``block.norm``, ``gap``, ``classifier``
* TinyTransformer: ``token_embedding``, ``position_embedding``, ``blocks.{i}``,
  ``blocks.{i}.ln_attn``, ``blocks.{i}.attn``, ``blocks.{i}.attn.qkv``,
  ``blocks.{i}.attn.out``, ``blocks.{i}.ln_mlp``, ``blocks.{i}.mlp``,
  ``blocks.{i}.mlp.fc_in``, ``blocks.{i}.mlp.fc_out``, ``ln_final``, ``lm_head``
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from contextlib import contextmanager

import torch
import torch.nn.functional as F
from torch import nn

__all__ = ["TinyCNN", "TinyMLP", "TinyTransformer", "seeded_init"]


@contextmanager
def seeded_init(seed: int) -> Iterator[None]:
    """Seed the default CPU generator inside an isolated fork of its state.

    On exit the CPU generator is restored exactly (also on error). Uses the CPU
    generator's own ``manual_seed`` rather than ``torch.manual_seed``, which would
    also reseed CUDA/MPS generators that ``fork_rng(devices=[])`` does not restore.
    """
    with torch.random.fork_rng(devices=[]):
        torch.default_generator.manual_seed(seed)
        yield


# ------------------------------------------------------------------ TinyMLP


class _SharedBlock(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.linear = nn.Linear(width, width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + torch.tanh(self.linear(x))


class TinyMLP(nn.Module):
    """``projection -> shared -> F.gelu -> shared (same instance) -> * scale -> head``."""

    def __init__(self, d_in: int = 4, d_hidden: int = 8, d_out: int = 3, *, seed: int = 0) -> None:
        super().__init__()
        with seeded_init(seed):
            self.projection = nn.Linear(d_in, d_hidden)
            self.shared = _SharedBlock(d_hidden)
            self.head = nn.Linear(d_hidden, d_out)

    def forward(self, x: torch.Tensor, *, scale: float = 1.0) -> torch.Tensor:
        h = self.shared(self.projection(x))
        h = F.gelu(h)  # functional: not visible to module hooks
        h = self.shared(h)  # second call of the same module instance
        out: torch.Tensor = self.head(h * scale)
        return out


# ------------------------------------------------------------------ TinyCNN


class _ConvBlock(nn.Module):
    def __init__(self, c_in: int, c_out: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(c_in, c_out, kernel_size=3, padding=1)
        self.norm = nn.BatchNorm2d(c_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.norm(self.conv(x)))  # functional activation


class TinyCNN(nn.Module):
    """``stem -> pool -> block -> gap -> classifier`` for N x C x 8 x 8 inputs."""

    def __init__(self, in_channels: int = 1, n_classes: int = 4, *, seed: int = 0) -> None:
        super().__init__()
        with seeded_init(seed):
            self.stem = _ConvBlock(in_channels, 4)
            self.pool = nn.MaxPool2d(2)
            self.block = _ConvBlock(4, 8)
            self.gap = nn.AdaptiveAvgPool2d(1)
            self.classifier = nn.Linear(8, n_classes)

    def forward(
        self, x: torch.Tensor, *, return_features: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        h = self.block(self.pool(self.stem(x)))
        features = torch.flatten(self.gap(h), 1)
        logits: torch.Tensor = self.classifier(features)
        if return_features:
            return logits, features
        return logits


# ------------------------------------------------------------------ TinyTransformer


class _SelfAttention(nn.Module):
    """Explicit multi-head attention; returns ``(output, attention_weights)``."""

    def __init__(self, d_model: int, n_heads: int) -> None:
        super().__init__()
        if d_model % n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        self.n_heads = n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)

    def forward(
        self, x: torch.Tensor, causal_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        n, t, d = x.shape
        q, k, v = self.qkv(x).split(d, dim=-1)
        q, k, v = (z.view(n, t, self.n_heads, d // self.n_heads).transpose(1, 2) for z in (q, k, v))
        scores = q @ k.transpose(-2, -1) / math.sqrt(d // self.n_heads)
        scores = scores.masked_fill(~causal_mask[:t, :t], float("-inf"))
        weights = scores.softmax(dim=-1)
        mixed = (weights @ v).transpose(1, 2).reshape(n, t, d)
        return self.out(mixed), weights


class _MLP(nn.Module):
    def __init__(self, d_model: int, d_ff: int) -> None:
        super().__init__()
        self.fc_in = nn.Linear(d_model, d_ff)
        self.fc_out = nn.Linear(d_ff, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out: torch.Tensor = self.fc_out(F.gelu(self.fc_in(x)))
        return out


class _Block(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int) -> None:
        super().__init__()
        self.ln_attn = nn.LayerNorm(d_model)
        self.attn = _SelfAttention(d_model, n_heads)
        self.ln_mlp = nn.LayerNorm(d_model)
        self.mlp = _MLP(d_model, d_ff)

    def forward(self, x: torch.Tensor, causal_mask: torch.Tensor) -> torch.Tensor:
        attended, _weights = self.attn(self.ln_attn(x), causal_mask)
        x = x + attended  # residual
        out: torch.Tensor = x + self.mlp(self.ln_mlp(x))  # residual
        return out


class TinyTransformer(nn.Module):
    """A tiny pre-norm causal decoder over integer token ids ``(N, T)``.

    ``lm_head.weight`` IS ``token_embedding.weight`` (one Parameter). The
    non-persistent buffers ``position_ids`` (indexes the position embedding) and
    ``causal_mask`` (masks attention) are used in every forward.
    """

    def __init__(
        self,
        vocab_size: int = 32,
        d_model: int = 16,
        n_heads: int = 2,
        n_layers: int = 2,
        max_seq_len: int = 8,
        *,
        seed: int = 0,
    ) -> None:
        super().__init__()
        with seeded_init(seed):
            self.token_embedding = nn.Embedding(vocab_size, d_model)
            self.position_embedding = nn.Embedding(max_seq_len, d_model)
            self.blocks = nn.ModuleList(
                _Block(d_model, n_heads, 2 * d_model) for _ in range(n_layers)
            )
            self.ln_final = nn.LayerNorm(d_model)
            self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight  # tie: the same Parameter
        self.register_buffer("position_ids", torch.arange(max_seq_len), persistent=False)
        self.register_buffer(
            "causal_mask",
            torch.ones(max_seq_len, max_seq_len, dtype=torch.bool).tril(),
            persistent=False,
        )
        self.max_seq_len = max_seq_len

    def forward(
        self, input_ids: torch.Tensor, *, return_dict: bool = False
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        if input_ids.dtype not in (torch.int32, torch.int64):
            raise TypeError("input_ids must be an integer tensor")
        t = input_ids.shape[1]
        if t > self.max_seq_len:
            raise ValueError(f"sequence length {t} exceeds max_seq_len {self.max_seq_len}")
        positions = self.get_buffer("position_ids")[:t]
        h = self.token_embedding(input_ids) + self.position_embedding(positions)
        mask = self.get_buffer("causal_mask")
        for block in self.blocks:
            h = block(h, mask)
        hidden = self.ln_final(h)
        logits: torch.Tensor = self.lm_head(hidden)
        if return_dict:
            return {"logits": logits, "hidden_states": hidden}
        return logits
