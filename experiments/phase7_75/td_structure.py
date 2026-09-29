"""Structural (weight-based) map of TD components to program variables; no interventions.
Usage (InterpBench env, from experiments/phase7_5/artifacts): python ../../phase7_75/td_structure.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import td_programs as TD  # noqa: E402


def variables_written(hl: Any, weight: torch.Tensor) -> list[str]:
    """Program variables whose residual dimensions ``weight`` (d_in x d_model) writes to."""
    labels = list(hl.residual_stream_labels)
    dims = (weight.abs().sum(0) > 0).nonzero().flatten().tolist()
    return sorted({re.sub(r"_\d+$", "", labels[d].split(":")[0]) for d in dims})


def structure(name: str) -> dict[str, Any]:
    case = TD.TDCase(TD.PROGRAMS[name])
    hl = case.get_hl_model(device="cpu").eval()
    roles = case.roles
    comps: dict[str, Any] = {}
    for L in range(hl.cfg.n_layers):
        for h in range(hl.cfg.n_heads):
            w = variables_written(hl, hl.blocks[L].attn.W_O[h])
            comps[f"l{L}_h{h}"] = w
        comps[f"l{L}_mlp"] = variables_written(hl, hl.blocks[L].mlp.W_out)

    def label(ws: list[str]) -> str:
        if not ws:
            return "empty"
        kinds = {("output" if v == "out" else roles.get(v, "unlabelled")) for v in ws}
        if kinds <= {"used", "output"}:
            return "known_true"
        if kinds == {"decoy"}:
            return "known_false"
        return "structurally_mixed"

    return {
        "cfg": {"n_layers": hl.cfg.n_layers, "n_heads": hl.cfg.n_heads, "d_model": hl.cfg.d_model},
        "roles": roles,
        "components": {k: {"writes": v, "label": label(v)} for k, v in comps.items()},
    }


if __name__ == "__main__":
    which = sys.argv[1:] or list(TD.PROGRAMS)
    out = {n: structure(n) for n in which}
    print(json.dumps(out, indent=1))
