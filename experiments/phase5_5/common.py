"""Phase 5.5 shared infrastructure: data, deterministic training, pinned external
artifacts, seeded held-out sample selection (docs/PHASE_5_5_PLAN.md §3-§5).

Experiment code only: it uses BeyondNN's public API and is not part of the package.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

import beyondnn as bnn

HERE = Path(__file__).resolve().parent
ARTIFACTS = HERE / "artifacts"
RESULTS = HERE / "results"
ARTIFACTS.mkdir(exist_ok=True)
RESULTS.mkdir(exist_ok=True)

BERT_REPO = "M-FAC/bert-tiny-finetuned-sst2"
BERT_REVISION = "41ad6709ec46b414749b37daf49cf5ca1c7dba7c"
GLUE_REVISION = "bcdcba79d07bc864c1c254ccfcedcce55bcc9a8c"
SST2_URL = (
    f"https://huggingface.co/datasets/nyu-mll/glue/resolve/{GLUE_REVISION}/sst2/"
    "validation-00000-of-00001.parquet"
)
SELECTION_SEED = 1234


def environment() -> dict[str, Any]:
    import numpy
    import sklearn

    out: dict[str, Any] = {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "numpy": numpy.__version__,
        "sklearn": sklearn.__version__,
        "platform": f"{platform.system()} {platform.machine()}",
        "processor": platform.processor(),
        "torch_threads": torch.get_num_threads(),
        "beyondnn_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=HERE
        ).stdout.strip(),
    }
    try:
        import captum

        out["captum"] = captum.__version__
    except ImportError:
        out["captum"] = None
    try:
        import transformers

        out["transformers"] = transformers.__version__
    except ImportError:
        out["transformers"] = None
    return out


def state_digest(model: nn.Module) -> str:
    """BeyondNN's own FULL model fingerprint (state digest) of the checkpoint."""
    from beyondnn.provenance import fingerprint_model

    return fingerprint_model(model).state_digest


# ------------------------------------------------------------------ tabular / images


def _split(n: int, labels: torch.Tensor) -> tuple[list[int], list[int], list[int]]:
    from sklearn.model_selection import train_test_split

    idx = list(range(n))
    train, rest = train_test_split(idx, test_size=0.4, random_state=0, stratify=labels.numpy())
    val, test = train_test_split(rest, test_size=0.5, random_state=0, stratify=labels[rest].numpy())
    return sorted(train), sorted(val), sorted(test)


@dataclass
class Data:
    name: str
    x: torch.Tensor
    y: torch.Tensor
    train: list[int]
    val: list[int]
    test: list[int]
    meta: dict[str, Any]


def breast_cancer() -> Data:
    from sklearn.datasets import load_breast_cancer

    raw = load_breast_cancer()
    x = torch.tensor(raw.data, dtype=torch.float32)
    y = torch.tensor(raw.target, dtype=torch.long)
    train, val, test = _split(len(y), y)
    mean, std = x[train].mean(0), x[train].std(0)
    x = (x - mean) / std
    return Data(
        "breast_cancer",
        x,
        y,
        train,
        val,
        test,
        {
            "source": "sklearn.datasets.load_breast_cancer",
            "n": len(y),
            "features": list(raw.feature_names),
            "standardised_with": "train mean/std",
        },
    )


def digits() -> Data:
    from sklearn.datasets import load_digits

    raw = load_digits()
    x = torch.tensor(raw.images, dtype=torch.float32).unsqueeze(1) / 16.0
    y = torch.tensor(raw.target, dtype=torch.long)
    train, val, test = _split(len(y), y)
    return Data(
        "digits",
        x,
        y,
        train,
        val,
        test,
        {"source": "sklearn.datasets.load_digits", "n": len(y), "scale": "/16"},
    )


class MLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(30, 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 2)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out: torch.Tensor = self.net(x)
        return out


class CNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, 3, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(8, 16, 3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool = nn.MaxPool2d(2)
        self.head = nn.Linear(16 * 4 * 4, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.pool(self.relu2(self.conv2(self.relu1(self.conv1(x)))))
        out: torch.Tensor = self.head(torch.flatten(h, 1))
        return out


def _accuracy(model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> float:
    with torch.no_grad():
        return float((model(x).argmax(1) == y).float().mean())


def train(kind: str) -> tuple[nn.Module, Data, dict[str, Any]]:
    """Deterministic CPU training (plan §3); cached checkpoint, verified by digest."""
    data = breast_cancer() if kind == "A" else digits()
    torch.manual_seed(0)
    model: nn.Module = MLP() if kind == "A" else CNN()
    path = ARTIFACTS / f"model_{kind}.pt"
    config = (
        {
            "optimizer": "Adam",
            "lr": 1e-3,
            "loss": "cross_entropy",
            "epochs": 300,
            "batch": "full",
            "seed": 0,
        }
        if kind == "A"
        else {
            "optimizer": "Adam",
            "lr": 1e-3,
            "loss": "cross_entropy",
            "epochs": 30,
            "batch": 64,
            "seed": 0,
        }
    )
    if path.exists():
        model.load_state_dict(torch.load(path, weights_only=True))
    else:
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        loss_fn = nn.CrossEntropyLoss()
        xtr, ytr = data.x[data.train], data.y[data.train]
        model.train()
        g = torch.Generator().manual_seed(0)
        for _ in range(config["epochs"]):
            if kind == "A":
                opt.zero_grad()
                loss_fn(model(xtr), ytr).backward()
                opt.step()
            else:
                for batch in torch.randperm(len(ytr), generator=g).split(64):
                    opt.zero_grad()
                    loss_fn(model(xtr[batch]), ytr[batch]).backward()
                    opt.step()
        torch.save(model.state_dict(), path)
    model.eval()
    info = {
        "architecture": repr(model),
        "training": config,
        "checkpoint_state_digest": state_digest(model),
        "accuracy": {
            s: _accuracy(model, data.x[getattr(data, s)], data.y[getattr(data, s)])
            for s in ("train", "val", "test")
        },
        "dataset": data.meta,
        "split": {
            "rule": "stratified 60/20/20, train_test_split random_state=0 twice",
            "sizes": [len(data.train), len(data.val), len(data.test)],
        },
    }
    return model, data, info


def select_samples(
    model: nn.Module, x: torch.Tensor, y: torch.Tensor, pool: list[int], n: int
) -> list[int]:
    """Plan §5: correctly classified pool members, seeded random n."""
    with torch.no_grad():
        pred = model(x[pool]).argmax(1)
    correct = [i for i, p in zip(pool, pred.tolist(), strict=True) if p == int(y[i])]
    order = torch.randperm(len(correct), generator=torch.Generator().manual_seed(SELECTION_SEED))
    return [correct[i] for i in order[:n].tolist()]


def margin_target(logits: torch.Tensor, path: str = "") -> tuple[Any, int, int, float]:
    """Plan §5: margin = logit[pred] - logit[runner-up], fixed from the clean pass."""
    top = torch.topk(logits.detach().flatten(), 2)
    pred, runner = int(top.indices[0]), int(top.indices[1])
    metric = bnn.interventions.metrics.difference([0, pred], [0, runner], path=path)
    return metric, pred, runner, float(top.values[0] - top.values[1])


# ------------------------------------------------------------------ transformer


def bert() -> tuple[nn.Module, Any, dict[str, Any]]:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(BERT_REPO, revision=BERT_REVISION)
    model = AutoModelForSequenceClassification.from_pretrained(BERT_REPO, revision=BERT_REVISION)
    model.eval()
    info = {
        "repo": BERT_REPO,
        "revision": BERT_REVISION,
        "parameters": sum(p.numel() for p in model.parameters()),
        "checkpoint_state_digest": state_digest(model),
    }
    return model, tok, info


def sst2_validation() -> list[dict[str, Any]]:
    import pyarrow.parquet as pq

    path = ARTIFACTS / f"sst2_validation_{GLUE_REVISION[:12]}.parquet"
    if not path.exists():
        urllib.request.urlretrieve(SST2_URL, path)
    table = pq.read_table(path).to_pylist()
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    for row in table:
        row["_file_sha256"] = digest
    return table


def save(name: str, payload: dict[str, Any]) -> Path:
    path = RESULTS / f"{name}.json"
    path.write_text(json.dumps(payload, indent=1, sort_keys=True, default=str))
    return path
