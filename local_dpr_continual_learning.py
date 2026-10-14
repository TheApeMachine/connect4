#!/usr/bin/env python3
"""
Local Dynamic Precision Regularization (Local DPR)
==================================================

A single-file continual-learning experiment for testing whether surprise-driven,
block-local weight decay can decide *where* to forget.

The central local controller combines:

  1. Global surprise: a loss increase relative to an exponential moving baseline.
  2. Gradient conflict: negative cosine similarity between the current-task
     gradient and gradients measured on small anchor memories from older tasks.
  3. Reuse protection: old-task gradient magnitude acts as a coarse importance
     signal, reducing decay in blocks that still carry useful old-task structure.
  4. Current pressure: blocks receiving stronger current gradients receive more
     of the available forgetting budget when they are also in conflict.

For controlled comparisons, the script includes:

  * local_dpr            -- true block-local allocation
  * local_dpr_permuted   -- the exact same per-step decay values, reassigned to
                            the wrong blocks (spatial assignment control)
  * hierarchical_dpr     -- protected features, plastic classifier
  * global_dpr           -- one surprise-controlled decay for every block
  * constant_low         -- nearly no decay
  * constant_high        -- high uniform decay

Outputs are written in both CSV and JSONL, with a detailed Matplotlib figure pack.
The default two-task preset recreates the spirit of a Permuted-MNIST drift test;
the sequence preset extends it to multiple task shifts.

Example (Apple Silicon):

    python local_dpr_continual_learning.py \
        --preset two-task \
        --device mps \
        --output-dir results/local-dpr-two-task

Longer task sequence:

    python local_dpr_continual_learning.py \
        --preset sequence \
        --device mps \
        --output-dir results/local-dpr-sequence

A fast self-contained smoke test that does not download MNIST:

    python local_dpr_continual_learning.py \
        --preset smoke \
        --device cpu \
        --output-dir results/smoke

Notes
-----
* Anchor examples are used only to measure old-task gradients; they are never
  included in the optimization loss. This is diagnostic memory, not replay.
* AdamW's decoupled weight_decay value is changed independently for each
  controlled parameter group immediately before optimizer.step().
* Biases are placed in a separate zero-decay optimizer group.
* MPS is first-class: --device auto prefers CUDA, then MPS, then CPU. The script
  enables PyTorch's optional MPS CPU fallback unless --no-mps-fallback is used.
"""

from __future__ import annotations

# This variable is read by PyTorch when an unsupported MPS op is encountered.
# It is intentionally set before importing torch. Users can disable it through
# --no-mps-fallback; in that case main() removes it before device work begins.
import os
import sys

if "--no-mps-fallback" in sys.argv:
    os.environ.pop("PYTORCH_ENABLE_MPS_FALLBACK", None)
else:
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import argparse
import csv
import dataclasses
import json
import math
import random
import statistics
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor, nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, Subset


# -----------------------------------------------------------------------------
# Configuration and CLI
# -----------------------------------------------------------------------------

ALL_METHODS = (
    "local_dpr",
    "local_dpr_permuted",
    "hierarchical_dpr",
    "global_dpr",
    "constant_low",
    "constant_high",
)

PRESETS: dict[str, dict[str, Any]] = {
    "smoke": {
        "dataset": "synthetic",
        "num_tasks": 2,
        "epochs_per_task": 1,
        "train_subset": 512,
        "test_subset": 256,
        "eval_subset": 256,
        "batch_size": 128,
        "eval_every_steps": 4,
        "reference_refresh_steps": 4,
        "memory_samples_per_task": 64,
        "anchor_batch_size": 32,
        "seeds": [0],
        "methods": ["local_dpr", "local_dpr_permuted", "constant_low"],
        "hidden_dims": [64, 32],
    },
    "two-task": {
        "dataset": "mnist",
        "num_tasks": 2,
        "epochs_per_task": 3,
        "train_subset": 0,
        "test_subset": 0,
        "eval_subset": 2500,
        "batch_size": 256,
        "eval_every_steps": 50,
        "reference_refresh_steps": 50,
        "memory_samples_per_task": 384,
        "anchor_batch_size": 128,
        "seeds": [0],
        "methods": list(ALL_METHODS),
        "hidden_dims": [512, 512, 256],
    },
    "sequence": {
        "dataset": "mnist",
        "num_tasks": 5,
        "epochs_per_task": 2,
        "train_subset": 30000,
        "test_subset": 5000,
        "eval_subset": 2000,
        "batch_size": 256,
        "eval_every_steps": 100,
        "reference_refresh_steps": 75,
        "memory_samples_per_task": 256,
        "anchor_batch_size": 96,
        "seeds": [0, 1, 2],
        "methods": [
            "local_dpr",
            "local_dpr_permuted",
            "hierarchical_dpr",
            "global_dpr",
            "constant_low",
        ],
        "hidden_dims": [512, 512, 256],
    },
    "stress": {
        "dataset": "mnist",
        "num_tasks": 10,
        "epochs_per_task": 3,
        "train_subset": 0,
        "test_subset": 0,
        "eval_subset": 2000,
        "batch_size": 256,
        "eval_every_steps": 150,
        "reference_refresh_steps": 100,
        "memory_samples_per_task": 256,
        "anchor_batch_size": 96,
        "seeds": [0, 1, 2, 3, 4],
        "methods": [
            "local_dpr",
            "local_dpr_permuted",
            "hierarchical_dpr",
            "global_dpr",
            "constant_low",
        ],
        "hidden_dims": [768, 512, 256],
    },
}


def build_parser() -> argparse.ArgumentParser:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--preset", choices=sorted(PRESETS), default="two-task")
    known, _ = pre.parse_known_args()
    defaults = PRESETS[known.preset]

    parser = argparse.ArgumentParser(
        description="Block-local surprise/conflict-controlled weight decay for continual learning.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--preset", choices=sorted(PRESETS), default=known.preset)
    parser.add_argument("--output-dir", type=Path, default=Path("results/local-dpr"))
    parser.add_argument(
        "--device", choices=["auto", "mps", "cuda", "cpu"], default="auto"
    )
    parser.add_argument(
        "--mps-fallback",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Allow unsupported MPS operations to fall back to CPU.",
    )
    parser.add_argument("--mps-memory-fraction", type=float, default=0.0)
    parser.add_argument("--dry-run", action="store_true")

    parser.add_argument(
        "--dataset",
        choices=["mnist", "fashion-mnist", "synthetic"],
        default=defaults["dataset"],
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument(
        "--download", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--num-tasks", type=int, default=defaults["num_tasks"])
    parser.add_argument(
        "--epochs-per-task", type=int, default=defaults["epochs_per_task"]
    )
    parser.add_argument("--train-subset", type=int, default=defaults["train_subset"])
    parser.add_argument("--test-subset", type=int, default=defaults["test_subset"])
    parser.add_argument("--eval-subset", type=int, default=defaults["eval_subset"])
    parser.add_argument("--permutation-seed", type=int, default=1729)
    parser.add_argument("--synthetic-train-size", type=int, default=10000)
    parser.add_argument("--synthetic-test-size", type=int, default=2500)
    parser.add_argument("--synthetic-noise", type=float, default=0.65)

    parser.add_argument(
        "--methods", nargs="+", choices=ALL_METHODS, default=defaults["methods"]
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=defaults["seeds"])
    parser.add_argument(
        "--hidden-dims", nargs="+", type=int, default=defaults["hidden_dims"]
    )
    parser.add_argument("--batch-size", type=int, default=defaults["batch_size"])
    parser.add_argument("--eval-batch-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--betas", nargs=2, type=float, default=(0.9, 0.999))
    parser.add_argument("--gradient-clip", type=float, default=5.0)

    parser.add_argument("--lambda-low", type=float, default=1e-5)
    parser.add_argument("--lambda-high", type=float, default=3.5)
    parser.add_argument("--surprise-beta", type=float, default=0.98)
    parser.add_argument("--surprise-gain", type=float, default=1.8)
    parser.add_argument("--bootstrap-steps", type=int, default=120)
    parser.add_argument("--local-floor", type=float, default=0.0)
    parser.add_argument("--conflict-power", type=float, default=1.0)
    parser.add_argument("--pressure-power", type=float, default=0.5)
    parser.add_argument("--protection-strength", type=float, default=1.5)
    parser.add_argument("--max-reference-tasks", type=int, default=5)
    parser.add_argument(
        "--reference-refresh-steps",
        type=int,
        default=defaults["reference_refresh_steps"],
    )
    parser.add_argument(
        "--memory-samples-per-task",
        type=int,
        default=defaults["memory_samples_per_task"],
    )
    parser.add_argument(
        "--anchor-batch-size", type=int, default=defaults["anchor_batch_size"]
    )

    parser.add_argument(
        "--eval-every-steps", type=int, default=defaults["eval_every_steps"]
    )
    parser.add_argument("--log-every-steps", type=int, default=5)
    parser.add_argument("--save-checkpoints", action="store_true")
    parser.add_argument(
        "--plot-format", nargs="+", choices=["png", "pdf"], default=["png", "pdf"]
    )
    parser.add_argument("--bootstrap-resamples", type=int, default=2000)
    parser.add_argument(
        "--verbose", action=argparse.BooleanOptionalAction, default=True
    )
    return parser


# -----------------------------------------------------------------------------
# General utilities
# -----------------------------------------------------------------------------


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch, "mps") and hasattr(torch.mps, "manual_seed"):
        try:
            torch.mps.manual_seed(seed)
        except Exception:
            pass


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested, but CUDA is unavailable.")
    if requested == "mps" and not torch.backends.mps.is_available():
        built = torch.backends.mps.is_built()
        raise RuntimeError(
            f"--device mps was requested, but MPS is unavailable (built={built})."
        )
    return torch.device(requested)


def sync_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif (
        device.type == "mps"
        and hasattr(torch, "mps")
        and hasattr(torch.mps, "synchronize")
    ):
        torch.mps.synchronize()


def clear_device_cache(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif (
        device.type == "mps"
        and hasattr(torch, "mps")
        and hasattr(torch.mps, "empty_cache")
    ):
        torch.mps.empty_cache()


def device_memory_snapshot(device: torch.device) -> dict[str, float | str | None]:
    result: dict[str, float | str | None] = {"device": str(device)}
    if device.type == "cuda":
        result["allocated_mb"] = torch.cuda.memory_allocated(device) / (1024**2)
        result["reserved_mb"] = torch.cuda.memory_reserved(device) / (1024**2)
    elif device.type == "mps" and hasattr(torch, "mps"):
        for key, fn_name in (
            ("allocated_mb", "current_allocated_memory"),
            ("driver_allocated_mb", "driver_allocated_memory"),
            ("recommended_max_mb", "recommended_max_memory"),
        ):
            fn = getattr(torch.mps, fn_name, None)
            if callable(fn):
                try:
                    result[key] = float(fn()) / (1024**2)
                except Exception:
                    result[key] = None
    return result


def to_jsonable(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return to_jsonable(dataclasses.asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Tensor):
        if value.numel() == 1:
            return value.detach().cpu().item()
        return value.detach().cpu().tolist()
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    if isinstance(value, Mapping):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(to_jsonable(payload), indent=2, sort_keys=True), encoding="utf-8"
    )


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(to_jsonable(row), sort_keys=True) + "\n")


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fields.append(key)
                seen.add(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            flat: dict[str, Any] = {}
            for key in fields:
                value = to_jsonable(row.get(key))
                if isinstance(value, (dict, list)):
                    flat[key] = json.dumps(value, sort_keys=True)
                elif value is None:
                    flat[key] = ""
                else:
                    flat[key] = value
            writer.writerow(flat)


def write_dual(output_dir: Path, stem: str, rows: Sequence[Mapping[str, Any]]) -> None:
    write_jsonl(output_dir / f"{stem}.jsonl", rows)
    write_csv(output_dir / f"{stem}.csv", rows)


def save_figure(
    fig: plt.Figure, output_dir: Path, stem: str, formats: Sequence[str]
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        fig.savefig(output_dir / f"{stem}.{fmt}", dpi=180, bbox_inches="tight")
    plt.close(fig)


def bootstrap_ci(
    values: Sequence[float], resamples: int, seed: int = 0
) -> tuple[float, float]:
    arr = np.asarray(values, dtype=np.float64)
    if len(arr) == 0:
        return (float("nan"), float("nan"))
    if len(arr) == 1:
        return (float(arr[0]), float(arr[0]))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(arr), size=(resamples, len(arr)))
    means = arr[idx].mean(axis=1)
    return (float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975)))


# -----------------------------------------------------------------------------
# Data
# -----------------------------------------------------------------------------


class PermutedTensorDataset(Dataset[tuple[Tensor, Tensor]]):
    def __init__(self, images: Tensor, labels: Tensor, permutation: Tensor) -> None:
        if images.ndim != 2:
            raise ValueError(
                f"Expected flattened images [N,D], got {tuple(images.shape)}"
            )
        self.images = images
        self.labels = labels
        self.permutation = permutation.long()

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        x = self.images[index].index_select(0, self.permutation)
        return x, self.labels[index]


@dataclass
class TaskData:
    task_id: int
    train: Dataset[tuple[Tensor, Tensor]]
    test: Dataset[tuple[Tensor, Tensor]]
    permutation: Tensor


def deterministic_subset(dataset: Dataset[Any], count: int, seed: int) -> Dataset[Any]:
    if count <= 0 or count >= len(dataset):
        return dataset
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(dataset), generator=generator)[:count].tolist()
    return Subset(dataset, indices)


def load_base_tensors(
    args: argparse.Namespace,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    if args.dataset == "synthetic":
        generator = torch.Generator().manual_seed(args.permutation_seed + 991)
        input_dim = 28 * 28
        prototypes = torch.randn(10, input_dim, generator=generator)
        prototypes = prototypes / prototypes.norm(dim=1, keepdim=True).clamp_min(1e-8)

        def make_split(size: int) -> tuple[Tensor, Tensor]:
            labels = torch.randint(0, 10, (size,), generator=generator)
            noise = (
                torch.randn(size, input_dim, generator=generator) * args.synthetic_noise
            )
            images = prototypes[labels] * math.sqrt(input_dim) + noise
            images = (images - images.mean(dim=1, keepdim=True)) / images.std(
                dim=1, keepdim=True
            ).clamp_min(1e-5)
            return images.float(), labels.long()

        train_x, train_y = make_split(args.synthetic_train_size)
        test_x, test_y = make_split(args.synthetic_test_size)
        return train_x, train_y, test_x, test_y

    try:
        from torchvision.datasets import FashionMNIST, MNIST
    except Exception as exc:
        raise RuntimeError(
            "torchvision is required for MNIST/FashionMNIST. Install it with: pip install torchvision"
        ) from exc

    dataset_cls = MNIST if args.dataset == "mnist" else FashionMNIST
    train_ds = dataset_cls(root=args.data_dir, train=True, download=args.download)
    test_ds = dataset_cls(root=args.data_dir, train=False, download=args.download)
    train_x = train_ds.data.float().reshape(len(train_ds), -1).div_(255.0)
    test_x = test_ds.data.float().reshape(len(test_ds), -1).div_(255.0)
    # Standardize with statistics computed from the training split. This avoids
    # hard-coding dataset-specific constants and treats synthetic data similarly.
    mean = train_x.mean()
    std = train_x.std().clamp_min(1e-6)
    train_x = (train_x - mean) / std
    test_x = (test_x - mean) / std
    return train_x, train_ds.targets.long(), test_x, test_ds.targets.long()


def build_tasks(args: argparse.Namespace) -> list[TaskData]:
    train_x, train_y, test_x, test_y = load_base_tensors(args)
    input_dim = int(train_x.shape[1])
    tasks: list[TaskData] = []
    for task_id in range(args.num_tasks):
        if task_id == 0:
            permutation = torch.arange(input_dim)
        else:
            generator = torch.Generator().manual_seed(
                args.permutation_seed + task_id * 1009
            )
            permutation = torch.randperm(input_dim, generator=generator)
        train_ds: Dataset[tuple[Tensor, Tensor]] = PermutedTensorDataset(
            train_x, train_y, permutation
        )
        test_ds: Dataset[tuple[Tensor, Tensor]] = PermutedTensorDataset(
            test_x, test_y, permutation
        )
        train_ds = deterministic_subset(
            train_ds, args.train_subset, args.permutation_seed + 17 * task_id
        )
        test_ds = deterministic_subset(
            test_ds, args.test_subset, args.permutation_seed + 31 * task_id
        )
        tasks.append(
            TaskData(
                task_id=task_id, train=train_ds, test=test_ds, permutation=permutation
            )
        )
    return tasks


def make_loader(
    dataset: Dataset[Any],
    batch_size: int,
    shuffle: bool,
    seed: int,
    num_workers: int,
    device: torch.device,
) -> DataLoader[Any]:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
        generator=generator,
        drop_last=False,
    )


# -----------------------------------------------------------------------------
# Model and parameter groups
# -----------------------------------------------------------------------------


class FeedForwardBlock(nn.Module):
    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.activation = nn.GELU()

    def forward(self, x: Tensor) -> Tensor:
        return self.activation(self.linear(x))


class MultiBlockMLP(nn.Module):
    def __init__(
        self, input_dim: int, hidden_dims: Sequence[int], num_classes: int = 10
    ) -> None:
        super().__init__()
        dims = [input_dim, *hidden_dims]
        self.feature_blocks = nn.ModuleList(
            [FeedForwardBlock(dims[i], dims[i + 1]) for i in range(len(hidden_dims))]
        )
        self.classifier = nn.Linear(hidden_dims[-1], num_classes)

    def forward(self, x: Tensor) -> Tensor:
        for block in self.feature_blocks:
            x = block(x)
        return self.classifier(x)

    def controlled_blocks(self) -> list[tuple[str, nn.Module]]:
        blocks = [
            (f"features_{i}", block) for i, block in enumerate(self.feature_blocks)
        ]
        blocks.append(("classifier", self.classifier))
        return blocks


@dataclass
class ParameterLayout:
    block_params: dict[str, list[nn.Parameter]]
    no_decay_params: list[nn.Parameter]
    block_order: list[str]


def build_parameter_layout(model: MultiBlockMLP) -> ParameterLayout:
    block_params: dict[str, list[nn.Parameter]] = {}
    no_decay: list[nn.Parameter] = []
    claimed: set[int] = set()
    for block_name, module in model.controlled_blocks():
        decayable: list[nn.Parameter] = []
        for _, param in module.named_parameters(recurse=True):
            if not param.requires_grad:
                continue
            claimed.add(id(param))
            if param.ndim >= 2:
                decayable.append(param)
            else:
                no_decay.append(param)
        if not decayable:
            raise RuntimeError(
                f"Controlled block {block_name!r} has no matrix parameters."
            )
        block_params[block_name] = decayable
    for param in model.parameters():
        if param.requires_grad and id(param) not in claimed:
            no_decay.append(param)
    return ParameterLayout(
        block_params=block_params,
        no_decay_params=no_decay,
        block_order=list(block_params),
    )


def build_optimizer(
    model: MultiBlockMLP, layout: ParameterLayout, args: argparse.Namespace
) -> AdamW:
    groups: list[dict[str, Any]] = []
    for block_name in layout.block_order:
        groups.append(
            {
                "params": layout.block_params[block_name],
                "weight_decay": args.lambda_low,
                "block_name": block_name,
                "controlled_decay": True,
            }
        )
    if layout.no_decay_params:
        groups.append(
            {
                "params": layout.no_decay_params,
                "weight_decay": 0.0,
                "block_name": "no_decay",
                "controlled_decay": False,
            }
        )
    # foreach=False avoids backend-dependent tensor-list behavior and is the most
    # conservative path on MPS.
    return AdamW(
        groups,
        lr=args.learning_rate,
        betas=tuple(args.betas),
        eps=1e-8,
        foreach=False,
    )


def set_optimizer_decays(optimizer: AdamW, lambdas: Mapping[str, float]) -> None:
    for group in optimizer.param_groups:
        if bool(group.get("controlled_decay", False)):
            block_name = str(group["block_name"])
            group["weight_decay"] = float(lambdas[block_name])
        else:
            group["weight_decay"] = 0.0


# -----------------------------------------------------------------------------
# Surprise and gradient-reference machinery
# -----------------------------------------------------------------------------


@dataclass
class SurpriseState:
    loss: float
    loss_ema: float
    relative_surprise: float
    surprise_gate: float
    bootstrap_gate: float


class SurpriseTracker:
    def __init__(self, beta: float, gain: float, bootstrap_steps: int) -> None:
        self.beta = beta
        self.gain = gain
        self.bootstrap_steps = max(1, bootstrap_steps)
        self.loss_ema: float | None = None
        self.steps = 0

    def observe(self, loss_value: float) -> SurpriseState:
        self.steps += 1
        if self.loss_ema is None:
            prior = loss_value
            relative = 1.0
        else:
            prior = self.loss_ema
            relative = max(0.0, (loss_value - prior) / (abs(prior) + 1e-8))
        data_gate = 1.0 - math.exp(-self.gain * relative)
        bootstrap_gate = math.exp(-(self.steps - 1) / self.bootstrap_steps)
        gate = max(data_gate, bootstrap_gate)
        if self.loss_ema is None:
            self.loss_ema = loss_value
        else:
            self.loss_ema = self.beta * self.loss_ema + (1.0 - self.beta) * loss_value
        return SurpriseState(
            loss=loss_value,
            loss_ema=float(prior),
            relative_surprise=float(relative),
            surprise_gate=float(min(max(gate, 0.0), 1.0)),
            bootstrap_gate=float(bootstrap_gate),
        )


@dataclass
class GradientReference:
    task_id: int
    normalized: list[Tensor]
    rms: float
    norm: float
    numel: int


@dataclass
class BlockGradientStats:
    block: str
    current_rms: float
    current_norm: float
    conflict: float
    alignment: float
    mean_cosine: float
    old_importance: float
    reference_count: int
    pressure_normalized: float = 0.0
    importance_normalized: float = 0.0
    local_score: float = 0.0
    local_allocation: float = 0.0


class AnchorMemory:
    """Stores small CPU anchor sets and periodically measures old-task gradients."""

    def __init__(
        self, samples_per_task: int, anchor_batch_size: int, max_reference_tasks: int
    ) -> None:
        self.samples_per_task = samples_per_task
        self.anchor_batch_size = anchor_batch_size
        self.max_reference_tasks = max_reference_tasks
        self.anchors: dict[int, tuple[Tensor, Tensor]] = {}
        self.references: dict[str, list[GradientReference]] = defaultdict(list)
        self.last_refresh_step = -1

    def add_task(
        self, task_id: int, dataset: Dataset[tuple[Tensor, Tensor]], seed: int
    ) -> None:
        count = min(self.samples_per_task, len(dataset))
        generator = torch.Generator().manual_seed(seed)
        indices = torch.randperm(len(dataset), generator=generator)[:count].tolist()
        xs: list[Tensor] = []
        ys: list[Tensor] = []
        for idx in indices:
            x, y = dataset[idx]
            xs.append(x.detach().cpu().float())
            ys.append(torch.as_tensor(y).detach().cpu().long())
        self.anchors[task_id] = (torch.stack(xs), torch.stack(ys))

    def has_references(self) -> bool:
        return any(self.references.values())

    def refresh(
        self,
        model: MultiBlockMLP,
        layout: ParameterLayout,
        device: torch.device,
        step: int,
        seed: int,
    ) -> None:
        if not self.anchors:
            self.references.clear()
            self.last_refresh_step = step
            return

        selected_tasks = sorted(self.anchors)[-self.max_reference_tasks :]
        all_params: list[nn.Parameter] = []
        param_slices: dict[str, tuple[int, int]] = {}
        cursor = 0
        for block in layout.block_order:
            params = layout.block_params[block]
            all_params.extend(params)
            param_slices[block] = (cursor, cursor + len(params))
            cursor += len(params)

        new_refs: dict[str, list[GradientReference]] = defaultdict(list)
        was_training = model.training
        model.eval()
        rng = torch.Generator().manual_seed(seed + step * 7919)

        for task_id in selected_tasks:
            x_cpu, y_cpu = self.anchors[task_id]
            batch_count = min(self.anchor_batch_size, len(x_cpu))
            sample_idx = torch.randperm(len(x_cpu), generator=rng)[:batch_count]
            x = x_cpu.index_select(0, sample_idx).to(
                device, non_blocking=device.type == "cuda"
            )
            y = y_cpu.index_select(0, sample_idx).to(
                device, non_blocking=device.type == "cuda"
            )
            logits = model(x)
            loss = nn.functional.cross_entropy(logits, y)
            grads = torch.autograd.grad(
                loss,
                all_params,
                retain_graph=False,
                create_graph=False,
                allow_unused=True,
            )
            for block in layout.block_order:
                start, end = param_slices[block]
                block_grads: list[Tensor] = []
                norm_sq = torch.zeros((), device=device, dtype=torch.float32)
                numel = 0
                for param, grad in zip(all_params[start:end], grads[start:end]):
                    if grad is None:
                        g = torch.zeros_like(param, memory_format=torch.preserve_format)
                    else:
                        g = grad.detach()
                    g32 = g.float()
                    block_grads.append(g32)
                    norm_sq = norm_sq + g32.square().sum()
                    numel += g32.numel()
                norm = float(torch.sqrt(norm_sq + 1e-20).detach().cpu())
                if norm <= 1e-12:
                    normalized = [torch.zeros_like(g) for g in block_grads]
                else:
                    normalized = [g / norm for g in block_grads]
                rms = norm / math.sqrt(max(1, numel))
                new_refs[block].append(
                    GradientReference(
                        task_id=task_id,
                        normalized=normalized,
                        rms=float(rms),
                        norm=float(norm),
                        numel=numel,
                    )
                )
            del x, y, logits, loss, grads

        self.references = new_refs
        self.last_refresh_step = step
        model.train(was_training)

    def current_block_stats(
        self, layout: ParameterLayout, device: torch.device
    ) -> dict[str, BlockGradientStats]:
        stats: dict[str, BlockGradientStats] = {}
        for block in layout.block_order:
            params = layout.block_params[block]
            norm_sq = torch.zeros((), device=device, dtype=torch.float32)
            numel = 0
            current_grads: list[Tensor] = []
            for param in params:
                if param.grad is None:
                    grad = torch.zeros_like(param)
                else:
                    grad = param.grad.detach()
                grad32 = grad.float()
                current_grads.append(grad32)
                norm_sq = norm_sq + grad32.square().sum()
                numel += grad32.numel()
            norm_tensor = torch.sqrt(norm_sq + 1e-20)
            norm = float(norm_tensor.detach().cpu())
            rms = norm / math.sqrt(max(1, numel))
            cosines: list[float] = []
            refs = self.references.get(block, [])
            if norm > 1e-12:
                for ref in refs:
                    dot = torch.zeros((), device=device, dtype=torch.float32)
                    for grad, ref_grad in zip(current_grads, ref.normalized):
                        dot = dot + (grad * ref_grad).sum()
                    cosines.append(float((dot / norm_tensor).detach().cpu()))
            conflict = max([max(0.0, -c) for c in cosines], default=0.0)
            alignment = max([max(0.0, c) for c in cosines], default=0.0)
            mean_cosine = float(np.mean(cosines)) if cosines else 0.0
            old_importance = max([ref.rms for ref in refs], default=0.0)
            stats[block] = BlockGradientStats(
                block=block,
                current_rms=float(rms),
                current_norm=float(norm),
                conflict=float(conflict),
                alignment=float(alignment),
                mean_cosine=mean_cosine,
                old_importance=float(old_importance),
                reference_count=len(refs),
            )
        return stats


# -----------------------------------------------------------------------------
# Decay controllers
# -----------------------------------------------------------------------------


@dataclass
class ControllerOutput:
    applied_lambdas: dict[str, float]
    source_lambdas: dict[str, float]
    block_stats: dict[str, BlockGradientStats]
    surprise: SurpriseState
    decay_budget_mean: float
    decay_budget_sum: float
    assignment: dict[str, str]


class DecayController:
    def __init__(
        self,
        method: str,
        block_order: Sequence[str],
        args: argparse.Namespace,
        seed: int,
    ) -> None:
        self.method = method
        self.block_order = list(block_order)
        self.args = args
        self.surprise = SurpriseTracker(
            args.surprise_beta, args.surprise_gain, args.bootstrap_steps
        )
        self.permutation_rng = random.Random(seed + 40499)

    def _normalize_stats(self, stats: dict[str, BlockGradientStats]) -> None:
        pressures = [s.current_rms for s in stats.values() if s.current_rms > 0]
        importances = [s.old_importance for s in stats.values() if s.old_importance > 0]
        pressure_scale = statistics.median(pressures) if pressures else 1.0
        importance_scale = statistics.median(importances) if importances else 1.0
        max_score = 0.0
        for block in self.block_order:
            s = stats[block]
            s.pressure_normalized = min(8.0, s.current_rms / (pressure_scale + 1e-12))
            s.importance_normalized = (
                min(8.0, s.old_importance / (importance_scale + 1e-12))
                if importances
                else 0.0
            )
            if s.reference_count == 0:
                # On the first task, use global surprise to remove arbitrary
                # initialization structure. Once old-task references exist,
                # allocation becomes genuinely local.
                score = 1.0
            else:
                conflict_term = s.conflict**self.args.conflict_power
                pressure_term = s.pressure_normalized**self.args.pressure_power
                protection = (
                    1.0 + self.args.protection_strength * s.importance_normalized
                )
                score = conflict_term * pressure_term / protection
            s.local_score = float(score)
            max_score = max(max_score, score)
        for block in self.block_order:
            s = stats[block]
            if max_score <= 1e-12:
                s.local_allocation = 0.0
            else:
                s.local_allocation = float(s.local_score / max_score)

    def compute(
        self,
        loss_value: float,
        stats: dict[str, BlockGradientStats],
    ) -> ControllerOutput:
        surprise = self.surprise.observe(loss_value)
        self._normalize_stats(stats)
        low = float(self.args.lambda_low)
        high = float(self.args.lambda_high)
        span = high - low
        source: dict[str, float] = {}

        if self.method == "constant_low":
            source = {block: low for block in self.block_order}
        elif self.method == "constant_high":
            source = {block: high for block in self.block_order}
        elif self.method == "global_dpr":
            value = low + span * surprise.surprise_gate
            source = {block: value for block in self.block_order}
        elif self.method == "hierarchical_dpr":
            value = low + span * surprise.surprise_gate
            source = {
                block: (value if block == "classifier" else low)
                for block in self.block_order
            }
        elif self.method in {"local_dpr", "local_dpr_permuted"}:
            for block in self.block_order:
                allocation = (
                    self.args.local_floor
                    + (1.0 - self.args.local_floor) * stats[block].local_allocation
                )
                source[block] = low + span * surprise.surprise_gate * allocation
        else:
            raise ValueError(f"Unknown method: {self.method}")

        assignment = {block: block for block in self.block_order}
        applied = dict(source)
        if self.method == "local_dpr_permuted" and len(self.block_order) > 1:
            sources = list(self.block_order)
            shuffled = list(self.block_order)
            # Use a derangement so no block receives its own locally computed
            # decay. The multiset and total decay budget remain identical.
            for _ in range(32):
                self.permutation_rng.shuffle(shuffled)
                if all(
                    target != source_block
                    for target, source_block in zip(sources, shuffled)
                ):
                    break
            else:
                shuffled = shuffled[1:] + shuffled[:1]
            applied = {}
            assignment = {}
            for target, source_block in zip(self.block_order, shuffled):
                applied[target] = source[source_block]
                assignment[target] = source_block

        values = list(applied.values())
        return ControllerOutput(
            applied_lambdas=applied,
            source_lambdas=source,
            block_stats=stats,
            surprise=surprise,
            decay_budget_mean=float(np.mean(values)),
            decay_budget_sum=float(np.sum(values)),
            assignment=assignment,
        )


# -----------------------------------------------------------------------------
# Training and evaluation
# -----------------------------------------------------------------------------


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataset: Dataset[tuple[Tensor, Tensor]],
    device: torch.device,
    batch_size: int,
    subset_count: int,
    seed: int,
    num_workers: int,
) -> tuple[float, float, int]:
    eval_ds = deterministic_subset(dataset, subset_count, seed)
    loader = make_loader(eval_ds, batch_size, False, seed, num_workers, device)
    was_training = model.training
    model.eval()
    total = 0
    correct = 0
    loss_sum = 0.0
    for x, y in loader:
        x = x.to(device, non_blocking=device.type == "cuda")
        y = y.to(device, non_blocking=device.type == "cuda")
        logits = model(x)
        loss = nn.functional.cross_entropy(logits, y, reduction="sum")
        loss_sum += float(loss.detach().cpu())
        correct += int((logits.argmax(dim=-1) == y).sum().detach().cpu())
        total += int(y.numel())
    model.train(was_training)
    return correct / max(1, total), loss_sum / max(1, total), total


def evaluate_all_tasks(
    model: nn.Module,
    tasks: Sequence[TaskData],
    device: torch.device,
    args: argparse.Namespace,
    seed: int,
    method: str,
    global_step: int,
    train_task: int,
    event: str,
    full: bool = False,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    subset = 0 if full else args.eval_subset
    for task in tasks:
        accuracy, loss, count = evaluate(
            model,
            task.test,
            device,
            args.eval_batch_size,
            subset,
            seed + task.task_id * 101,
            args.num_workers,
        )
        rows.append(
            {
                "seed": seed,
                "method": method,
                "global_step": global_step,
                "train_task": train_task,
                "eval_task": task.task_id,
                "event": event,
                "accuracy": accuracy,
                "loss": loss,
                "examples": count,
            }
        )
    return rows


def zero_block_stats(layout: ParameterLayout) -> dict[str, BlockGradientStats]:
    return {
        block: BlockGradientStats(
            block=block,
            current_rms=0.0,
            current_norm=0.0,
            conflict=0.0,
            alignment=0.0,
            mean_cosine=0.0,
            old_importance=0.0,
            reference_count=0,
        )
        for block in layout.block_order
    }


@dataclass
class RunArtifacts:
    train_rows: list[dict[str, Any]] = field(default_factory=list)
    controller_rows: list[dict[str, Any]] = field(default_factory=list)
    eval_rows: list[dict[str, Any]] = field(default_factory=list)
    task_end_rows: list[dict[str, Any]] = field(default_factory=list)
    memory_rows: list[dict[str, Any]] = field(default_factory=list)
    timing_rows: list[dict[str, Any]] = field(default_factory=list)


def run_single_method(
    args: argparse.Namespace,
    tasks: Sequence[TaskData],
    device: torch.device,
    seed: int,
    method: str,
    initial_state: Mapping[str, Tensor],
    run_dir: Path,
) -> RunArtifacts:
    set_seed(seed)
    input_dim = int(tasks[0].permutation.numel())
    model = MultiBlockMLP(input_dim, args.hidden_dims).to(device)
    model.load_state_dict(initial_state)
    layout = build_parameter_layout(model)
    optimizer = build_optimizer(model, layout, args)
    controller = DecayController(method, layout.block_order, args, seed)
    memory = AnchorMemory(
        args.memory_samples_per_task, args.anchor_batch_size, args.max_reference_tasks
    )
    artifacts = RunArtifacts()
    global_step = 0

    artifacts.eval_rows.extend(
        evaluate_all_tasks(
            model,
            tasks,
            device,
            args,
            seed,
            method,
            global_step,
            -1,
            "initial",
            full=False,
        )
    )

    run_start = time.perf_counter()
    for task in tasks:
        task_start = time.perf_counter()
        if method.startswith("local_dpr") and memory.anchors:
            memory.refresh(model, layout, device, global_step, seed)

        for epoch in range(args.epochs_per_task):
            loader_seed = seed * 1_000_003 + task.task_id * 10_007 + epoch * 97
            train_loader = make_loader(
                task.train,
                args.batch_size,
                True,
                loader_seed,
                args.num_workers,
                device,
            )
            epoch_start = time.perf_counter()
            epoch_loss = 0.0
            epoch_correct = 0
            epoch_seen = 0
            for batch_index, (x, y) in enumerate(train_loader):
                global_step += 1
                x = x.to(device, non_blocking=device.type == "cuda")
                y = y.to(device, non_blocking=device.type == "cuda")
                optimizer.zero_grad(set_to_none=True)
                logits = model(x)
                loss = nn.functional.cross_entropy(logits, y)
                loss.backward()

                if method.startswith("local_dpr"):
                    refresh_due = memory.anchors and (
                        memory.last_refresh_step < 0
                        or global_step - memory.last_refresh_step
                        >= args.reference_refresh_steps
                    )
                    if refresh_due:
                        memory.refresh(model, layout, device, global_step, seed)
                    stats = memory.current_block_stats(layout, device)
                else:
                    # Current gradient RMS remains useful in the logs, even for
                    # controllers that do not consume conflict information.
                    stats = AnchorMemory(1, 1, 1).current_block_stats(layout, device)

                output = controller.compute(float(loss.detach().cpu()), stats)
                set_optimizer_decays(optimizer, output.applied_lambdas)

                if args.gradient_clip > 0:
                    grad_norm = float(
                        nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
                        .detach()
                        .cpu()
                    )
                else:
                    grad_norm = float("nan")
                optimizer.step()

                batch_size_actual = int(y.numel())
                epoch_loss += float(loss.detach().cpu()) * batch_size_actual
                epoch_correct += int((logits.argmax(dim=-1) == y).sum().detach().cpu())
                epoch_seen += batch_size_actual

                if global_step % args.log_every_steps == 0 or batch_index == 0:
                    artifacts.train_rows.append(
                        {
                            "seed": seed,
                            "method": method,
                            "task": task.task_id,
                            "epoch": epoch,
                            "batch": batch_index,
                            "global_step": global_step,
                            "loss": float(loss.detach().cpu()),
                            "batch_accuracy": float(
                                (logits.argmax(dim=-1) == y)
                                .float()
                                .mean()
                                .detach()
                                .cpu()
                            ),
                            "gradient_norm_before_clip": grad_norm,
                            "loss_ema": output.surprise.loss_ema,
                            "relative_surprise": output.surprise.relative_surprise,
                            "surprise_gate": output.surprise.surprise_gate,
                            "bootstrap_gate": output.surprise.bootstrap_gate,
                            "decay_budget_mean": output.decay_budget_mean,
                            "decay_budget_sum": output.decay_budget_sum,
                        }
                    )
                    for block in layout.block_order:
                        s = output.block_stats[block]
                        artifacts.controller_rows.append(
                            {
                                "seed": seed,
                                "method": method,
                                "task": task.task_id,
                                "epoch": epoch,
                                "batch": batch_index,
                                "global_step": global_step,
                                "block": block,
                                "lambda": output.applied_lambdas[block],
                                "source_lambda": output.source_lambdas[block],
                                "lambda_source_block": output.assignment[block],
                                "surprise_gate": output.surprise.surprise_gate,
                                "relative_surprise": output.surprise.relative_surprise,
                                "current_gradient_rms": s.current_rms,
                                "gradient_conflict": s.conflict,
                                "gradient_alignment": s.alignment,
                                "mean_reference_cosine": s.mean_cosine,
                                "old_task_importance": s.old_importance,
                                "pressure_normalized": s.pressure_normalized,
                                "importance_normalized": s.importance_normalized,
                                "local_score": s.local_score,
                                "local_allocation": s.local_allocation,
                                "reference_count": s.reference_count,
                            }
                        )

                if (
                    args.eval_every_steps > 0
                    and global_step % args.eval_every_steps == 0
                ):
                    artifacts.eval_rows.extend(
                        evaluate_all_tasks(
                            model,
                            tasks,
                            device,
                            args,
                            seed,
                            method,
                            global_step,
                            task.task_id,
                            "periodic",
                            full=False,
                        )
                    )

            sync_device(device)
            epoch_seconds = time.perf_counter() - epoch_start
            if args.verbose:
                controlled = {
                    str(group["block_name"]): float(group["weight_decay"])
                    for group in optimizer.param_groups
                    if group.get("controlled_decay", False)
                }
                print(
                    f"{method:20s} | seed={seed} task={task.task_id} epoch={epoch + 1}/{args.epochs_per_task} "
                    f"loss={epoch_loss / max(1, epoch_seen):.4f} acc={epoch_correct / max(1, epoch_seen):.4f} "
                    f"decay={controlled}"
                )
            artifacts.timing_rows.append(
                {
                    "seed": seed,
                    "method": method,
                    "task": task.task_id,
                    "epoch": epoch,
                    "seconds": epoch_seconds,
                    "examples": epoch_seen,
                    "examples_per_second": epoch_seen / max(epoch_seconds, 1e-9),
                }
            )

        # Store anchors only after the task has been learned. They are never used
        # as replay samples in the training loss.
        if method.startswith("local_dpr"):
            memory.add_task(task.task_id, task.train, seed + task.task_id * 811)
            memory.refresh(model, layout, device, global_step, seed)
            for block in layout.block_order:
                refs = memory.references.get(block, [])
                for ref in refs:
                    artifacts.memory_rows.append(
                        {
                            "seed": seed,
                            "method": method,
                            "after_task": task.task_id,
                            "reference_task": ref.task_id,
                            "global_step": global_step,
                            "block": block,
                            "reference_gradient_rms": ref.rms,
                            "reference_gradient_norm": ref.norm,
                            "reference_numel": ref.numel,
                        }
                    )

        task_eval = evaluate_all_tasks(
            model,
            tasks,
            device,
            args,
            seed,
            method,
            global_step,
            task.task_id,
            "task_end",
            full=True,
        )
        artifacts.eval_rows.extend(task_eval)
        artifacts.task_end_rows.extend(task_eval)
        sync_device(device)
        artifacts.timing_rows.append(
            {
                "seed": seed,
                "method": method,
                "task": task.task_id,
                "epoch": -1,
                "seconds": time.perf_counter() - task_start,
                "examples": len(task.train) * args.epochs_per_task,
                "examples_per_second": (len(task.train) * args.epochs_per_task)
                / max(time.perf_counter() - task_start, 1e-9),
            }
        )

        if args.save_checkpoints:
            checkpoint_dir = run_dir / "checkpoints"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "seed": seed,
                    "method": method,
                    "task": task.task_id,
                    "global_step": global_step,
                },
                checkpoint_dir / f"seed-{seed}_{method}_task-{task.task_id}.pt",
            )

    artifacts.timing_rows.append(
        {
            "seed": seed,
            "method": method,
            "task": -1,
            "epoch": -1,
            "seconds": time.perf_counter() - run_start,
            "examples": sum(len(t.train) for t in tasks) * args.epochs_per_task,
            "examples_per_second": sum(len(t.train) for t in tasks)
            * args.epochs_per_task
            / max(time.perf_counter() - run_start, 1e-9),
        }
    )
    del optimizer, model, memory
    clear_device_cache(device)
    return artifacts


# -----------------------------------------------------------------------------
# Summaries and paired comparisons
# -----------------------------------------------------------------------------


def build_run_summaries(
    eval_rows: Sequence[Mapping[str, Any]],
    task_end_rows: Sequence[Mapping[str, Any]],
    num_tasks: int,
    bootstrap_resamples: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    grouped: dict[tuple[int, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in eval_rows:
        grouped[(int(row["seed"]), str(row["method"]))].append(row)

    task_end_grouped: dict[tuple[int, str, int, int], float] = {}
    for row in task_end_rows:
        task_end_grouped[
            (
                int(row["seed"]),
                str(row["method"]),
                int(row["train_task"]),
                int(row["eval_task"]),
            )
        ] = float(row["accuracy"])

    summaries: list[dict[str, Any]] = []
    for (seed, method), rows in sorted(grouped.items()):
        final_step = max(int(r["global_step"]) for r in rows)
        final_by_task: dict[int, float] = {}
        maxima: dict[int, float] = defaultdict(float)
        for row in rows:
            task = int(row["eval_task"])
            acc = float(row["accuracy"])
            maxima[task] = max(maxima[task], acc)
            if int(row["global_step"]) == final_step:
                final_by_task[task] = acc
        final_values = [final_by_task.get(t, float("nan")) for t in range(num_tasks)]
        forgetting_values = [
            maxima[t] - final_by_task.get(t, maxima[t])
            for t in range(max(0, num_tasks - 1))
        ]
        diagonal = [
            task_end_grouped.get((seed, method, task, task), float("nan"))
            for task in range(num_tasks)
        ]
        summaries.append(
            {
                "seed": seed,
                "method": method,
                "final_step": final_step,
                "final_average_accuracy": float(np.nanmean(final_values)),
                "final_first_task_accuracy": final_by_task.get(0, float("nan")),
                "final_last_task_accuracy": final_by_task.get(
                    num_tasks - 1, float("nan")
                ),
                "average_forgetting": float(np.mean(forgetting_values))
                if forgetting_values
                else 0.0,
                "max_forgetting": float(np.max(forgetting_values))
                if forgetting_values
                else 0.0,
                "mean_task_acquisition": float(np.nanmean(diagonal)),
                "final_task_accuracies": final_values,
                "task_end_diagonal": diagonal,
            }
        )

    aggregate: list[dict[str, Any]] = []
    by_method: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in summaries:
        by_method[str(row["method"])].append(row)
    metrics = (
        "final_average_accuracy",
        "final_first_task_accuracy",
        "final_last_task_accuracy",
        "average_forgetting",
        "mean_task_acquisition",
    )
    for method, rows in sorted(by_method.items()):
        result: dict[str, Any] = {"method": method, "seeds": len(rows)}
        for metric in metrics:
            values = [float(r[metric]) for r in rows]
            lo, hi = bootstrap_ci(values, bootstrap_resamples, seed=917)
            result[f"{metric}_mean"] = float(np.mean(values))
            result[f"{metric}_std"] = (
                float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
            )
            result[f"{metric}_ci_low"] = lo
            result[f"{metric}_ci_high"] = hi
        aggregate.append(result)

    paired: list[dict[str, Any]] = []
    local_rows = {int(r["seed"]): r for r in summaries if r["method"] == "local_dpr"}
    for method, rows in sorted(by_method.items()):
        if method == "local_dpr":
            continue
        other_rows = {int(r["seed"]): r for r in rows}
        common = sorted(set(local_rows) & set(other_rows))
        for metric in metrics:
            differences = [
                float(local_rows[s][metric]) - float(other_rows[s][metric])
                for s in common
            ]
            if not differences:
                continue
            lo, hi = bootstrap_ci(differences, bootstrap_resamples, seed=1217)
            paired.append(
                {
                    "reference_method": "local_dpr",
                    "comparison_method": method,
                    "metric": metric,
                    "paired_seeds": len(common),
                    "mean_difference": float(np.mean(differences)),
                    "median_difference": float(np.median(differences)),
                    "ci_low": lo,
                    "ci_high": hi,
                    "differences_by_seed": {
                        str(s): differences[i] for i, s in enumerate(common)
                    },
                }
            )
    return summaries, aggregate, paired


# -----------------------------------------------------------------------------
# Visualizations
# -----------------------------------------------------------------------------


def _sorted_unique(rows: Sequence[Mapping[str, Any]], key: str) -> list[Any]:
    return sorted({row[key] for row in rows})


def plot_two_task_dashboard(
    eval_rows: Sequence[Mapping[str, Any]],
    controller_rows: Sequence[Mapping[str, Any]],
    output_dir: Path,
    formats: Sequence[str],
) -> None:
    seeds = _sorted_unique(eval_rows, "seed")
    if not seeds:
        return
    seed = seeds[0]
    methods = _sorted_unique([r for r in eval_rows if r["seed"] == seed], "method")
    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
    for method in methods:
        for eval_task, ax in ((0, axes[0]), (1, axes[1])):
            points = sorted(
                [
                    r
                    for r in eval_rows
                    if r["seed"] == seed
                    and r["method"] == method
                    and int(r["eval_task"]) == eval_task
                ],
                key=lambda r: int(r["global_step"]),
            )
            if points:
                ax.plot(
                    [int(r["global_step"]) for r in points],
                    [float(r["accuracy"]) for r in points],
                    label=method,
                )
    axes[0].set_ylabel("Task 0 accuracy")
    axes[1].set_ylabel("Task 1 accuracy")
    axes[0].set_title("Local DPR: Retention and Drift Adaptation")
    axes[0].legend(loc="best")
    axes[1].legend(loc="best")
    boundary_candidates = [
        int(r["global_step"])
        for r in eval_rows
        if r["seed"] == seed and r["event"] == "task_end" and int(r["train_task"]) == 0
    ]
    drift_step = min(boundary_candidates) if boundary_candidates else None
    if drift_step is not None:
        axes[0].axvline(drift_step, linestyle="--", alpha=0.7, label="task drift")
        axes[1].axvline(drift_step, linestyle="--", alpha=0.7)

    local = sorted(
        [
            r
            for r in controller_rows
            if r["seed"] == seed and r["method"] == "local_dpr"
        ],
        key=lambda r: (int(r["global_step"]), str(r["block"])),
    )
    for block in _sorted_unique(local, "block"):
        points = [r for r in local if r["block"] == block]
        axes[2].plot(
            [int(r["global_step"]) for r in points],
            [float(r["lambda"]) for r in points],
            label=str(block),
        )
    if drift_step is not None:
        axes[2].axvline(drift_step, linestyle="--", alpha=0.7)
    axes[2].set_ylabel("Local weight decay λ")
    axes[2].set_xlabel("Cumulative training steps")
    axes[2].legend(loc="best")
    for ax in axes:
        ax.grid(True, alpha=0.3)
    save_figure(fig, output_dir, "two_task_dashboard", formats)


def plot_accuracy_trajectories(
    eval_rows: Sequence[Mapping[str, Any]], output_dir: Path, formats: Sequence[str]
) -> None:
    for seed in _sorted_unique(eval_rows, "seed"):
        seed_rows = [r for r in eval_rows if r["seed"] == seed]
        for method in _sorted_unique(seed_rows, "method"):
            rows = [r for r in seed_rows if r["method"] == method]
            fig, ax = plt.subplots(figsize=(12, 6))
            for task in _sorted_unique(rows, "eval_task"):
                points = sorted(
                    [r for r in rows if r["eval_task"] == task],
                    key=lambda r: int(r["global_step"]),
                )
                ax.plot(
                    [int(r["global_step"]) for r in points],
                    [float(r["accuracy"]) for r in points],
                    label=f"task {task}",
                )
            ax.set_title(f"Accuracy trajectories — {method}, seed {seed}")
            ax.set_xlabel("Cumulative training steps")
            ax.set_ylabel("Test accuracy")
            ax.grid(True, alpha=0.3)
            ax.legend(loc="best")
            save_figure(
                fig, output_dir, f"accuracy_trajectories_seed-{seed}_{method}", formats
            )


def plot_controller_lines(
    controller_rows: Sequence[Mapping[str, Any]],
    output_dir: Path,
    formats: Sequence[str],
    field: str,
    ylabel: str,
    stem: str,
) -> None:
    local_rows = [r for r in controller_rows if r["method"] == "local_dpr"]
    for seed in _sorted_unique(local_rows, "seed"):
        rows = [r for r in local_rows if r["seed"] == seed]
        if not rows:
            continue
        fig, ax = plt.subplots(figsize=(12, 6))
        for block in _sorted_unique(rows, "block"):
            points = sorted(
                [r for r in rows if r["block"] == block],
                key=lambda r: int(r["global_step"]),
            )
            ax.plot(
                [int(r["global_step"]) for r in points],
                [float(r[field]) for r in points],
                label=str(block),
            )
        ax.set_title(f"{ylabel} — local_dpr, seed {seed}")
        ax.set_xlabel("Cumulative training steps")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")
        save_figure(fig, output_dir, f"{stem}_seed-{seed}", formats)


def plot_controller_heatmap(
    controller_rows: Sequence[Mapping[str, Any]],
    output_dir: Path,
    formats: Sequence[str],
    field: str,
    title: str,
    stem: str,
) -> None:
    local_rows = [r for r in controller_rows if r["method"] == "local_dpr"]
    for seed in _sorted_unique(local_rows, "seed"):
        rows = [r for r in local_rows if r["seed"] == seed]
        steps = _sorted_unique(rows, "global_step")
        blocks = _sorted_unique(rows, "block")
        if not steps or not blocks:
            continue
        lookup = {
            (int(r["global_step"]), str(r["block"])): float(r[field]) for r in rows
        }
        matrix = np.array(
            [
                [lookup.get((int(step), str(block)), np.nan) for step in steps]
                for block in blocks
            ]
        )
        fig, ax = plt.subplots(figsize=(13, max(3.5, len(blocks) * 0.8)))
        image = ax.imshow(matrix, aspect="auto", interpolation="nearest")
        ax.set_title(f"{title} — seed {seed}")
        ax.set_xlabel("Logged controller step")
        ax.set_ylabel("Block")
        ax.set_yticks(range(len(blocks)), labels=blocks)
        tick_positions = np.linspace(0, len(steps) - 1, min(8, len(steps))).astype(int)
        ax.set_xticks(tick_positions, labels=[steps[i] for i in tick_positions])
        fig.colorbar(image, ax=ax, label=field)
        save_figure(fig, output_dir, f"{stem}_seed-{seed}", formats)


def plot_task_end_matrices(
    task_end_rows: Sequence[Mapping[str, Any]], output_dir: Path, formats: Sequence[str]
) -> None:
    for seed in _sorted_unique(task_end_rows, "seed"):
        seed_rows = [r for r in task_end_rows if r["seed"] == seed]
        for method in _sorted_unique(seed_rows, "method"):
            rows = [r for r in seed_rows if r["method"] == method]
            train_tasks = _sorted_unique(rows, "train_task")
            eval_tasks = _sorted_unique(rows, "eval_task")
            lookup = {
                (int(r["train_task"]), int(r["eval_task"])): float(r["accuracy"])
                for r in rows
            }
            matrix = np.array(
                [
                    [
                        lookup.get((int(train_task), int(eval_task)), np.nan)
                        for eval_task in eval_tasks
                    ]
                    for train_task in train_tasks
                ]
            )
            fig, ax = plt.subplots(figsize=(7, 6))
            image = ax.imshow(
                matrix, vmin=0.0, vmax=1.0, aspect="auto", interpolation="nearest"
            )
            ax.set_title(f"Task-end accuracy matrix — {method}, seed {seed}")
            ax.set_xlabel("Evaluation task")
            ax.set_ylabel("After training task")
            ax.set_xticks(range(len(eval_tasks)), labels=eval_tasks)
            ax.set_yticks(range(len(train_tasks)), labels=train_tasks)
            for i in range(matrix.shape[0]):
                for j in range(matrix.shape[1]):
                    if np.isfinite(matrix[i, j]):
                        ax.text(
                            j,
                            i,
                            f"{matrix[i, j]:.2f}",
                            ha="center",
                            va="center",
                            fontsize=8,
                        )
            fig.colorbar(image, ax=ax, label="accuracy")
            save_figure(
                fig, output_dir, f"task_end_matrix_seed-{seed}_{method}", formats
            )


def plot_aggregate_bars(
    aggregate_rows: Sequence[Mapping[str, Any]],
    output_dir: Path,
    formats: Sequence[str],
) -> None:
    if not aggregate_rows:
        return
    methods = [str(r["method"]) for r in aggregate_rows]
    for metric, label, stem in (
        ("final_average_accuracy", "Final average accuracy", "final_average_accuracy"),
        (
            "final_first_task_accuracy",
            "Final first-task accuracy",
            "first_task_retention",
        ),
        ("average_forgetting", "Average forgetting", "average_forgetting"),
        ("mean_task_acquisition", "Mean task acquisition", "mean_task_acquisition"),
    ):
        means = np.array([float(r[f"{metric}_mean"]) for r in aggregate_rows])
        lows = np.array([float(r[f"{metric}_ci_low"]) for r in aggregate_rows])
        highs = np.array([float(r[f"{metric}_ci_high"]) for r in aggregate_rows])
        errors = np.vstack([means - lows, highs - means])
        fig, ax = plt.subplots(figsize=(11, 5.5))
        ax.bar(np.arange(len(methods)), means, yerr=errors, capsize=4)
        ax.set_xticks(np.arange(len(methods)), labels=methods, rotation=25, ha="right")
        ax.set_ylabel(label)
        ax.set_title(f"{label} across seeds")
        ax.grid(True, axis="y", alpha=0.3)
        save_figure(fig, output_dir, stem, formats)


def plot_retention_plasticity(
    summary_rows: Sequence[Mapping[str, Any]], output_dir: Path, formats: Sequence[str]
) -> None:
    if not summary_rows:
        return
    fig, ax = plt.subplots(figsize=(8, 7))
    methods = sorted({str(r["method"]) for r in summary_rows})
    for method in methods:
        rows = [r for r in summary_rows if r["method"] == method]
        x = [float(r["mean_task_acquisition"]) for r in rows]
        y = [float(r["final_first_task_accuracy"]) for r in rows]
        ax.scatter(x, y, label=method, s=55)
        ax.annotate(
            method,
            (float(np.mean(x)), float(np.mean(y))),
            xytext=(4, 4),
            textcoords="offset points",
        )
    ax.set_xlabel("Plasticity: mean task-end acquisition")
    ax.set_ylabel("Retention: final first-task accuracy")
    ax.set_title("Retention–plasticity map")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    save_figure(fig, output_dir, "retention_plasticity", formats)


def plot_decay_assignment_control(
    controller_rows: Sequence[Mapping[str, Any]],
    output_dir: Path,
    formats: Sequence[str],
) -> None:
    rows = [
        r for r in controller_rows if r["method"] in {"local_dpr", "local_dpr_permuted"}
    ]
    if not rows:
        return
    for seed in _sorted_unique(rows, "seed"):
        seed_rows = [r for r in rows if r["seed"] == seed]
        fig, ax = plt.subplots(figsize=(12, 6))
        for method in ("local_dpr", "local_dpr_permuted"):
            method_rows = [
                r
                for r in seed_rows
                if r["method"] == method and r["block"] == "classifier"
            ]
            method_rows = sorted(method_rows, key=lambda r: int(r["global_step"]))
            if method_rows:
                ax.plot(
                    [int(r["global_step"]) for r in method_rows],
                    [float(r["lambda"]) for r in method_rows],
                    label=f"{method}: classifier-applied λ",
                )
        ax.set_title(f"Spatial assignment control — seed {seed}")
        ax.set_xlabel("Cumulative training steps")
        ax.set_ylabel("Applied classifier decay λ")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")
        save_figure(fig, output_dir, f"assignment_control_seed-{seed}", formats)


def plot_timing(
    timing_rows: Sequence[Mapping[str, Any]], output_dir: Path, formats: Sequence[str]
) -> None:
    total_rows = [
        r for r in timing_rows if int(r["task"]) == -1 and int(r["epoch"]) == -1
    ]
    if not total_rows:
        return
    methods = sorted({str(r["method"]) for r in total_rows})
    means = []
    for method in methods:
        values = [
            float(r["examples_per_second"]) for r in total_rows if r["method"] == method
        ]
        means.append(float(np.mean(values)))
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(np.arange(len(methods)), means)
    ax.set_xticks(np.arange(len(methods)), labels=methods, rotation=25, ha="right")
    ax.set_ylabel("Training examples / second")
    ax.set_title("End-to-end training throughput")
    ax.grid(True, axis="y", alpha=0.3)
    save_figure(fig, output_dir, "training_throughput", formats)


def generate_plots(
    args: argparse.Namespace,
    all_artifacts: RunArtifacts,
    summary_rows: Sequence[Mapping[str, Any]],
    aggregate_rows: Sequence[Mapping[str, Any]],
) -> None:
    plot_dir = args.output_dir / "figures"
    if args.num_tasks == 2:
        plot_two_task_dashboard(
            all_artifacts.eval_rows,
            all_artifacts.controller_rows,
            plot_dir,
            args.plot_format,
        )
    plot_accuracy_trajectories(all_artifacts.eval_rows, plot_dir, args.plot_format)
    plot_controller_lines(
        all_artifacts.controller_rows,
        plot_dir,
        args.plot_format,
        "lambda",
        "Weight decay λ",
        "local_lambda",
    )
    plot_controller_lines(
        all_artifacts.controller_rows,
        plot_dir,
        args.plot_format,
        "gradient_conflict",
        "Gradient conflict",
        "gradient_conflict",
    )
    plot_controller_lines(
        all_artifacts.controller_rows,
        plot_dir,
        args.plot_format,
        "old_task_importance",
        "Old-task gradient importance",
        "old_task_importance",
    )
    plot_controller_lines(
        all_artifacts.controller_rows,
        plot_dir,
        args.plot_format,
        "local_allocation",
        "Allocated forgetting fraction",
        "local_allocation",
    )
    plot_controller_heatmap(
        all_artifacts.controller_rows,
        plot_dir,
        args.plot_format,
        "gradient_conflict",
        "Block-local gradient conflict",
        "gradient_conflict_heatmap",
    )
    plot_controller_heatmap(
        all_artifacts.controller_rows,
        plot_dir,
        args.plot_format,
        "lambda",
        "Block-local applied decay",
        "local_decay_heatmap",
    )
    plot_task_end_matrices(all_artifacts.task_end_rows, plot_dir, args.plot_format)
    plot_aggregate_bars(aggregate_rows, plot_dir, args.plot_format)
    plot_retention_plasticity(summary_rows, plot_dir, args.plot_format)
    plot_decay_assignment_control(
        all_artifacts.controller_rows, plot_dir, args.plot_format
    )
    plot_timing(all_artifacts.timing_rows, plot_dir, args.plot_format)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def validate_args(args: argparse.Namespace) -> None:
    if args.num_tasks < 1:
        raise ValueError("--num-tasks must be at least 1")
    if args.epochs_per_task < 1:
        raise ValueError("--epochs-per-task must be at least 1")
    if not args.hidden_dims or any(dim <= 0 for dim in args.hidden_dims):
        raise ValueError("--hidden-dims must contain positive integers")
    if args.lambda_low < 0 or args.lambda_high < args.lambda_low:
        raise ValueError("Require 0 <= lambda-low <= lambda-high")
    if not 0 <= args.local_floor <= 1:
        raise ValueError("--local-floor must be in [0,1]")
    if not 0 < args.surprise_beta < 1:
        raise ValueError("--surprise-beta must be in (0,1)")
    if args.reference_refresh_steps < 1:
        raise ValueError("--reference-refresh-steps must be >= 1")
    if args.mps_memory_fraction and not 0 < args.mps_memory_fraction <= 2:
        raise ValueError("--mps-memory-fraction must be 0 or in (0,2]")


def merge_artifacts(target: RunArtifacts, source: RunArtifacts) -> None:
    target.train_rows.extend(source.train_rows)
    target.controller_rows.extend(source.controller_rows)
    target.eval_rows.extend(source.eval_rows)
    target.task_end_rows.extend(source.task_end_rows)
    target.memory_rows.extend(source.memory_rows)
    target.timing_rows.extend(source.timing_rows)


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    validate_args(args)

    if not args.mps_fallback:
        os.environ.pop("PYTORCH_ENABLE_MPS_FALLBACK", None)
    device = resolve_device(args.device)
    if device.type == "mps" and args.mps_memory_fraction > 0:
        setter = getattr(torch.mps, "set_per_process_memory_fraction", None)
        if callable(setter):
            setter(args.mps_memory_fraction)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = vars(args).copy()
    config["resolved_device"] = str(device)
    config["torch_version"] = torch.__version__
    config["mps_built"] = bool(torch.backends.mps.is_built())
    config["mps_available"] = bool(torch.backends.mps.is_available())
    config["device_memory_at_start"] = device_memory_snapshot(device)
    write_json(args.output_dir / "config.json", config)

    print(json.dumps(to_jsonable(config), indent=2, sort_keys=True))
    if args.dry_run:
        print("Dry run complete; no dataset or model was instantiated.")
        return 0

    tasks = build_tasks(args)
    input_dim = int(tasks[0].permutation.numel())
    all_artifacts = RunArtifacts()

    for seed in args.seeds:
        set_seed(seed)
        template_model = MultiBlockMLP(input_dim, args.hidden_dims)
        initial_state = {
            name: tensor.detach().clone()
            for name, tensor in template_model.state_dict().items()
        }
        del template_model
        for method in args.methods:
            print(f"\n--- Running {method} | seed={seed} | device={device} ---")
            artifacts = run_single_method(
                args,
                tasks,
                device,
                seed,
                method,
                initial_state,
                args.output_dir,
            )
            merge_artifacts(all_artifacts, artifacts)

    summary_rows, aggregate_rows, paired_rows = build_run_summaries(
        all_artifacts.eval_rows,
        all_artifacts.task_end_rows,
        args.num_tasks,
        args.bootstrap_resamples,
    )

    write_dual(args.output_dir, "train_metrics", all_artifacts.train_rows)
    write_dual(args.output_dir, "controller_metrics", all_artifacts.controller_rows)
    write_dual(args.output_dir, "evaluation_metrics", all_artifacts.eval_rows)
    write_dual(args.output_dir, "task_end_accuracy", all_artifacts.task_end_rows)
    write_dual(args.output_dir, "gradient_reference_memory", all_artifacts.memory_rows)
    write_dual(args.output_dir, "timing", all_artifacts.timing_rows)
    write_dual(args.output_dir, "run_summary", summary_rows)
    write_dual(args.output_dir, "aggregate_summary", aggregate_rows)
    write_dual(args.output_dir, "paired_comparisons", paired_rows)

    generate_plots(args, all_artifacts, summary_rows, aggregate_rows)

    final_metadata = {
        "completed": True,
        "resolved_device": str(device),
        "device_memory_at_end": device_memory_snapshot(device),
        "records": {
            "train": len(all_artifacts.train_rows),
            "controller": len(all_artifacts.controller_rows),
            "evaluation": len(all_artifacts.eval_rows),
            "task_end": len(all_artifacts.task_end_rows),
            "memory": len(all_artifacts.memory_rows),
        },
    }
    write_json(args.output_dir / "run_complete.json", final_metadata)

    print("\n=== Aggregate summary ===")
    for row in aggregate_rows:
        print(
            f"{row['method']:20s} | final avg={row['final_average_accuracy_mean']:.4f} "
            f"first retention={row['final_first_task_accuracy_mean']:.4f} "
            f"forgetting={row['average_forgetting_mean']:.4f} "
            f"acquisition={row['mean_task_acquisition_mean']:.4f}"
        )
    print(f"\nOutputs written to: {args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        raise SystemExit(130)
