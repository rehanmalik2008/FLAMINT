"""The training entry point.

Reads cached targets (``scripts/precompute_targets.py`` must be run first),
splits leak-free by time (``filament.data.splits``), trains
``FilamentUNet`` with ``CombinedLoss``, and checkpoints the best-val-loss
model.

Deliberately simple for a first working run: no LR schedule beyond a basic
step decay, no mixed precision, no augmentation yet. The goal of this first
version is an end-to-end proof that the loop trains and loss decreases,
runnable on this laptop's CPU at tiny scale to verify correctness before the
same script (with more epochs, a larger base_channels, and a CUDA device) is
handed to a Kaggle GPU notebook for the real run. Every one of those upgrades
is a config change, not a rewrite -- see TrainConfig.

Usage
-----
    python -m filament.train --config configs/smoke_test.yaml
    python -m filament.train --config configs/baseline.yaml

CPU MEMORY NOTE (found during smoke testing on the reference dev laptop):
FilamentUNet at 1024x1024 input can trigger a hard OS-level crash (Windows
access violation / SIGSEGV, not a clean MemoryError) rather than a graceful
out-of-memory exception, when base_channels x batch_size x resolution exceeds
available RAM -- e.g. base_channels=16 at batch_size=2 crashed on an 8GB
laptop where base_channels=8 at the same batch_size did not. This is a
platform/allocator characteristic (observed identically with
OMP_NUM_THREADS=1), not a bug in this codebase's logic -- confirmed by
bisecting purely on model width and batch size with everything else held
fixed. configs/smoke_test.yaml and smoke_test_medium.yaml are sized to stay
well under this ceiling; configs/baseline.yaml targets a CUDA device (Kaggle),
where GPU memory behaves differently and this ceiling does not directly apply
-- but if a CUDA OOM is hit there, reduce batch_size or base_channels first.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from filament.data.coco import load_magfilo
from filament.data.dataset_cached import CachedFilamentDataset
from filament.data.splits import assert_no_leakage, time_grouped_split
from filament.losses.segmentation import CombinedLoss
from filament.models.unet import FilamentUNet

__all__ = ["TrainConfig", "train"]


@dataclass
class TrainConfig:
    json_path: str = "data/MAGFiLO_1.0_Annotations_kaggle2026_train.json"
    cache_dir: str = "data/cache"
    checkpoint_dir: str = "outputs/checkpoints"
    val_fraction: float = 0.15
    test_fraction: float = 0.15
    base_channels: int = 32
    batch_size: int = 4
    epochs: int = 20
    lr: float = 1e-3
    lr_step_epochs: int = 10
    lr_gamma: float = 0.5
    num_workers: int = 0
    device: str = "cpu"
    limit_train: int | None = None  # debug: cap training set size
    limit_val: int | None = None
    seed: int = 0

    @classmethod
    def from_yaml(cls, path: str | Path) -> "TrainConfig":
        import yaml  # local import: only needed for this convenience path

        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        return cls(**raw)


def build_dataloaders(cfg: TrainConfig) -> tuple[DataLoader, DataLoader, dict]:
    dataset = load_magfilo(cfg.json_path)
    assignment = time_grouped_split(
        dataset, val_fraction=cfg.val_fraction, test_fraction=cfg.test_fraction
    )
    assert_no_leakage(dataset, assignment)

    train_ids = assignment.image_ids("train")
    val_ids = assignment.image_ids("val")
    if cfg.limit_train:
        train_ids = train_ids[: cfg.limit_train]
    if cfg.limit_val:
        val_ids = val_ids[: cfg.limit_val]

    train_ds = CachedFilamentDataset(cfg.cache_dir, train_ids)
    val_ds = CachedFilamentDataset(cfg.cache_dir, val_ids)

    # collate_fn=list keeps each batch as a plain list of FilamentSample
    # (dataclasses, not tensors, so torch's default collate can't stack them
    # directly); _collate_batch does the actual stacking after the DataLoader
    # hands the list back.
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        collate_fn=list,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        collate_fn=list,
    )

    info = {
        "n_train": len(train_ids),
        "n_val": len(val_ids),
        "n_test_held_out": len(assignment.image_ids("test")),
    }
    return train_loader, val_loader, info


def _collate_batch(samples) -> dict[str, torch.Tensor]:
    return {
        "image": torch.stack([s.image for s in samples]),
        "mask": torch.stack([s.mask for s in samples]),
        "spine": torch.stack([s.spine for s in samples]),
        "offsets": torch.stack([s.offsets for s in samples]),
    }


def run_epoch(
    model: FilamentUNet,
    loader: DataLoader,
    criterion: CombinedLoss,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
) -> dict[str, float]:
    """One pass over `loader`. Training if `optimizer` is given, else eval."""
    is_train = optimizer is not None
    model.train(is_train)

    totals = {"mask": 0.0, "spine": 0.0, "offset": 0.0, "total": 0.0}
    n_batches = 0

    for samples in loader:
        batch = _collate_batch(samples)
        image = batch["image"].to(device)
        mask_target = batch["mask"].to(device)
        spine_target = batch["spine"].to(device)
        offset_target = batch["offsets"].to(device)

        with torch.set_grad_enabled(is_train):
            out = model(image)
            losses = criterion(
                out.mask_logits, out.spine_logits, out.offsets,
                mask_target, spine_target, offset_target,
            )
            if is_train:
                optimizer.zero_grad()
                losses["total"].backward()
                optimizer.step()

        for k in totals:
            totals[k] += float(losses[k].detach())
        n_batches += 1

    return {k: v / max(n_batches, 1) for k, v in totals.items()}


def train(cfg: TrainConfig) -> dict:
    torch.manual_seed(cfg.seed)
    device = torch.device(cfg.device)

    train_loader, val_loader, info = build_dataloaders(cfg)

    print(f"train={info['n_train']} val={info['n_val']} test_held_out={info['n_test_held_out']}")

    model = FilamentUNet(in_channels=1, base_channels=cfg.base_channels).to(device)
    criterion = CombinedLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=cfg.lr_step_epochs, gamma=cfg.lr_gamma
    )

    checkpoint_dir = Path(cfg.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    history = []
    best_val_loss = float("inf")

    for epoch in range(cfg.epochs):
        t0 = time.time()
        train_metrics = run_epoch(model, train_loader, criterion, device, optimizer)
        val_metrics = run_epoch(model, val_loader, criterion, device, None)
        scheduler.step()
        elapsed = time.time() - t0

        row = {
            "epoch": epoch,
            "train": train_metrics,
            "val": val_metrics,
            "lr": optimizer.param_groups[0]["lr"],
            "elapsed_s": elapsed,
        }
        history.append(row)
        print(
            f"epoch {epoch:3d}  train_loss={train_metrics['total']:.4f}  "
            f"val_loss={val_metrics['total']:.4f}  lr={row['lr']:.2e}  "
            f"({elapsed:.1f}s)"
        )

        if val_metrics["total"] < best_val_loss:
            best_val_loss = val_metrics["total"]
            torch.save(
                {"model_state": model.state_dict(), "config": asdict(cfg), "epoch": epoch},
                checkpoint_dir / "best.pt",
            )

    with open(checkpoint_dir / "history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    return {"history": history, "best_val_loss": best_val_loss, "info": info}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None)
    args = parser.parse_args()

    cfg = TrainConfig.from_yaml(args.config) if args.config else TrainConfig()
    train(cfg)


if __name__ == "__main__":
    main()
