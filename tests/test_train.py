"""Tests for the training loop's plumbing, using tiny synthetic cached data.

Does not test real convergence quality (that needs real data and a real
training budget -- see the smoke-test config for the manual end-to-end check
against real cached data). These tests check the loop itself is wired
correctly: a forward/backward pass runs, loss decreases over a few steps on
a trivially learnable synthetic case, checkpoints save and are loadable, and
the leakage guard actually fires during training setup.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

torch = pytest.importorskip("torch")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from precompute_targets import process_one  # noqa: E402

from filament.data.coco import FilamentAnnotation, ImageRecord, MagfiloDataset  # noqa: E402
from filament.train import TrainConfig, build_dataloaders, run_epoch, train  # noqa: E402
from filament.models.unet import FilamentUNet  # noqa: E402
from filament.losses.segmentation import CombinedLoss  # noqa: E402


def rect_polygon(r0, r1, c0, c1) -> list[list[float]]:
    return [[c0, r0, c1, r0, c1, r1, c0, r1]]


def make_tiny_project(tmp_path, n_images=12, size=48):
    """A small synthetic annotation JSON + matching precomputed cache, so
    the training loop can run end to end without any real data or network
    access. Each image gets a distinct fake timestamp so the time-grouped
    split has something meaningful to divide."""
    json_dir = tmp_path / "raw"
    json_dir.mkdir()
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()

    images_json = []
    annotations_json = []
    rng = np.random.default_rng(0)

    for i in range(n_images):
        image_id = f"img-{i:04d}"
        ts = f"2015{(i % 12) + 1:02d}{(i % 27) + 1:02d}120000"
        file_name = f"{ts}Xh_{i}.jpg"

        images_json.append(
            {
                "id": image_id,
                "file_name": file_name,
                "height": size,
                "width": size,
                "date_captured": f"2015-{(i % 12) + 1:02d}-{(i % 27) + 1:02d} 12:00:00",
            }
        )
        seg = rect_polygon(10, 20, 10, 40)
        annotations_json.append(
            {
                "id": f"ann-{i}",
                "image_id": image_id,
                "category_id": 1,
                "segmentation": seg,
                "bbox": [10, 10, 30, 10],
            }
        )

        # Build the corresponding cache entry directly (bypassing JPEG I/O
        # for speed -- process_one still needs a real file on disk, though,
        # since it opens the path).
        arr = (rng.random((size, size)) * 255).astype(np.uint8)
        img_path = json_dir / file_name
        Image.fromarray(arr, mode="L").save(img_path)
        data = process_one(img_path, [seg], size, size, size)
        np.savez_compressed(cache_dir / f"{image_id}.npz", **data)

    import json

    payload = {
        "info": {},
        "licenses": {},
        "categories": [{"id": 1, "name": "Left"}],
        "images": images_json,
        "annotations": annotations_json,
    }
    json_path = tmp_path / "annotations.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f)

    return json_path, cache_dir


# --------------------------------------------------------------------------
# build_dataloaders
# --------------------------------------------------------------------------


def test_build_dataloaders_respects_limits(tmp_path):
    json_path, cache_dir = make_tiny_project(tmp_path, n_images=12)
    cfg = TrainConfig(
        json_path=str(json_path),
        cache_dir=str(cache_dir),
        val_fraction=0.2,
        test_fraction=0.2,
        limit_train=3,
        limit_val=2,
    )
    train_loader, val_loader, info = build_dataloaders(cfg)
    assert len(train_loader.dataset) == 3
    assert len(val_loader.dataset) == 2
    assert info["n_test_held_out"] > 0


def test_build_dataloaders_is_leak_free(tmp_path):
    """build_dataloaders must not raise -- it calls assert_no_leakage
    internally, so this passing at all is the leakage check."""
    json_path, cache_dir = make_tiny_project(tmp_path, n_images=12)
    cfg = TrainConfig(json_path=str(json_path), cache_dir=str(cache_dir))
    build_dataloaders(cfg)  # must not raise


# --------------------------------------------------------------------------
# run_epoch
# --------------------------------------------------------------------------


def test_run_epoch_train_mode_updates_weights(tmp_path):
    json_path, cache_dir = make_tiny_project(tmp_path, n_images=6, size=32)
    cfg = TrainConfig(
        json_path=str(json_path), cache_dir=str(cache_dir),
        limit_train=6, base_channels=8,
    )
    train_loader, _, _ = build_dataloaders(cfg)

    model = FilamentUNet(in_channels=1, base_channels=8)
    criterion = CombinedLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

    before = model.mask_head.weight.clone()
    run_epoch(model, train_loader, criterion, torch.device("cpu"), optimizer)
    after = model.mask_head.weight

    assert not torch.allclose(before, after), "weights did not change after a training epoch"


def test_run_epoch_eval_mode_does_not_update_weights(tmp_path):
    json_path, cache_dir = make_tiny_project(tmp_path, n_images=6, size=32)
    cfg = TrainConfig(json_path=str(json_path), cache_dir=str(cache_dir), limit_train=6)
    train_loader, _, _ = build_dataloaders(cfg)

    model = FilamentUNet(in_channels=1, base_channels=8)
    criterion = CombinedLoss()

    before = model.mask_head.weight.clone()
    run_epoch(model, train_loader, criterion, torch.device("cpu"), optimizer=None)
    after = model.mask_head.weight

    assert torch.allclose(before, after), "eval-mode epoch must not update weights"


def test_run_epoch_returns_finite_losses(tmp_path):
    json_path, cache_dir = make_tiny_project(tmp_path, n_images=6, size=32)
    cfg = TrainConfig(json_path=str(json_path), cache_dir=str(cache_dir), limit_train=6)
    train_loader, _, _ = build_dataloaders(cfg)
    model = FilamentUNet(in_channels=1, base_channels=8)
    criterion = CombinedLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    metrics = run_epoch(model, train_loader, criterion, torch.device("cpu"), optimizer)
    for k, v in metrics.items():
        assert np.isfinite(v), f"{k} is not finite: {v}"


# --------------------------------------------------------------------------
# train() end to end
# --------------------------------------------------------------------------


def test_train_loss_decreases_on_trivially_learnable_case(tmp_path):
    """The strongest correctness signal available without real data: train
    for a handful of epochs on a small, consistent synthetic case (every
    image has the identical rectangle in the identical place) and confirm
    the loss actually goes down. If the loop were wired wrong (e.g. loss not
    connected to the right output, or gradients not flowing), this would
    very likely fail to improve."""
    json_path, cache_dir = make_tiny_project(tmp_path, n_images=8, size=32)
    cfg = TrainConfig(
        json_path=str(json_path),
        cache_dir=str(cache_dir),
        checkpoint_dir=str(tmp_path / "ckpt"),
        val_fraction=0.25,
        test_fraction=0.25,
        base_channels=8,
        batch_size=2,
        epochs=6,
        lr=5e-3,
        limit_train=4,
        limit_val=2,
    )
    result = train(cfg)
    losses = [row["train"]["total"] for row in result["history"]]
    assert losses[-1] < losses[0], f"loss did not decrease: {losses}"


def test_train_saves_a_loadable_checkpoint(tmp_path):
    json_path, cache_dir = make_tiny_project(tmp_path, n_images=6, size=32)
    ckpt_dir = tmp_path / "ckpt"
    cfg = TrainConfig(
        json_path=str(json_path),
        cache_dir=str(cache_dir),
        checkpoint_dir=str(ckpt_dir),
        val_fraction=0.3,
        test_fraction=0.2,
        base_channels=8,
        batch_size=2,
        epochs=2,
        limit_train=3,
        limit_val=2,
    )
    train(cfg)

    ckpt_path = ckpt_dir / "best.pt"
    assert ckpt_path.exists()

    checkpoint = torch.load(ckpt_path, weights_only=False)
    model = FilamentUNet(in_channels=1, base_channels=8)
    model.load_state_dict(checkpoint["model_state"])  # must not raise
    assert checkpoint["config"]["base_channels"] == 8


def test_train_writes_history_json(tmp_path):
    json_path, cache_dir = make_tiny_project(tmp_path, n_images=6, size=32)
    ckpt_dir = tmp_path / "ckpt"
    cfg = TrainConfig(
        json_path=str(json_path),
        cache_dir=str(cache_dir),
        checkpoint_dir=str(ckpt_dir),
        val_fraction=0.3,
        test_fraction=0.2,
        base_channels=8,
        epochs=2,
        limit_train=3,
        limit_val=2,
    )
    train(cfg)
    assert (ckpt_dir / "history.json").exists()


def test_config_round_trips_through_yaml(tmp_path):
    cfg = TrainConfig(epochs=3, base_channels=16, lr=0.005)
    import yaml

    path = tmp_path / "cfg.yaml"
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(vars(cfg), f)

    loaded = TrainConfig.from_yaml(path)
    assert loaded.epochs == 3
    assert loaded.base_channels == 16
    assert loaded.lr == 0.005
