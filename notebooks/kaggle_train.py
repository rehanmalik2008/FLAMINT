"""Entry point for a real training run on a Kaggle GPU notebook.

Run on this laptop: proof-of-correctness only (base_channels=8, 60 images,
CPU). Real scale happens here, on Kaggle's free T4/P100:
  - full ~1,150-image training set, not a 60-image slice
  - base_channels=32 (the model's actual documented default)
  - many more epochs, GPU-speed (expect ~50-100x this laptop's throughput,
    based on the ~200s/60-images-per-epoch measured on CPU)

Usage inside a Kaggle notebook cell:
    !git clone https://github.com/rehanmalik2008/FLAMINT.git
    %cd FLAMINT
    !pip install -e . --quiet
    !python scripts/precompute_targets.py --resolution 1024
    !python notebooks/kaggle_train.py

Data setup: attach the competition as a Kaggle input dataset (Add Data ->
filament-segmentation-2026), then symlink/copy its train_images/ and the
annotation JSON into data/ before running precompute_targets.py -- see the
DATA_SETUP note below, filled in with the actual Kaggle input path once the
notebook is created (Kaggle mounts inputs under /kaggle/input/<slug>/).
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch  # noqa: E402

from filament.train import TrainConfig, train  # noqa: E402

# --- DATA_SETUP -------------------------------------------------------
# Adjust this path to match the actual Kaggle input mount once the dataset
# is attached to the notebook (typically /kaggle/input/<dataset-slug>/).
KAGGLE_INPUT_ROOT = Path("/kaggle/input/filament-segmentation-2026")
LOCAL_DATA_DIR = ROOT / "data"


def link_kaggle_data() -> None:
    """Point data/ at the Kaggle-mounted competition files, if present.

    Falls back silently if the input isn't mounted (e.g. when this script is
    imported/tested outside a Kaggle notebook) -- the caller is then
    responsible for having data/ populated some other way (as on the local
    dev laptop, where the data was downloaded and extracted manually).
    """
    if not KAGGLE_INPUT_ROOT.exists():
        print(f"Kaggle input not found at {KAGGLE_INPUT_ROOT}; assuming data/ "
              "is already populated (e.g. local dev setup).")
        return

    LOCAL_DATA_DIR.mkdir(exist_ok=True)
    json_src = next(KAGGLE_INPUT_ROOT.rglob("*_train.json"))
    images_src = next(p for p in KAGGLE_INPUT_ROOT.rglob("train_images") if p.is_dir())

    json_dst = LOCAL_DATA_DIR / json_src.name
    images_dst = LOCAL_DATA_DIR / "train_images"

    if not json_dst.exists():
        shutil.copy(json_src, json_dst)
    if not images_dst.exists():
        images_dst.symlink_to(images_src, target_is_directory=True)

    print(f"linked annotations -> {json_dst}")
    print(f"linked train_images -> {images_dst} -> {images_src}")


def main() -> None:
    link_kaggle_data()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print(
            "WARNING: no CUDA device found. This script is meant to run on a "
            "Kaggle GPU notebook (enable GPU under Notebook Settings). "
            "Falling back to CPU will be extremely slow at this scale."
        )

    # Find whatever the real annotation file is named locally (matches the
    # pattern used by link_kaggle_data / the manually-downloaded local copy).
    candidates = list(LOCAL_DATA_DIR.glob("*_train.json"))
    if not candidates:
        raise FileNotFoundError(
            f"no *_train.json found under {LOCAL_DATA_DIR} -- attach the "
            "competition dataset or populate data/ manually first"
        )
    json_path = candidates[0]

    cfg = TrainConfig(
        json_path=str(json_path),
        cache_dir=str(LOCAL_DATA_DIR / "cache"),
        checkpoint_dir=str(ROOT / "outputs" / "baseline_kaggle"),
        val_fraction=0.15,
        test_fraction=0.15,
        base_channels=32,
        batch_size=8,
        epochs=60,
        lr=1e-3,
        lr_step_epochs=20,
        lr_gamma=0.5,
        num_workers=2,
        device=device,
        seed=0,
    )

    print(f"training with config: {cfg}")
    result = train(cfg)
    print(f"best val loss: {result['best_val_loss']:.4f}")
    print(f"checkpoint saved to: {cfg.checkpoint_dir}/best.pt")


if __name__ == "__main__":
    main()
