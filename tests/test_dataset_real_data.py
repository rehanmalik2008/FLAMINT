"""Reconciliation tests for FilamentDataset against real MAGFiLO images.

Unlike test_dataset.py (synthetic JPEGs), this runs the actual FilamentDataset
against the real annotation file and the real downloaded training photos --
the second half of P0's data audit gate, completing what
test_coco_real_data.py started for annotations alone. Skipped automatically
if the images are not present.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from filament.data.coco import load_magfilo
from filament.data.dataset import FilamentDataset

ROOT = Path(__file__).resolve().parents[1]
JSON_PATH = ROOT / "data" / "MAGFiLO_1.0_Annotations_kaggle2026_train.json"
IMAGE_DIR = ROOT / "data" / "train_images"

pytestmark = pytest.mark.skipif(
    not (JSON_PATH.exists() and IMAGE_DIR.exists() and any(IMAGE_DIR.iterdir())),
    reason=f"real training images not present at {IMAGE_DIR}",
)


@pytest.fixture(scope="module")
def real_dataset_and_ids():
    ds = load_magfilo(JSON_PATH)
    available = {p.name for p in IMAGE_DIR.iterdir()}
    usable_ids = [iid for iid, rec in ds.images.items() if rec.file_name in available]
    return ds, usable_ids


# --------------------------------------------------------------------------
# Coverage
# --------------------------------------------------------------------------


def test_every_annotated_image_has_a_downloaded_file(real_dataset_and_ids):
    """Confirms the earlier audit finding: 707 unique files serve 1,154
    annotation rows because duplicate re-annotations share one file_name."""
    ds, usable_ids = real_dataset_and_ids
    assert len(usable_ids) == len(ds)

    unique_file_names = {rec.file_name for rec in ds.images.values()}
    n_files_on_disk = len(list(IMAGE_DIR.iterdir()))
    assert len(unique_file_names) == n_files_on_disk == 707


# --------------------------------------------------------------------------
# Real image loading and target generation
# --------------------------------------------------------------------------


def test_loads_a_real_sample_with_correct_shapes(real_dataset_and_ids):
    ds, usable_ids = real_dataset_and_ids
    dataset = FilamentDataset(ds, IMAGE_DIR, usable_ids[:5])
    for i in range(len(dataset)):
        sample = dataset[i]
        assert sample.image.shape == (1, 2048, 2048)
        assert sample.mask.shape == (1, 2048, 2048)
        assert sample.spine.shape == (1, 2048, 2048)
        assert sample.offsets.shape == (2, 2048, 2048)
        assert torch.isfinite(sample.image).all()
        assert torch.isfinite(sample.offsets).all()


def test_real_images_are_normalised_to_unit_range(real_dataset_and_ids):
    ds, usable_ids = real_dataset_and_ids
    dataset = FilamentDataset(ds, IMAGE_DIR, usable_ids[:10])
    for i in range(len(dataset)):
        img = dataset[i].image
        assert img.min() >= 0.0
        assert img.max() <= 1.0


def test_real_annotated_images_have_nonzero_mask(real_dataset_and_ids):
    """Every image in this file has at least one filament annotation (by
    construction of the dataset), so the mask must never be empty here."""
    ds, usable_ids = real_dataset_and_ids
    dataset = FilamentDataset(ds, IMAGE_DIR, usable_ids[:15])
    for i in range(len(dataset)):
        sample = dataset[i]
        assert sample.mask.sum() > 0, f"{sample.image_id} has an empty mask"


def test_mask_foreground_fraction_is_small_and_plausible(real_dataset_and_ids):
    """Sanity range check against the P0 audit's measured mean GT area
    (~2,120px per filament) -- a full 2048x2048 frame is ~4.19M px, so even
    a frame with several filaments should be well under 1% foreground."""
    ds, usable_ids = real_dataset_and_ids
    dataset = FilamentDataset(ds, IMAGE_DIR, usable_ids[:20])
    fractions = []
    for i in range(len(dataset)):
        sample = dataset[i]
        fractions.append(float(sample.mask.mean()))
    assert max(fractions) < 0.05, "an implausibly large fraction of a frame is foreground"


# --------------------------------------------------------------------------
# Geometry channels on real images
# --------------------------------------------------------------------------


def test_disk_fit_is_sane_on_real_images(real_dataset_and_ids):
    """The fitted disk centre must land near the frame centre and the radius
    must be a substantial, plausible fraction of the frame -- not a
    degenerate near-zero or near-infinite fit."""
    from filament.geometry.disk import fit_disk

    ds, usable_ids = real_dataset_and_ids
    from PIL import Image

    for image_id in usable_ids[:8]:
        file_name = ds.images[image_id].file_name
        arr = np.asarray(Image.open(IMAGE_DIR / file_name).convert("L"), dtype=np.float32) / 255.0
        is_bright = arr > np.percentile(arr, 60)
        geo = fit_disk(is_bright)

        h, w = arr.shape
        assert abs(geo.center_row - h / 2) < 150, "fitted centre far from frame centre"
        assert abs(geo.center_col - w / 2) < 150
        assert 0.25 * min(h, w) < geo.radius < 0.55 * min(h, w)


def test_geometry_channels_run_on_real_images(real_dataset_and_ids):
    ds, usable_ids = real_dataset_and_ids
    dataset = FilamentDataset(ds, IMAGE_DIR, usable_ids[:5], include_geometry_channels=True)
    for i in range(len(dataset)):
        sample = dataset[i]
        assert sample.image.shape == (3, 2048, 2048)
        assert torch.isfinite(sample.image).all()
        # r/R_sun channel should span a real range, not be constant (which
        # would indicate the disk fit silently fell back to the degenerate
        # frame-centred default on every image).
        r_channel = sample.image[1]
        assert r_channel.max() - r_channel.min() > 0.3
