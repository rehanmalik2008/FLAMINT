"""Reconciliation tests against the real competition annotation file.

Unlike every other test in this suite, these run against actual MAGFiLO data
rather than a synthetic fixture -- they are P0's data-audit gate. Every
assumption baked into `filament.data.coco` and `filament.data.splits` (ID
format, field names, the duplicate-observation mechanism, polygon area
consistency) is checked here against ground truth, not against a fixture we
wrote ourselves.

Skipped automatically if the data file is not present (e.g. in a CI
environment without competition data access), so the rest of the suite still
runs everywhere -- but this file must pass locally before any training run is
trusted.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from filament.data.coco import load_magfilo, polygon_to_mask
from filament.data.splits import assert_no_leakage, time_grouped_split

DATA_PATH = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "MAGFiLO_1.0_Annotations_kaggle2026_train.json"
)

pytestmark = pytest.mark.skipif(
    not DATA_PATH.exists(), reason=f"real data not present at {DATA_PATH}"
)


@pytest.fixture(scope="module")
def real_dataset():
    return load_magfilo(DATA_PATH)


# --------------------------------------------------------------------------
# Structural sanity
# --------------------------------------------------------------------------


def test_loads_without_error(real_dataset):
    assert len(real_dataset) > 0
    assert real_dataset.n_annotations() > 0


def test_image_and_annotation_counts_match_known_audit_numbers(real_dataset):
    """Pinned to the counts confirmed during the P0 data audit (2026-08-25).
    A change here on a re-download means the file was updated upstream --
    worth knowing, not silently absorbing."""
    assert len(real_dataset) == 1154
    assert real_dataset.n_annotations() == 8199


def test_every_annotation_references_a_real_image(real_dataset):
    for img_id, anns in real_dataset.annotations_by_image.items():
        assert img_id in real_dataset.images
        for ann in anns:
            assert ann.image_id == img_id


def test_image_dimensions_are_the_documented_2048_square(real_dataset):
    dims = {(img.height, img.width) for img in real_dataset.images.values()}
    assert dims == {(2048, 2048)}


def test_date_captured_parses_for_every_image(real_dataset):
    """If this fails, the observation_key fallback chain is silently doing
    more work than expected -- worth knowing explicitly rather than only
    finding out via a degraded (non-chronological) split."""
    missing = [
        img.image_id for img in real_dataset.images.values() if img.date_captured is None
    ]
    assert missing == [], f"{len(missing)} images had unparseable date_captured"


# --------------------------------------------------------------------------
# The duplicate-observation mechanism the plan's leakage argument rests on
# --------------------------------------------------------------------------


def test_duplicate_observations_actually_exist(real_dataset):
    """Confirms the core premise behind filament.data.splits: some frames
    really are annotated more than once under different image ids."""
    keys = real_dataset.observation_keys()
    key_counts: dict[str, int] = {}
    for key in keys.values():
        key_counts[key] = key_counts.get(key, 0) + 1

    duplicated = {k: c for k, c in key_counts.items() if c > 1}
    assert len(duplicated) > 0, "expected at least some re-annotated observations"

    n_unique = len(key_counts)
    n_images = len(real_dataset)
    assert n_unique < n_images, (
        f"expected unique observations ({n_unique}) < total images "
        f"({n_images}) given known duplicate re-annotations"
    )


def test_a_known_duplicate_group_resolves_to_one_key(real_dataset):
    """Spot-check the specific duplicate group found during the audit:
    "050101-...Lh", "050102-...Lh", "050103-...Lh" share one timestamp."""
    candidates = [
        img_id
        for img_id in real_dataset.images
        if img_id.startswith("0501") and img_id.endswith("20111116063134Lh")
    ]
    if not candidates:
        pytest.skip("the specific spot-checked duplicate group was not found "
                     "in this copy of the data -- file may have been updated")
    keys = {real_dataset.images[i].observation_key for i in candidates}
    assert len(keys) == 1, f"expected one shared observation_key, got {keys}"


# --------------------------------------------------------------------------
# Splitting and leakage, on the real dataset
# --------------------------------------------------------------------------


def test_time_grouped_split_runs_and_is_leak_free_on_real_data(real_dataset):
    assignment = time_grouped_split(real_dataset, val_fraction=0.15, test_fraction=0.15)
    assert_no_leakage(real_dataset, assignment)  # must not raise

    total = len(real_dataset)
    n_train = len(assignment.image_ids("train"))
    n_val = len(assignment.image_ids("val"))
    n_test = len(assignment.image_ids("test"))
    assert n_train + n_val + n_test == total
    assert n_train > n_val and n_train > n_test  # train is the majority split


# --------------------------------------------------------------------------
# Polygon rasterisation vs. the file's own reported area
# --------------------------------------------------------------------------


def test_polygon_area_matches_reported_area_field(real_dataset):
    """For a sample of real annotations, our rasteriser's pixel count should
    closely match the annotation's own `area` field -- a check that does not
    depend on pycocotools at all, using the file's own ground truth instead.
    """
    rng = np.random.default_rng(0)
    all_anns = [a for anns in real_dataset.annotations_by_image.values() for a in anns]
    sample = rng.choice(len(all_anns), size=min(40, len(all_anns)), replace=False)

    rel_errors = []
    for idx in sample:
        ann = all_anns[idx]
        if ann.area is None or ann.area <= 0:
            continue
        img = real_dataset.images[ann.image_id]
        mask = polygon_to_mask(ann.segmentation, img.height, img.width)
        computed_area = float(mask.sum())
        rel_errors.append(abs(computed_area - ann.area) / ann.area)

    assert rel_errors, "no annotations with a usable area field in the sample"
    median_error = float(np.median(rel_errors))
    assert median_error < 0.05, f"median relative area error {median_error:.3f} too high"


# --------------------------------------------------------------------------
# GT structural rules (single connected component, no holes) -- MAGFiLO's
# own annotation protocol claims this always holds; verify it does.
# --------------------------------------------------------------------------


def test_ground_truth_masks_are_single_connected_component(real_dataset):
    from scipy import ndimage as ndi

    rng = np.random.default_rng(1)
    all_anns = [a for anns in real_dataset.annotations_by_image.values() for a in anns]
    sample = rng.choice(len(all_anns), size=min(60, len(all_anns)), replace=False)

    multi_component = 0
    for idx in sample:
        ann = all_anns[idx]
        img = real_dataset.images[ann.image_id]
        mask = polygon_to_mask(ann.segmentation, img.height, img.width)
        if not mask.any():
            continue
        _, n = ndi.label(mask)
        if n > 1:
            multi_component += 1

    # Allow a small tolerance for rasterisation-boundary artefacts (a
    # single-pixel-wide bridge that our scanline fill happens to miss);
    # the vast majority must still be single-component per the annotation
    # rule, so a high violation rate would indicate a rasteriser bug, not
    # normal noise.
    assert multi_component <= 3, (
        f"{multi_component}/{len(sample)} sampled GT masks were multi-component; "
        "expected ~0 per MAGFiLO's single-connected-component annotation rule"
    )
