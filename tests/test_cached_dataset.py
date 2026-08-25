"""Tests for the precompute-and-cache fast path.

Strategy: write a synthetic JPEG already at the target cache resolution (so
resizing is a no-op and outputs can be compared exactly), run it through
`process_one` (the same function `scripts/precompute_targets.py` calls),
save the result, then load it back through `CachedFilamentDataset` and check
round-trip fidelity against what `FilamentDataset`'s live path would produce.
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

from filament.data.dataset_cached import CachedFilamentDataset  # noqa: E402
from filament.data.targets import build_offset_target, build_spine_target  # noqa: E402
from skimage.morphology import skeletonize  # noqa: E402


def rect_polygon(r0, r1, c0, c1) -> list[list[float]]:
    return [[c0, r0, c1, r0, c1, r1, c0, r1]]


@pytest.fixture
def synthetic_case(tmp_path):
    """A 96x96 image (== the cache resolution used in these tests, so
    process_one's internal resize is a no-op) with one rectangular filament."""
    size = 96
    arr = (np.random.default_rng(0).random((size, size)) * 255).astype(np.uint8)
    image_path = tmp_path / "img.jpeg"
    Image.fromarray(arr, mode="L").save(image_path, quality=95)

    segmentation = rect_polygon(20, 30, 20, 80)
    return image_path, segmentation, size


# --------------------------------------------------------------------------
# process_one
# --------------------------------------------------------------------------


def test_process_one_output_shapes_and_dtypes(synthetic_case):
    image_path, segmentation, size = synthetic_case
    data = process_one(image_path, [segmentation], size, size, size)

    assert data["image"].shape == (size, size)
    assert data["image"].dtype == np.uint8
    assert data["mask_shape"].tolist() == [size, size]
    assert data["offsets"].shape == (2, size, size)
    assert data["offsets"].dtype == np.float16

    # Packed bool arrays: ceil(size*size / 8) bytes.
    expected_packed_len = (size * size + 7) // 8
    assert data["mask"].shape == (expected_packed_len,)
    assert data["spine"].shape == (expected_packed_len,)


def test_process_one_mask_matches_direct_rasterisation(synthetic_case):
    """At a no-op resize (image already at target size), the cached mask
    must exactly match directly rasterising the polygon -- no resize-induced
    drift to account for."""
    image_path, segmentation, size = synthetic_case
    data = process_one(image_path, [segmentation], size, size, size)

    mask = np.unpackbits(data["mask"])[: size * size].reshape(size, size).astype(bool)
    from filament.data.coco import polygon_to_mask

    expected = polygon_to_mask([segmentation], size, size)
    assert np.array_equal(mask, expected)


def test_process_one_offsets_match_live_computation(synthetic_case):
    image_path, segmentation, size = synthetic_case
    data = process_one(image_path, [segmentation], size, size, size)
    offsets = data["offsets"].astype(np.float32)

    from filament.data.coco import polygon_to_mask

    mask = polygon_to_mask([segmentation], size, size)
    skeleton = skeletonize(mask)
    expected = build_offset_target(mask, skeleton)

    # float16 storage introduces small precision loss vs the float32 live
    # computation; a loose tolerance confirms it's a storage-precision
    # difference, not a logic divergence.
    np.testing.assert_allclose(offsets, expected, atol=1e-2)


def test_process_one_handles_no_annotations(synthetic_case):
    image_path, _, size = synthetic_case
    data = process_one(image_path, [], size, size, size)
    mask = np.unpackbits(data["mask"])[: size * size]
    assert not mask.any()


# --------------------------------------------------------------------------
# CachedFilamentDataset
# --------------------------------------------------------------------------


def test_cached_dataset_round_trip(tmp_path, synthetic_case):
    image_path, segmentation, size = synthetic_case
    data = process_one(image_path, [segmentation], size, size, size)

    image_id = "cached-test-0001"
    np.savez_compressed(tmp_path / f"{image_id}.npz", **data)

    dataset = CachedFilamentDataset(tmp_path, [image_id])
    assert len(dataset) == 1

    sample = dataset[0]
    assert sample.image.shape == (1, size, size)
    assert sample.mask.shape == (1, size, size)
    assert sample.spine.shape == (1, size, size)
    assert sample.offsets.shape == (2, size, size)
    assert sample.image_id == image_id

    assert sample.image.min() >= 0.0 and sample.image.max() <= 1.0


def test_cached_dataset_mask_matches_source(tmp_path, synthetic_case):
    image_path, segmentation, size = synthetic_case
    data = process_one(image_path, [segmentation], size, size, size)

    image_id = "cached-test-0002"
    np.savez_compressed(tmp_path / f"{image_id}.npz", **data)
    dataset = CachedFilamentDataset(tmp_path, [image_id])
    sample = dataset[0]

    from filament.data.coco import polygon_to_mask

    expected_mask = polygon_to_mask([segmentation], size, size)
    got_mask = sample.mask[0].numpy().astype(bool)
    assert np.array_equal(got_mask, expected_mask)


def test_cached_dataset_missing_file_raises_clear_error(tmp_path):
    dataset = CachedFilamentDataset(tmp_path, ["nonexistent-id"])
    with pytest.raises(FileNotFoundError, match="precompute_targets"):
        dataset[0]


def test_cached_dataset_sanitises_slashes_in_image_id(tmp_path, synthetic_case):
    """Real MAGFiLO ids don't contain slashes, but the sanitisation must be
    consistent between the writer (precompute script) and reader."""
    image_path, segmentation, size = synthetic_case
    data = process_one(image_path, [segmentation], size, size, size)

    image_id = "group/with-a-slash"
    np.savez_compressed(tmp_path / f"{image_id.replace('/', '_')}.npz", **data)
    dataset = CachedFilamentDataset(tmp_path, [image_id])
    sample = dataset[0]  # must not raise
    assert sample.image_id == image_id
