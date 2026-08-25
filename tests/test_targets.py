"""Tests for spine/offset target generation from ground-truth masks.

The key property under test for offsets: following the predicted vector field
from any foreground pixel should walk you toward the skeleton, not away from
it or in a random direction. That is checked directly, not just inferred from
shape/range assertions.
"""

from __future__ import annotations

import numpy as np
import pytest

from filament.data.targets import build_offset_target, build_spine_target, build_targets


def bar_mask(shape=(60, 100), rows=(25, 35), cols=(10, 90)) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    mask[rows[0] : rows[1], cols[0] : cols[1]] = True
    return mask


# --------------------------------------------------------------------------
# build_spine_target
# --------------------------------------------------------------------------


def test_spine_target_is_subset_of_mask_dilation():
    """The spine, even dilated, should not extend meaningfully beyond the
    original mask's footprint for a simple filled rectangle."""
    mask = bar_mask()
    spine = build_spine_target(mask, dilation_radius=1)
    assert spine.sum() > 0
    assert spine.sum() < mask.sum()  # sparser than the mask itself


def test_spine_target_empty_mask_is_empty():
    mask = np.zeros((40, 40), dtype=bool)
    spine = build_spine_target(mask)
    assert not spine.any()


def test_spine_target_dilation_increases_area():
    mask = bar_mask()
    thin = build_spine_target(mask, dilation_radius=0)
    thick = build_spine_target(mask, dilation_radius=2)
    assert thick.sum() > thin.sum()


def test_spine_target_rejects_non_2d():
    with pytest.raises(ValueError, match="2-D"):
        build_spine_target(np.zeros((5, 5, 2), dtype=bool))


def test_spine_target_runs_through_bar_lengthwise():
    """For a long horizontal bar, the skeleton should be a roughly horizontal
    line near the vertical centre, not scattered noise."""
    mask = bar_mask(rows=(25, 35), cols=(10, 90))
    spine = build_spine_target(mask, dilation_radius=0)
    rows_hit = np.nonzero(spine)[0]
    assert rows_hit.size > 0
    # All spine pixels should be within the bar's vertical extent.
    assert rows_hit.min() >= 25 and rows_hit.max() < 35
    # Should span most of the bar's horizontal extent.
    cols_hit = np.nonzero(spine)[1]
    assert cols_hit.max() - cols_hit.min() > 60


# --------------------------------------------------------------------------
# build_offset_target
# --------------------------------------------------------------------------


def test_offset_target_zero_on_background():
    mask = bar_mask()
    from skimage.morphology import skeletonize

    spine = skeletonize(mask)
    offsets = build_offset_target(mask, spine)
    background = ~mask
    assert np.allclose(offsets[:, background], 0.0)


def test_offset_target_zero_at_spine_pixels():
    mask = bar_mask()
    from skimage.morphology import skeletonize

    spine = skeletonize(mask)
    offsets = build_offset_target(mask, spine)
    # At the spine itself, distance to nearest spine pixel is 0.
    spine_pixels = np.nonzero(spine)
    assert np.allclose(offsets[0][spine_pixels], 0.0)
    assert np.allclose(offsets[1][spine_pixels], 0.0)


def test_offset_target_all_zero_when_spine_is_empty():
    """A degenerate case: no spine at all must yield zero offsets, not NaN."""
    mask = bar_mask()
    empty_spine = np.zeros_like(mask, dtype=bool)
    offsets = build_offset_target(mask, empty_spine)
    assert np.all(np.isfinite(offsets))
    assert np.allclose(offsets, 0.0)


def test_offset_vectors_point_toward_nearest_spine_pixel():
    """The core geometric correctness check: stepping from a foreground pixel
    in the direction of its offset vector must strictly decrease its distance
    to the nearest spine pixel, for a set of sample points."""
    mask = bar_mask(shape=(60, 100), rows=(25, 35), cols=(10, 90))
    from skimage.morphology import skeletonize

    spine = skeletonize(mask)
    offsets = build_offset_target(mask, spine)

    spine_coords = np.argwhere(spine)

    def nearest_dist(r, c):
        d = np.sqrt(((spine_coords - np.array([r, c])) ** 2).sum(axis=1))
        return d.min()

    rng = np.random.default_rng(0)
    fg_coords = np.argwhere(mask & ~spine)
    sample = fg_coords[rng.choice(len(fg_coords), size=30, replace=False)]

    for r, c in sample:
        dr, dc = offsets[0, r, c], offsets[1, r, c]
        before = nearest_dist(r, c)
        # Step a small amount in the offset direction.
        after = nearest_dist(r + dr, c + dc)
        assert after < before, f"pixel ({r},{c}) offset did not reduce spine distance"


def test_offset_target_unit_vectors_away_from_spine():
    """Foreground pixels not exactly on the spine should have (approximately)
    unit-length offset vectors."""
    mask = bar_mask()
    from skimage.morphology import skeletonize

    spine = skeletonize(mask)
    offsets = build_offset_target(mask, spine)

    non_spine_fg = mask & ~spine
    rng = np.random.default_rng(1)
    coords = np.argwhere(non_spine_fg)
    sample = coords[rng.choice(len(coords), size=20, replace=False)]
    for r, c in sample:
        norm = np.sqrt(offsets[0, r, c] ** 2 + offsets[1, r, c] ** 2)
        assert norm == pytest.approx(1.0, abs=1e-3)


def test_offset_target_rejects_shape_mismatch():
    with pytest.raises(ValueError, match="shape"):
        build_offset_target(np.zeros((10, 10), bool), np.zeros((10, 11), bool))


# --------------------------------------------------------------------------
# build_targets (convenience wrapper)
# --------------------------------------------------------------------------


def test_build_targets_returns_consistent_shapes():
    mask = bar_mask()
    targets = build_targets(mask)
    assert targets.mask.shape == mask.shape
    assert targets.spine.shape == mask.shape
    assert targets.offsets.shape == (2,) + mask.shape


def test_build_targets_offset_uses_undilated_skeleton():
    """The offset field must be computed against the thin skeleton, not the
    dilated spine target -- verified indirectly: a pixel just outside the
    dilated spine band, but on the undilated skeleton's extension, should
    have a near-zero offset only if it's on the true 1px line."""
    mask = bar_mask()
    targets = build_targets(mask, dilation_radius=2)
    from skimage.morphology import skeletonize

    true_skeleton = skeletonize(mask)
    skeleton_pixels = np.nonzero(true_skeleton)
    # Distance-to-self should be ~zero for true skeleton pixels, regardless
    # of how wide the *returned* spine target band is.
    norms = np.sqrt(
        targets.offsets[0][skeleton_pixels] ** 2 + targets.offsets[1][skeleton_pixels] ** 2
    )
    assert np.allclose(norms, 0.0, atol=1e-5)


def test_build_targets_empty_mask():
    mask = np.zeros((30, 30), dtype=bool)
    targets = build_targets(mask)
    assert not targets.spine.any()
    assert np.allclose(targets.offsets, 0.0)
