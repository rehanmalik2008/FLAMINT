"""Tests for solar disk fitting and the heliographic meridian mask.

Synthetic-disk tests only -- this module's contract with real GONG frames
(image orientation, whether solar north is up, exact preprocessing) is
explicitly unverified until P0's data audit, per the caveat in
`disk.py`'s module docstring.
"""

from __future__ import annotations

import numpy as np
import pytest

from filament.geometry.disk import DiskGeometry, fit_disk, meridian_mask


def synthetic_disk(
    shape: tuple[int, int] = (400, 400),
    center: tuple[float, float] = (200.0, 200.0),
    radius: float = 150.0,
) -> np.ndarray:
    """A boolean mask of a filled circle -- a noise-free stand-in for a
    thresholded solar disk."""
    rows, cols = np.indices(shape)
    dr = rows - center[0]
    dc = cols - center[1]
    return (dr * dr + dc * dc) <= radius * radius


# --------------------------------------------------------------------------
# fit_disk
# --------------------------------------------------------------------------


def test_fit_disk_recovers_known_center_and_radius():
    mask = synthetic_disk(center=(210.0, 195.0), radius=160.0)
    geo = fit_disk(mask)
    assert geo.center_row == pytest.approx(210.0, abs=1.0)
    assert geo.center_col == pytest.approx(195.0, abs=1.0)
    assert geo.radius == pytest.approx(160.0, rel=0.02)


def test_fit_disk_robust_to_ragged_boundary():
    """A boundary perturbed by noise should not throw off the fit by much."""
    rng = np.random.default_rng(0)
    rows, cols = np.indices((400, 400))
    dr, dc = rows - 200.0, cols - 200.0
    radial = np.sqrt(dr * dr + dc * dc)
    # Jitter the effective radius per-pixel by a few percent.
    jitter = rng.normal(0, 3.0, radial.shape)
    mask = radial <= (150.0 + jitter)

    geo = fit_disk(mask)
    assert geo.center_row == pytest.approx(200.0, abs=2.0)
    assert geo.radius == pytest.approx(150.0, rel=0.05)


def test_fit_disk_rejects_empty_mask():
    with pytest.raises(ValueError, match="no foreground"):
        fit_disk(np.zeros((100, 100), dtype=bool))


def test_fit_disk_rejects_non_2d():
    with pytest.raises(ValueError, match="2-D"):
        fit_disk(np.zeros((10, 10, 3), dtype=bool))


def test_fit_disk_rejects_degenerate_tiny_mask():
    """A few scattered bright pixels are not a solar disk."""
    mask = np.zeros((400, 400), dtype=bool)
    mask[200, 200] = True
    mask[201, 200] = True
    with pytest.raises(ValueError, match="sanity floor"):
        fit_disk(mask)


# --------------------------------------------------------------------------
# DiskGeometry properties
# --------------------------------------------------------------------------


def test_normalized_radius_at_center_is_zero():
    geo = DiskGeometry(200.0, 200.0, 150.0, (400, 400))
    assert geo.normalized_radius(np.array([200.0]), np.array([200.0]))[0] == 0.0


def test_normalized_radius_at_limb_is_one():
    geo = DiskGeometry(200.0, 200.0, 150.0, (400, 400))
    r = geo.normalized_radius(np.array([200.0]), np.array([350.0]))
    assert r[0] == pytest.approx(1.0)


def test_longitude_is_zero_at_central_meridian():
    geo = DiskGeometry(200.0, 200.0, 150.0, (400, 400))
    lon = geo.longitude(np.array([200.0, 50.0, 350.0]), np.array([200.0, 200.0, 200.0]))
    np.testing.assert_allclose(lon, [0.0, 0.0, 0.0], atol=1e-9)


def test_longitude_at_east_west_limb_is_ninety_degrees():
    geo = DiskGeometry(200.0, 200.0, 150.0, (400, 400))
    lon = geo.longitude(np.array([200.0, 200.0]), np.array([50.0, 350.0]))
    assert lon[0] == pytest.approx(-90.0)
    assert lon[1] == pytest.approx(90.0)


def test_longitude_clips_beyond_the_fitted_limb():
    """Off-disk pixels must not raise or produce NaN from arcsin."""
    geo = DiskGeometry(200.0, 200.0, 150.0, (400, 400))
    lon = geo.longitude(np.array([200.0]), np.array([1000.0]))
    assert np.isfinite(lon).all()
    assert lon[0] == pytest.approx(90.0)


def test_on_disk_respects_margin():
    geo = DiskGeometry(200.0, 200.0, 150.0, (400, 400))
    rows = np.array([200.0, 200.0])
    cols = np.array([340.0, 360.0])  # r/R = 0.933, 1.067
    inside = geo.on_disk(rows, cols, margin=1.0)
    assert inside[0] and not inside[1]


# --------------------------------------------------------------------------
# meridian_mask -- the actual filter applied to predictions
# --------------------------------------------------------------------------


def test_meridian_mask_excludes_disk_center_never():
    geo = DiskGeometry(200.0, 200.0, 150.0, (400, 400))
    mask = meridian_mask(geo)
    assert mask[200, 200]  # disk centre is always within +/-70 deg


def test_meridian_mask_excludes_far_limb():
    """A point at sin(lon)=1 (90 deg) must be excluded by a 70 deg cutoff."""
    geo = DiskGeometry(200.0, 200.0, 150.0, (400, 400))
    mask = meridian_mask(geo, max_longitude=70.0)
    assert not mask[200, 349]  # near the east limb, ~90 deg


def test_meridian_mask_boundary_is_at_asin_of_cutoff():
    """The mask boundary should sit at col = center + R*sin(70deg), not R itself."""
    geo = DiskGeometry(200.0, 200.0, 150.0, (400, 400))
    mask = meridian_mask(geo, max_longitude=70.0)

    boundary_col = 200.0 + 150.0 * np.sin(np.radians(70.0))
    just_inside = int(np.floor(boundary_col)) - 1
    just_outside = int(np.ceil(boundary_col)) + 1

    assert mask[200, just_inside]
    assert not mask[200, just_outside]


def test_meridian_mask_excludes_off_disk_pixels_even_at_zero_longitude():
    """A pixel due north/south of centre, off the disk, must still be excluded."""
    geo = DiskGeometry(200.0, 200.0, 150.0, (400, 400))
    mask = meridian_mask(geo, max_longitude=70.0)
    # Straight up from centre, well beyond the limb: longitude is 0 by this
    # projection (only column offset matters) but the pixel is off-disk.
    assert not mask[0, 200]


def test_meridian_mask_shrinks_monotonically_with_cutoff():
    geo = DiskGeometry(200.0, 200.0, 150.0, (400, 400))
    wide = meridian_mask(geo, max_longitude=80.0)
    narrow = meridian_mask(geo, max_longitude=40.0)
    assert narrow.sum() < wide.sum()
    # Every pixel passing the narrow cutoff must also pass the wide one.
    assert np.array_equal(narrow & wide, narrow)


# --------------------------------------------------------------------------
# End-to-end: fit then mask
# --------------------------------------------------------------------------


def test_end_to_end_fit_and_mask_on_synthetic_disk():
    mask = synthetic_disk(center=(200.0, 200.0), radius=150.0)
    geo = fit_disk(mask)
    excl = meridian_mask(geo, max_longitude=70.0)
    # Central meridian point admitted; far-limb point excluded.
    assert excl[200, 200]
    assert not excl[200, 349]
