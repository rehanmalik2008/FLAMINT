"""Solar disk fitting and heliographic coordinates from an H-alpha frame.

MAGFiLO's annotation protocol excludes anything beyond +/-70 degrees of
heliographic longitude from central meridian, and excludes filaments too faint
or small to show a feature (Ahmadzadeh et al. 2024). That means the ground
truth provably contains no positive pixels near the limb. A prediction there is
a guaranteed false positive under Panoptic Quality, so masking that region
before post-processing is free precision -- provided the disk is located
correctly.

The dataset paper also states GONG JPEGs are produced with the disk centred and
diameter-normalised to within +/-3%, so a robust circle fit should nearly
always land close to the frame centre; large deviations are a signal that
something is wrong with that particular frame (a defect, a crop, or a fit
failure), not that the star moved.

This module makes no claim about the *default* image orientation of the JPEGs
(e.g. whether solar north is up) -- that must be confirmed against real files
in P0's data audit before `DiskGeometry.longitude` is trusted end to end. What
*is* geometry-only and independent of that convention is the disk location and
its projected radius; those are what `fit_disk` recovers.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "DiskGeometry",
    "fit_disk",
    "meridian_mask",
]


@dataclass(frozen=True)
class DiskGeometry:
    """A fitted solar disk: centre and apparent radius, in pixels."""

    center_row: float
    center_col: float
    radius: float
    image_shape: tuple[int, int]

    def normalized_radius(self, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
        """r / R_sun for the given pixel coordinates (0 at centre, 1 at limb)."""
        dr = rows - self.center_row
        dc = cols - self.center_col
        return np.sqrt(dr * dr + dc * dc) / self.radius

    def longitude(self, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
        """Approximate heliographic longitude from disk centre, in degrees.

        Uses the orthographic projection ``sin(longitude) = x / R_sun``, which
        is exact for a point on the solar equator viewed with zero B-angle and
        a good approximation elsewhere near disk centre. This intentionally
        ignores the P/B-angle corrections a full heliographic transform would
        need -- adequate for a coarse "how far from central meridian" mask, not
        for scientific longitude measurement.

        Pixels beyond the fitted limb (``|x/R| > 1``) are clipped to +/-90
        degrees rather than producing NaN, since they are off-disk regardless
        of exact longitude.
        """
        dc = cols - self.center_col
        x = np.clip(dc / self.radius, -1.0, 1.0)
        return np.degrees(np.arcsin(x))

    def on_disk(self, rows: np.ndarray, cols: np.ndarray, margin: float = 1.0) -> np.ndarray:
        """Boolean mask of pixels within ``margin`` * R_sun of the centre."""
        return self.normalized_radius(rows, cols) <= margin


def fit_disk(
    is_bright: np.ndarray,
    *,
    min_radius_fraction: float = 0.3,
) -> DiskGeometry:
    """Fit a circular solar disk from a foreground/background mask.

    Parameters
    ----------
    is_bright:
        Boolean array, True where the pixel belongs to the solar disk (e.g. an
        intensity threshold on the H-alpha frame; filaments are dark against a
        bright disk, so a simple global threshold typically suffices for this
        purpose even though it would be a poor filament detector on its own).
    min_radius_fraction:
        Sanity floor: the fitted radius must be at least this fraction of
        ``min(rows, cols) / 2``, or a ``ValueError`` is raised. Catches a
        degenerate fit on a mostly-empty mask rather than silently returning
        a tiny circle.

    Method
    ------
    The centroid of ``is_bright`` estimates the centre. The radius is then
    estimated twice -- from the area (``sqrt(area / pi)``) and from the 95th
    percentile of the radial distance of foreground pixels from the centroid
    -- and the two are averaged. Using both guards against two different
    failure modes: a ragged, non-circular thresholded boundary (area-based
    estimate is robust to that) and a mask that includes bright non-disk
    artefacts as a diffuse halo (the percentile-based estimate is a tighter
    envelope of the true disk edge).

    This is a lightweight, dependency-free estimator, not a sub-pixel limb fit.
    It is expected to be replaced or refined in P1 once real frames are
    available, since the dataset paper's own preprocessing already includes a
    disk-fitting step whose parameters we do not yet have visibility into.
    """
    if is_bright.ndim != 2:
        raise ValueError(f"expected a 2-D mask, got shape {is_bright.shape}")

    rows_idx, cols_idx = np.nonzero(is_bright)
    if rows_idx.size == 0:
        raise ValueError("no foreground pixels to fit a disk to")

    center_row = float(rows_idx.mean())
    center_col = float(cols_idx.mean())

    area = float(is_bright.sum())
    radius_from_area = float(np.sqrt(area / np.pi))

    dr = rows_idx - center_row
    dc = cols_idx - center_col
    radial_dist = np.sqrt(dr * dr + dc * dc)
    radius_from_percentile = float(np.percentile(radial_dist, 95))

    radius = 0.5 * (radius_from_area + radius_from_percentile)

    floor = min_radius_fraction * (min(is_bright.shape) / 2.0)
    if radius < floor:
        raise ValueError(
            f"fitted radius {radius:.1f}px is below the sanity floor "
            f"{floor:.1f}px ({min_radius_fraction:.0%} of half the shorter "
            "side) -- the foreground mask is likely not a solar disk"
        )

    return DiskGeometry(
        center_row=center_row,
        center_col=center_col,
        radius=radius,
        image_shape=is_bright.shape,
    )


def meridian_mask(geometry: DiskGeometry, max_longitude: float = 70.0) -> np.ndarray:
    """Boolean mask, True within ``max_longitude`` degrees of central meridian.

    This is the ground-truth exclusion zone from the MAGFiLO annotation
    protocol, applied to predictions: anything outside it is a pixel the
    annotators were instructed never to label, so a prediction there can only
    ever be a false positive.

    Also excludes off-disk pixels, since those are outside the meridian
    boundary in the trivial sense and would otherwise pass the longitude test
    at exactly +/-90 degrees due to the clipping in ``DiskGeometry.longitude``.
    """
    rows, cols = np.indices(geometry.image_shape)
    lon = geometry.longitude(rows, cols)
    return (np.abs(lon) <= max_longitude) & geometry.on_disk(rows, cols)
