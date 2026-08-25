"""Deriving spine and offset training targets from ground-truth masks.

The decoder's spine and offset heads (``filament.models.unet.FilamentUNet``)
need per-pixel supervision that MAGFiLO's mask polygons do not directly
provide. This module builds it from the mask alone:

- **Spine target**: ``skimage.morphology.skeletonize`` applied to the binary
  mask, dilated by one pixel to give it enough area to be a learnable target
  (a literal 1-px-wide line is an extreme class-imbalance case even relative
  to the already-sparse mask).
- **Offset target**: for every foreground pixel, the (row, col) unit vector
  toward its nearest skeleton pixel, computed via a Euclidean distance
  transform's return indices -- background pixels get a zero vector, since
  ``filament.losses.segmentation.offset_loss`` masks them out and their value
  is never used.

Rules-compliance note (see also ``filament.postproc.decompose`` and the
project plan): MAGFiLO ships its own spine polylines as ground truth, but the
competition rules bar training on "other ground-truth metadata" beyond
segmentation masks, and it is unresolved whether spine annotations count.
Deriving the skeleton from the mask itself, as this module does, sidesteps
the question entirely -- it is a deterministic function of data the rules
unambiguously permit, so training has no dependency on the ruling either way.
Using MAGFiLO's shipped spines as an additional signal remains an option if
the organizers confirm it is allowed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage as ndi
from skimage.morphology import dilation, skeletonize

__all__ = ["FilamentTargets", "build_spine_target", "build_offset_target", "build_targets"]


@dataclass(frozen=True)
class FilamentTargets:
    """Per-pixel training targets derived from one filament (or frame) mask.

    mask       : (H, W) bool     -- the input, echoed back for convenience
    spine      : (H, W) float32  -- {0, 1} target for the spine head
    offsets    : (2, H, W) float32 -- (row, col) unit vectors, zero on background
    """

    mask: np.ndarray
    spine: np.ndarray
    offsets: np.ndarray


def build_spine_target(mask: np.ndarray, dilation_radius: int = 1) -> np.ndarray:
    """Skeletonize a mask and dilate the result to a learnable target width.

    Parameters
    ----------
    mask:
        2-D boolean array. Works equally on a single filament's mask or a
        whole-frame union of filaments -- skeletonize does not need instances
        pre-separated; it operates on whatever connected structure is present.
    dilation_radius:
        Structuring-element radius (in the 8-connected sense) for
        ``skimage.morphology.dilation``. A radius of 1 turns a 1-px skeleton
        line into a ~3-px-wide band, which is still far sparser than the mask
        itself but no longer a near-impossible target for a
        discretely-sampled decoder to hit exactly. The square footprint used
        here is symmetric, so the dilation-mirroring caveat that applies to
        asymmetric footprints in newer skimage versions does not apply.

    Notes
    -----
    An all-background mask skeletonizes to all-background; this is handled
    correctly (returns an all-zero array) without a special case, since
    ``skeletonize`` and ``binary_dilation`` are both well-defined on an empty
    array.
    """
    if mask.ndim != 2:
        raise ValueError(f"expected a 2-D mask, got shape {mask.shape}")
    skeleton = skeletonize(mask.astype(bool))
    if dilation_radius > 0:
        footprint = np.ones((2 * dilation_radius + 1,) * 2, dtype=bool)
        skeleton = dilation(skeleton, footprint=footprint)
    return skeleton.astype(np.float32)


def build_offset_target(mask: np.ndarray, spine: np.ndarray) -> np.ndarray:
    """Per-foreground-pixel unit vector toward the nearest spine pixel.

    Parameters
    ----------
    mask:
        2-D boolean array; defines which pixels get a non-zero offset.
    spine:
        2-D array (bool or {0,1} float), the target this function points
        toward -- typically the *undilated* skeleton, so pixels are pointed
        at the true 1-px-wide spine rather than at the nearest edge of a
        dilated band, which would be a systematically shorter, less
        informative vector for pixels already near the skeleton.

    Returns
    -------
    np.ndarray
        Shape ``(2, H, W)``, float32, channel 0 = row component, channel 1 =
        col component. Zero everywhere the mask is background.

    Notes
    -----
    If ``spine`` is entirely empty (degenerate: skeletonize can occasionally
    collapse a very small or thin mask to nothing), every foreground pixel
    gets a zero-length offset rather than raising or producing NaN/inf from
    an undefined nearest-neighbour query -- a foreground region with no spine
    to point to still needs a defined (if uninformative) training target.
    """
    if mask.shape != spine.shape:
        raise ValueError(
            f"mask and spine must share a shape, got {mask.shape} and {spine.shape}"
        )
    h, w = mask.shape
    spine_bool = spine.astype(bool)

    if not spine_bool.any():
        return np.zeros((2, h, w), dtype=np.float32)

    # distance_transform_edt on the *complement* of the spine gives, for every
    # pixel, the distance to and (via return_indices) the coordinates of the
    # nearest spine pixel.
    _, indices = ndi.distance_transform_edt(~spine_bool, return_indices=True)
    nearest_row, nearest_col = indices[0], indices[1]

    rows, cols = np.indices((h, w))
    dr = (nearest_row - rows).astype(np.float32)
    dc = (nearest_col - cols).astype(np.float32)
    norm = np.sqrt(dr * dr + dc * dc)
    norm[norm == 0] = 1.0  # spine pixels themselves: zero vector, avoid 0/0

    offsets = np.stack([dr / norm, dc / norm], axis=0)
    fg = mask.astype(bool)
    offsets *= fg[np.newaxis, :, :]
    return offsets.astype(np.float32)


def build_targets(mask: np.ndarray, dilation_radius: int = 1) -> FilamentTargets:
    """Convenience wrapper: mask -> (spine target, offset target) in one call.

    The offset target is computed against the *undilated* skeleton (see
    ``build_offset_target``'s Notes), even though the returned ``spine``
    field is the dilated version used for the spine head's own supervision --
    the two targets intentionally use different widths of the same skeleton.
    """
    undilated = skeletonize(mask.astype(bool))
    spine_target = build_spine_target(mask, dilation_radius=dilation_radius)
    offset_target = build_offset_target(mask, undilated)
    return FilamentTargets(mask=mask.astype(bool), spine=spine_target, offsets=offset_target)
