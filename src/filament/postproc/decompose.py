"""Decomposing a semantic filament mask into instances.

The two failure modes the literature reports -- fragmentation (a filament's
barbs or thin bridges drop out, splitting it into pieces) and over-merging
(two nearby filaments fuse) -- are exactly what Panoptic Quality punishes
hardest: both collapse to PQ = 0.0 for the affected filaments rather than a
partial-credit score (see ``tests/test_panoptic.py`` for the proof). MORDEN
(Hu et al. 2026) needed a post-hoc DBSCAN pass to patch fragmentation and only
recovered 84.4% correctly, which is the field's current ceiling on this
specific problem.

This module takes a different approach: **seeded watershed**, seeded by
connected components of a *skeleton* heatmap rather than of the mask itself.
The mask alone fragments wherever a barb thins below the segmentation
threshold; the skeleton is a 1-D idealisation of the filament's spine, so a
seed persists as long as the model's confidence trace along the spine stays
above threshold anywhere -- a strictly weaker requirement than the whole mask
staying connected. Two touching filaments, conversely, remain separable
because they have geometrically distinct spines, which the watershed's ridge
lines will fall between.

Every output mask is post-processed to satisfy MAGFiLO's own two structural
annotation rules -- single connected piece, no holes -- since ground truth is
guaranteed to have both and violating them can only cost IoU.

At training time, the "skeleton heatmap" this module expects as an input is
produced by a mask-derived skeleton (``skimage.morphology.skeletonize`` on the
ground-truth mask, per the rules-compliance note in the project plan), used as
a supervision target for a dedicated spine head. This module operates purely
on model outputs at inference time and has no dependency on that training-time
choice.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage as ndi
from skimage.morphology import remove_small_objects
from skimage.segmentation import watershed

__all__ = [
    "DecomposedInstance",
    "decompose",
    "enforce_structural_rules",
]


@dataclass(frozen=True)
class DecomposedInstance:
    """One candidate filament instance produced by decomposition."""

    mask: np.ndarray  # 2-D bool
    confidence: float  # scalar summary, e.g. mean mask probability over the instance
    area: int


def enforce_structural_rules(mask: np.ndarray) -> np.ndarray:
    """Force a binary mask to satisfy MAGFiLO's annotation rules.

    Ground-truth filaments are, by the dataset's own annotation protocol,
    always a single connected piece with no holes. A predicted mask that
    violates either can only be losing IoU relative to the closest legal mask,
    so this is applied unconditionally as the last post-processing step for
    every emitted instance.

    - **Holes**: filled unconditionally via ``scipy.ndimage.binary_fill_holes``,
      which fills background regions that do not touch the array border --
      exactly "holes" in the topological sense, as opposed to
      ``skimage.morphology.remove_small_holes`` at a large threshold, which
      (perhaps surprisingly) also swallows border-connected background and is
      not what is wanted here.
    - **Multiple components**: only the largest connected component is kept.
      Prefer *not* triggering this path by seeding decomposition well
      (``decompose``); reaching it is a signal the mask should have been split
      into separate instances upstream, not silently pruned.
    """
    if mask.ndim != 2:
        raise ValueError(f"expected a 2-D mask, got shape {mask.shape}")
    if not mask.any():
        return mask.astype(bool)

    filled = ndi.binary_fill_holes(mask.astype(bool))

    labelled, n = ndi.label(filled)
    if n > 1:
        sizes = ndi.sum(filled, labelled, index=np.arange(1, n + 1))
        largest = int(np.argmax(sizes)) + 1
        filled = labelled == largest

    return filled


def decompose(
    mask_prob: np.ndarray,
    spine_prob: np.ndarray,
    *,
    mask_threshold: float = 0.5,
    spine_threshold: float = 0.5,
    min_area: int = 15,
) -> list[DecomposedInstance]:
    """Split a semantic probability map into per-instance masks.

    Parameters
    ----------
    mask_prob:
        Per-pixel probability of belonging to *some* filament (the semantic
        head's output), in ``[0, 1]``.
    spine_prob:
        Per-pixel probability of lying on a filament's spine (the auxiliary
        spine head's output), in ``[0, 1]``. Expected to be much sparser than
        ``mask_prob`` and roughly 1-pixel-wide along each filament's axis.
    mask_threshold:
        Binarisation threshold for the semantic mask. Note this is a *pixel*
        threshold used to build the watershed's flooding region -- it is
        independent of the *instance*-level emit threshold applied later by
        ``filament.postproc.emit_policy``.
    spine_threshold:
        Binarisation threshold for seed extraction. Kept separate from
        ``mask_threshold`` because the two heads are not calibrated against
        each other; a spine pixel surviving at a lower bar than the mask
        threshold is exactly the case this design is meant to rescue.
    min_area:
        Minimum instance area in pixels, applied *before* watershed to drop
        spurious seeds (isolated spine-probability noise) and *after*, to drop
        slivers the flooding produced. MAGFiLO's smallest annotated filaments
        are far larger than typical sensor noise, so this is a conservative
        floor rather than a competitive parameter.

    Returns
    -------
    list[DecomposedInstance]
        One entry per recovered instance, each satisfying
        ``enforce_structural_rules``. Empty list if the frame has no filament
        pixels above threshold.

    Notes
    -----
    If ``spine_prob`` yields zero seeds inside a region where ``mask_prob``
    still clears threshold (e.g. a small blob with a spine too weak to
    survive), that region is emitted as a single unsplit instance rather than
    silently dropped -- recall matters more than clean decomposition for such
    fragments, and Edge 1's emit policy can still choose to suppress it later
    based on the resulting confidence.
    """
    if mask_prob.shape != spine_prob.shape:
        raise ValueError(
            f"mask_prob and spine_prob must share a shape, got "
            f"{mask_prob.shape} and {spine_prob.shape}"
        )

    fg = mask_prob >= mask_threshold
    if not fg.any():
        return []

    # skimage>=0.26 renamed min_size->max_size with inclusive semantics
    # ("removes objects <= max_size"); max_size=min_area-1 reproduces the
    # original "keep objects with area >= min_area" intent.
    seeds_mask = remove_small_objects(
        spine_prob >= spine_threshold, max_size=max(min_area - 1, 0)
    )
    seed_labels, n_seeds = ndi.label(seeds_mask)

    if n_seeds == 0:
        # No confident spine anywhere: emit whatever foreground survives as
        # unsplit blobs, rather than discarding a detection outright.
        labelled, n_blobs = ndi.label(fg)
        results: list[DecomposedInstance] = []
        for i in range(1, n_blobs + 1):
            blob = labelled == i
            if blob.sum() < min_area:
                continue
            clean = enforce_structural_rules(blob)
            results.append(
                DecomposedInstance(
                    mask=clean,
                    confidence=float(mask_prob[clean].mean()) if clean.any() else 0.0,
                    area=int(clean.sum()),
                )
            )
        return results

    # Flood from the spine seeds, using inverted mask-probability as the
    # elevation so flooding prefers high-confidence filament pixels over
    # low-confidence ones.
    #
    # The flood domain is fg UNION the seed pixels themselves, not fg alone.
    # This is the mechanism that actually delivers the module's central
    # claim: a barb/bridge whose pixel-level mask probability dips below
    # `mask_threshold` but whose spine probability stays above
    # `spine_threshold` must not fragment. If the domain were `fg` alone, a
    # seed spanning such a gap would still be cut by the domain boundary --
    # the two sides of the gap would flood as separate components carrying
    # the *same* watershed label, which is topologically disconnected and
    # would then be pruned down to one side by `enforce_structural_rules`
    # (single-component rule), silently discarding the other half.
    elevation = 1.0 - mask_prob
    flood_domain = fg | seeds_mask
    labels = watershed(elevation, markers=seed_labels, mask=flood_domain)

    results = []
    for i in range(1, n_seeds + 1):
        instance = labels == i
        if instance.sum() < min_area:
            continue
        clean = enforce_structural_rules(instance)
        if clean.sum() < min_area:
            continue
        results.append(
            DecomposedInstance(
                mask=clean,
                confidence=float(mask_prob[clean].mean()),
                area=int(clean.sum()),
            )
        )
    return results
