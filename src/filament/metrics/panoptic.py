"""Panoptic Quality (PQ) for single-class instance segmentation.

Implements the metric of Kirillov et al., "Panoptic Segmentation" (CVPR 2019),
specialised to the one-class case used by the Solar Filament Segmentation
Challenge 2026:

    PQ = sum_{(g,p) in TP} IoU(g, p) / ( |TP| + 0.5 |FP| + 0.5 |FN| )

A ground-truth segment g and a predicted segment p form a true positive iff
IoU(g, p) > 0.5. Kirillov et al. prove that at this threshold the matching is
unique, so no assignment algorithm is required *provided* the segments within
each set are non-overlapping. Predictions submitted as independent RLEs are not
guaranteed to be disjoint, so `match_segments` falls back to greedy
descending-IoU matching, which is optimal here (see the module tests for the
pathological overlapping case).

Masks are represented sparsely, as sorted arrays of flat pixel indices. A
filament covers ~2,250 px of a 2048x2048 (4.2 M px) frame, so dense label maps
would waste three orders of magnitude of work.

WARNING
-------
The challenge does not state whether PQ is aggregated over the whole test set
or averaged per image. The two give different scores *and* different optimal
operating points, so both are implemented here (`PQAggregator.pq` and
`pq_per_image_mean`). Resolving which the organisers use is an open question
tracked in the project plan.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np

__all__ = [
    "PQComponents",
    "PQAggregator",
    "iou_matrix",
    "match_segments",
    "pq_single_image",
    "pq_per_image_mean",
]

# A segment is a sorted, unique array of flat pixel indices (int64).
Segment = np.ndarray


@dataclass(frozen=True)
class PQComponents:
    """The three sufficient statistics for PQ, plus derived quantities.

    Kept separate from the score itself so that partial results can be summed
    across images before dividing -- the dataset-aggregated PQ is *not* the
    mean of per-image PQs.
    """

    iou_sum: float = 0.0
    tp: int = 0
    fp: int = 0
    fn: int = 0

    def __add__(self, other: "PQComponents") -> "PQComponents":
        return PQComponents(
            self.iou_sum + other.iou_sum,
            self.tp + other.tp,
            self.fp + other.fp,
            self.fn + other.fn,
        )

    @property
    def denominator(self) -> float:
        return self.tp + 0.5 * self.fp + 0.5 * self.fn

    @property
    def pq(self) -> float:
        """PQ, or 0.0 when there is nothing to score (no GT and no prediction).

        The empty-empty case is genuinely undefined (0/0). Returning 0.0 would
        drag down a per-image mean and reward hallucinating segments on blank
        frames; callers that average per image should use `pq_per_image_mean`,
        which skips empty frames explicitly.
        """
        d = self.denominator
        return self.iou_sum / d if d > 0 else 0.0

    @property
    def is_empty(self) -> bool:
        return self.tp == 0 and self.fp == 0 and self.fn == 0

    @property
    def segmentation_quality(self) -> float:
        """SQ = mean IoU over matched pairs. PQ factorises as SQ * RQ."""
        return self.iou_sum / self.tp if self.tp > 0 else 0.0

    @property
    def recognition_quality(self) -> float:
        """RQ = F1 over segments. PQ factorises as SQ * RQ."""
        d = self.denominator
        return self.tp / d if d > 0 else 0.0


def iou_matrix(gt: Sequence[Segment], pred: Sequence[Segment]) -> np.ndarray:
    """Pairwise IoU between two sets of sparse segments.

    Returns an (len(gt), len(pred)) float64 array. Cost is
    O(sum(|pred|) * log(sum(|gt|))) via searchsorted against a single
    concatenated GT index, rather than the O(n_gt * n_pred) of naive pairwise
    intersection.
    """
    n_gt, n_pred = len(gt), len(pred)
    iou = np.zeros((n_gt, n_pred), dtype=np.float64)
    if n_gt == 0 or n_pred == 0:
        return iou

    gt_areas = np.array([g.size for g in gt], dtype=np.int64)
    pred_areas = np.array([p.size for p in pred], dtype=np.int64)

    # Flatten GT into one sorted index -> owning-segment lookup table.
    gt_pixels = np.concatenate(gt)
    gt_owner = np.repeat(np.arange(n_gt, dtype=np.int64), gt_areas)
    order = np.argsort(gt_pixels, kind="stable")
    gt_pixels = gt_pixels[order]
    gt_owner = gt_owner[order]

    for j, p in enumerate(pred):
        if p.size == 0:
            continue
        # Locate each predicted pixel in the GT index; keep exact hits only.
        pos = np.searchsorted(gt_pixels, p)
        pos = np.clip(pos, 0, gt_pixels.size - 1)
        hit = gt_pixels[pos] == p
        if not hit.any():
            continue
        inter = np.bincount(gt_owner[pos[hit]], minlength=n_gt).astype(np.float64)
        union = gt_areas + pred_areas[j] - inter
        np.divide(inter, union, out=iou[:, j], where=union > 0)

    return iou


def match_segments(
    iou: np.ndarray, threshold: float = 0.5
) -> tuple[list[tuple[int, int]], np.ndarray]:
    """Match GT to predictions at IoU > threshold.

    Returns (pairs, ious) where pairs is a list of (gt_i, pred_j). The
    comparison is strict (>), matching Kirillov et al.; at 0.5 the result is
    unique for disjoint segments, and greedy descending-IoU matching resolves
    the overlapping-prediction case optimally.
    """
    if iou.size == 0:
        return [], np.empty(0, dtype=np.float64)

    gt_idx, pred_idx = np.nonzero(iou > threshold)
    if gt_idx.size == 0:
        return [], np.empty(0, dtype=np.float64)

    values = iou[gt_idx, pred_idx]
    order = np.argsort(-values, kind="stable")

    used_gt: set[int] = set()
    used_pred: set[int] = set()
    pairs: list[tuple[int, int]] = []
    matched: list[float] = []
    for k in order:
        g, p = int(gt_idx[k]), int(pred_idx[k])
        if g in used_gt or p in used_pred:
            continue
        used_gt.add(g)
        used_pred.add(p)
        pairs.append((g, p))
        matched.append(float(values[k]))

    return pairs, np.array(matched, dtype=np.float64)


def pq_single_image(
    gt: Sequence[Segment], pred: Sequence[Segment], threshold: float = 0.5
) -> PQComponents:
    """PQ sufficient statistics for one image."""
    iou = iou_matrix(gt, pred)
    pairs, matched = match_segments(iou, threshold)
    return PQComponents(
        iou_sum=float(matched.sum()),
        tp=len(pairs),
        fp=len(pred) - len(pairs),
        fn=len(gt) - len(pairs),
    )


@dataclass
class PQAggregator:
    """Accumulates PQ over a dataset.

    Dataset-aggregated PQ pools TP/FP/FN across all images before dividing.
    This is what Kirillov et al. specify and is *not* the mean of per-image PQ.
    """

    total: PQComponents = field(default_factory=PQComponents)
    per_image: list[PQComponents] = field(default_factory=list)

    def update(
        self, gt: Sequence[Segment], pred: Sequence[Segment], threshold: float = 0.5
    ) -> PQComponents:
        comp = pq_single_image(gt, pred, threshold)
        self.total = self.total + comp
        self.per_image.append(comp)
        return comp

    @property
    def pq(self) -> float:
        return self.total.pq

    @property
    def pq_per_image(self) -> float:
        return pq_per_image_mean(self.per_image)


def pq_per_image_mean(components: Iterable[PQComponents]) -> float:
    """Mean of per-image PQ, skipping images with neither GT nor predictions.

    Skipping is deliberate: an empty-empty frame has an undefined PQ, and
    scoring it as 0.0 would make hallucinating a segment on a blank frame look
    like an improvement.
    """
    scores = [c.pq for c in components if not c.is_empty]
    return float(np.mean(scores)) if scores else 0.0
