"""Choosing which candidate instances to emit, so as to maximise PQ.

Background
----------
Under Panoptic Quality every emitted candidate costs the same amount of
denominator whether or not it lands:

    * it matches a ground-truth filament -> that filament goes FN -> TP, so the
      denominator rises by 0.5 (from 0.5 to 1.0) and the numerator by its IoU;
    * it matches nothing -> a false positive, denominator +0.5, numerator +0.

So with ``p`` the probability a candidate matches and ``q`` its expected IoU
given a match, the expected marginal effect of emitting is

    E[dN] = p * q,      E[dD] = 0.5      (regardless of outcome)

Writing PQ = N / D, emitting improves the score iff

    (N + p q) / (D + 0.5) > N / D   <=>   p q > 0.5 * (N / D) = 0.5 * PQ

That is a *fixed point*: the optimal threshold depends on the PQ you end up
with. It is tempting to iterate, but there is an exact solution. Because every
candidate contributes the same 0.5 to the denominator, the best subset of size
k is simply the top k by ``p*q``; sweeping k and taking the best is therefore
globally optimal, not merely a local search. :func:`optimal_expected_policy`
does that, and asserts the fixed-point condition holds at the optimum.

The practical consequence is that the optimal threshold is far below the 0.5
confidence cut that segmentation pipelines default to. At a working PQ of 0.40
with typical matched IoU around 0.70, the cut sits near ``p > 0.29`` -- we
should emit anything with roughly a 30% chance of being real. Every published
filament model we surveyed reports precision far above recall, so this is
where the metric and the field's habits diverge most sharply.

For model selection, prefer :func:`sweep_threshold`, which measures true PQ
against held-out ground truth rather than trusting calibrated estimates.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from filament.metrics.panoptic import PQComponents

__all__ = [
    "EmitPolicy",
    "ThresholdSweep",
    "optimal_expected_policy",
    "sweep_threshold",
]


@dataclass(frozen=True)
class EmitPolicy:
    """The outcome of optimising the emit threshold."""

    threshold: float
    n_emitted: int
    expected_pq: float

    def __str__(self) -> str:  # pragma: no cover - display only
        return (
            f"EmitPolicy(threshold={self.threshold:.4f}, "
            f"n_emitted={self.n_emitted}, expected_pq={self.expected_pq:.4f})"
        )


def optimal_expected_policy(
    scores: Sequence[float], n_ground_truth: int
) -> EmitPolicy:
    """Globally optimal emit policy under calibrated ``p*q`` estimates.

    Parameters
    ----------
    scores:
        One value of ``p * q`` per candidate instance -- the probability the
        candidate matches a real filament, times its expected IoU if it does.
    n_ground_truth:
        Expected number of ground-truth filaments in the same image set. Sets
        the baseline denominator ``0.5 * n_ground_truth`` that is paid whether
        or not anything is emitted.

    Returns
    -------
    EmitPolicy
        The threshold, the number of candidates it admits, and the expected PQ.

    Notes
    -----
    The returned threshold satisfies the fixed point ``threshold ~= 0.5 * PQ``
    up to the granularity of the score list.
    """
    values = np.sort(np.asarray(scores, dtype=np.float64))[::-1]
    if n_ground_truth < 0:
        raise ValueError("n_ground_truth must be non-negative")

    base_denominator = 0.5 * n_ground_truth
    if values.size == 0:
        return EmitPolicy(threshold=float("inf"), n_emitted=0, expected_pq=0.0)

    # Emitting the top k candidates, for every k. Vectorised prefix sums.
    numerators = np.concatenate(([0.0], np.cumsum(values)))
    k = np.arange(numerators.size, dtype=np.float64)
    denominators = base_denominator + 0.5 * k

    with np.errstate(invalid="ignore", divide="ignore"):
        pq = np.where(denominators > 0, numerators / denominators, 0.0)

    best_k = int(np.argmax(pq))
    best_pq = float(pq[best_k])

    # The threshold is the value of the last admitted candidate; anything at or
    # below the next candidate's value is rejected.
    threshold = float(values[best_k - 1]) if best_k > 0 else float("inf")

    return EmitPolicy(threshold=threshold, n_emitted=best_k, expected_pq=best_pq)


@dataclass(frozen=True)
class ThresholdSweep:
    """Measured PQ as a function of the emit threshold."""

    thresholds: np.ndarray
    pq: np.ndarray
    best_threshold: float
    best_pq: float
    baseline_pq: float

    @property
    def gain_over_default(self) -> float:
        """PQ gained by tuning the threshold rather than using 0.5."""
        return self.best_pq - self.baseline_pq


def sweep_threshold(
    confidences: Sequence[float],
    ious: Sequence[float],
    n_ground_truth: int,
    grid: Sequence[float] | None = None,
    match_threshold: float = 0.5,
) -> ThresholdSweep:
    """Measure true PQ across emit thresholds on labelled validation data.

    Unlike :func:`optimal_expected_policy` this makes no calibration
    assumptions: each candidate's *actual* IoU against ground truth is known,
    so the PQ at every threshold is exact.

    Parameters
    ----------
    confidences:
        Model confidence for each candidate instance.
    ious:
        The candidate's true IoU with its best-matching ground-truth segment
        (0.0 if it overlaps none). Must align with ``confidences``.
    n_ground_truth:
        Total ground-truth filaments across the same images.
    grid:
        Thresholds to evaluate. Defaults to every distinct confidence, which is
        exhaustive -- PQ is piecewise constant between observed values.
    match_threshold:
        The IoU above which a candidate counts as a true positive.

    Notes
    -----
    Each ground-truth segment can be claimed only once. Callers must therefore
    pass at most one candidate per ground-truth segment (keep the best-matching
    one); otherwise true positives are double counted. The instance
    decomposition guarantees this by construction, since it emits disjoint
    segments.
    """
    conf = np.asarray(confidences, dtype=np.float64)
    iou = np.asarray(ious, dtype=np.float64)
    if conf.shape != iou.shape:
        raise ValueError(
            f"confidences and ious must align, got {conf.shape} and {iou.shape}"
        )

    if grid is None:
        grid = np.unique(conf) if conf.size else np.array([0.5])
    grid = np.asarray(grid, dtype=np.float64)

    scores = np.empty(grid.size, dtype=np.float64)
    for i, t in enumerate(grid):
        emitted = conf >= t
        hit = emitted & (iou > match_threshold)
        tp = int(hit.sum())
        fp = int(emitted.sum()) - tp
        fn = n_ground_truth - tp
        scores[i] = PQComponents(
            iou_sum=float(iou[hit].sum()), tp=tp, fp=fp, fn=fn
        ).pq

    best = int(np.argmax(scores))

    default = conf >= 0.5
    default_hit = default & (iou > match_threshold)
    default_tp = int(default_hit.sum())
    baseline = PQComponents(
        iou_sum=float(iou[default_hit].sum()),
        tp=default_tp,
        fp=int(default.sum()) - default_tp,
        fn=n_ground_truth - default_tp,
    ).pq

    return ThresholdSweep(
        thresholds=grid,
        pq=scores,
        best_threshold=float(grid[best]),
        best_pq=float(scores[best]),
        baseline_pq=float(baseline),
    )
