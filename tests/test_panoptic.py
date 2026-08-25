"""Correctness tests for the Panoptic Quality implementation.

Every expected value here is computed by hand in the test body, so the suite
fails if the implementation drifts *or* if someone "fixes" it to agree with a
buggy reference. Several tests double as executable documentation of the
strategic claims in the project plan (the near-miss cliff, the fragmentation
and over-merge penalties).
"""

from __future__ import annotations

import numpy as np
import pytest

from filament.metrics.panoptic import (
    PQAggregator,
    PQComponents,
    iou_matrix,
    match_segments,
    pq_per_image_mean,
    pq_single_image,
)


def seg(start: int, stop: int) -> np.ndarray:
    """A segment occupying the half-open flat-index range [start, stop)."""
    return np.arange(start, stop, dtype=np.int64)


# --------------------------------------------------------------------------
# IoU
# --------------------------------------------------------------------------


def test_iou_matrix_basic():
    gt = [seg(0, 100)]
    pred = [seg(20, 120)]
    # intersection = |[20, 100)| = 80 ; union = 100 + 100 - 80 = 120
    assert iou_matrix(gt, pred)[0, 0] == pytest.approx(80 / 120)


def test_iou_identical_is_one():
    gt = [seg(0, 50)]
    assert iou_matrix(gt, [seg(0, 50)])[0, 0] == pytest.approx(1.0)


def test_iou_disjoint_is_zero():
    assert iou_matrix([seg(0, 50)], [seg(100, 150)])[0, 0] == 0.0


def test_iou_matrix_shape_and_empty_sets():
    assert iou_matrix([], [seg(0, 10)]).shape == (0, 1)
    assert iou_matrix([seg(0, 10)], []).shape == (1, 0)
    assert iou_matrix([], []).size == 0


def test_iou_matrix_multiple_segments_is_pairwise():
    gt = [seg(0, 100), seg(100, 200)]
    pred = [seg(50, 150)]
    iou = iou_matrix(gt, pred)
    # pred overlaps each GT in exactly 50 px; union = 100 + 100 - 50 = 150
    assert iou[0, 0] == pytest.approx(50 / 150)
    assert iou[1, 0] == pytest.approx(50 / 150)


def test_iou_unsorted_prediction_indices():
    """Segments must not be required to arrive sorted."""
    rng = np.random.default_rng(0)
    p = seg(20, 120)
    rng.shuffle(p)
    assert iou_matrix([seg(0, 100)], [p])[0, 0] == pytest.approx(80 / 120)


# --------------------------------------------------------------------------
# Matching
# --------------------------------------------------------------------------


def test_match_requires_strictly_greater_than_threshold():
    """IoU of exactly 0.5 must NOT match -- Kirillov et al. use strict >."""
    # |g| = |p| = 100, intersection 50 -> union 150. Use a tuned pair instead:
    # g = [0,100), p = [0,50) u [100,150) gives inter 50, union 150 = 1/3.
    # For exactly 0.5 we need inter/union = 0.5: g = [0,100), p = [0,75)
    # -> inter 75, union 100 -> 0.75. Construct directly on the matrix instead.
    iou = np.array([[0.5]])
    pairs, _ = match_segments(iou, threshold=0.5)
    assert pairs == []

    iou = np.array([[0.5 + 1e-9]])
    pairs, _ = match_segments(iou, threshold=0.5)
    assert pairs == [(0, 0)]


def test_match_is_greedy_by_descending_iou_when_predictions_overlap():
    """Overlapping predictions can both clear 0.5; the better one must win."""
    gt = [seg(0, 100)]
    pred = [seg(0, 80), seg(0, 90)]  # IoU 0.8 and 0.9
    iou = iou_matrix(gt, pred)
    assert iou[0, 0] == pytest.approx(0.8)
    assert iou[0, 1] == pytest.approx(0.9)

    pairs, matched = match_segments(iou)
    assert pairs == [(0, 1)]  # the 0.9 prediction
    assert matched[0] == pytest.approx(0.9)


def test_match_one_gt_per_prediction():
    """A single prediction cannot satisfy two ground-truth segments."""
    iou = np.array([[0.9], [0.7]])
    pairs, _ = match_segments(iou)
    assert len(pairs) == 1
    assert pairs[0] == (0, 0)


# --------------------------------------------------------------------------
# PQ, hand-computed
# --------------------------------------------------------------------------


def test_pq_perfect_prediction_is_one():
    gt = [seg(0, 100), seg(200, 300)]
    comp = pq_single_image(gt, list(gt))
    assert comp.tp == 2 and comp.fp == 0 and comp.fn == 0
    assert comp.pq == pytest.approx(1.0)


def test_pq_no_predictions_is_zero_and_all_fn():
    gt = [seg(0, 100), seg(200, 300)]
    comp = pq_single_image(gt, [])
    assert (comp.tp, comp.fp, comp.fn) == (0, 0, 2)
    assert comp.denominator == pytest.approx(1.0)
    assert comp.pq == 0.0


def test_pq_hand_computed_mixed_case():
    """One TP, one near-miss, one missed GT, one spurious prediction.

    g0 vs p0 : inter 80, union 120 -> IoU 2/3          -> TP
    g1 vs p1 : inter 40, union 160 -> IoU 0.25         -> FN + FP
    g2       : unpredicted                             -> FN
    p2       : spurious                                -> FP
    PQ = (2/3) / (1 + 0.5*2 + 0.5*2) = (2/3)/3 = 0.2222...
    """
    gt = [seg(0, 100), seg(200, 300), seg(400, 500)]
    pred = [seg(20, 120), seg(260, 360), seg(600, 700)]

    comp = pq_single_image(gt, pred)
    assert (comp.tp, comp.fp, comp.fn) == (1, 2, 2)
    assert comp.iou_sum == pytest.approx(2 / 3)
    assert comp.denominator == pytest.approx(3.0)
    assert comp.pq == pytest.approx((2 / 3) / 3)


def test_pq_factorises_into_sq_times_rq():
    gt = [seg(0, 100), seg(200, 300), seg(400, 500)]
    pred = [seg(20, 120), seg(260, 360), seg(600, 700)]
    comp = pq_single_image(gt, pred)
    assert comp.segmentation_quality * comp.recognition_quality == pytest.approx(
        comp.pq
    )


# --------------------------------------------------------------------------
# The strategic claims from the plan, as executable proofs
# --------------------------------------------------------------------------


def test_near_miss_is_strictly_worse_than_abstaining():
    """Edge 1: a prediction at IoU < 0.5 scores worse than emitting nothing.

    A near-miss is charged as FP *and* FN (denominator +1.0), whereas
    abstaining is charged as FN only (denominator +0.5). Both earn zero
    numerator, so silence strictly dominates a doomed guess.
    """
    anchor_gt, anchor_pred = seg(0, 100), seg(0, 100)  # a clean TP, IoU 1.0
    marginal_gt = seg(1000, 1100)

    # Near miss: inter 40, union 160 -> IoU 0.25
    near_miss = pq_single_image(
        [anchor_gt, marginal_gt], [anchor_pred, seg(1060, 1160)]
    )
    abstain = pq_single_image([anchor_gt, marginal_gt], [anchor_pred])

    assert (near_miss.tp, near_miss.fp, near_miss.fn) == (1, 1, 1)
    assert (abstain.tp, abstain.fp, abstain.fn) == (1, 0, 1)

    assert near_miss.pq == pytest.approx(1.0 / 2.0)  # 1.0 / (1 + 0.5 + 0.5)
    assert abstain.pq == pytest.approx(1.0 / 1.5)  # 1.0 / (1 + 0.5)
    assert abstain.pq > near_miss.pq


def test_fragmentation_is_catastrophic():
    """Splitting one filament into three pieces scores zero, not two-thirds."""
    gt = [seg(0, 300)]
    whole = pq_single_image(gt, [seg(0, 300)])
    fragmented = pq_single_image(gt, [seg(0, 100), seg(100, 200), seg(200, 300)])

    assert whole.pq == pytest.approx(1.0)
    # Each fragment has IoU 100/300 = 0.333 -> nothing matches.
    assert (fragmented.tp, fragmented.fp, fragmented.fn) == (0, 3, 1)
    assert fragmented.pq == 0.0


def test_over_merge_is_catastrophic():
    """Merging two adjacent filaments yields IoU exactly 0.5 -- which fails."""
    gt = [seg(0, 100), seg(100, 200)]
    merged = pq_single_image(gt, [seg(0, 200)])

    iou = iou_matrix(gt, [seg(0, 200)])
    assert iou[0, 0] == pytest.approx(0.5)  # exactly at the threshold
    assert (merged.tp, merged.fp, merged.fn) == (0, 1, 2)
    assert merged.pq == 0.0


# --------------------------------------------------------------------------
# Aggregation
# --------------------------------------------------------------------------


def test_dataset_pq_is_not_the_mean_of_per_image_pq():
    """The two aggregations genuinely differ -- which is why both exist."""
    agg = PQAggregator()
    agg.update([seg(0, 100)], [seg(0, 100)])  # image 1: PQ = 1.0
    agg.update(
        [seg(0, 100), seg(200, 300), seg(400, 500)],
        [seg(20, 120)],
    )  # image 2: TP=1 (IoU 2/3), FN=2 -> PQ = (2/3)/2 = 0.3333

    assert agg.per_image[0].pq == pytest.approx(1.0)
    assert agg.per_image[1].pq == pytest.approx((2 / 3) / 2)

    # Pooled: iou_sum = 1 + 2/3, TP = 2, FN = 2 -> (5/3) / 3
    assert agg.pq == pytest.approx((1 + 2 / 3) / 3)
    assert agg.pq_per_image == pytest.approx((1.0 + (2 / 3) / 2) / 2)
    assert agg.pq != pytest.approx(agg.pq_per_image)


def test_empty_images_are_skipped_by_per_image_mean():
    """An image with no GT and no prediction has undefined PQ, not zero."""
    empty = PQComponents()
    assert empty.is_empty
    assert pq_per_image_mean([empty]) == 0.0  # nothing scoreable at all

    perfect = PQComponents(iou_sum=1.0, tp=1)
    # The empty frame must not drag the mean from 1.0 down to 0.5.
    assert pq_per_image_mean([perfect, empty]) == pytest.approx(1.0)


def test_components_add_is_associative_and_pure():
    a = PQComponents(iou_sum=1.0, tp=1, fp=2, fn=3)
    b = PQComponents(iou_sum=0.5, tp=1, fp=0, fn=1)
    total = a + b
    assert (total.iou_sum, total.tp, total.fp, total.fn) == (1.5, 2, 2, 4)
    # Originals unchanged (frozen dataclass).
    assert (a.tp, a.fp, a.fn) == (1, 2, 3)
