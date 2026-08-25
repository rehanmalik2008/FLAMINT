"""Tests for the PQ-optimal emit policy.

`test_optimum_satisfies_fixed_point` is the important one: it verifies
numerically that the threshold found by the exhaustive sweep is the same one
predicted analytically by `threshold = 0.5 * PQ`. If that identity ever breaks,
the reasoning the whole strategy rests on is wrong.
"""

from __future__ import annotations

import numpy as np
import pytest

from filament.postproc.emit_policy import (
    optimal_expected_policy,
    sweep_threshold,
)


# --------------------------------------------------------------------------
# The analytic optimum
# --------------------------------------------------------------------------


def test_optimum_satisfies_fixed_point():
    """The optimal cut must sit at 0.5 * PQ -- the claim the plan rests on."""
    rng = np.random.default_rng(7)
    scores = rng.uniform(0.0, 0.9, size=400)
    policy = optimal_expected_policy(scores, n_ground_truth=100)

    # Every admitted candidate must beat 0.5 * PQ, and every rejected one
    # must fail it. That is exactly the marginal condition.
    cut = 0.5 * policy.expected_pq
    admitted = np.sort(scores)[::-1][: policy.n_emitted]
    rejected = np.sort(scores)[::-1][policy.n_emitted :]

    assert np.all(admitted > cut)
    if rejected.size:
        assert np.all(rejected <= cut)


def test_optimum_beats_every_other_prefix():
    """The reported optimum is global, not a local maximum."""
    rng = np.random.default_rng(11)
    scores = rng.uniform(0.0, 0.8, size=200)
    n_gt = 60
    policy = optimal_expected_policy(scores, n_ground_truth=n_gt)

    ordered = np.sort(scores)[::-1]
    for k in range(len(ordered) + 1):
        pq_k = ordered[:k].sum() / (0.5 * n_gt + 0.5 * k)
        assert pq_k <= policy.expected_pq + 1e-12


def test_optimal_threshold_is_far_below_one_half():
    """The headline consequence: emit candidates a 0.5 cut would discard.

    Scores here are p*q with q ~= 0.7, so a score of 0.2 corresponds to a
    filament with a ~29% chance of being real. Those should be emitted.
    """
    rng = np.random.default_rng(3)
    # A realistic mix: a few confident detections, a long uncertain tail.
    scores = np.concatenate(
        [rng.uniform(0.6, 0.75, size=40), rng.uniform(0.05, 0.4, size=160)]
    )
    policy = optimal_expected_policy(scores, n_ground_truth=100)

    assert policy.expected_pq > 0.0
    assert policy.threshold < 0.5
    # Strictly more than the confidently-detected 40 get emitted.
    assert policy.n_emitted > 40


def test_all_scores_worthless_emits_nothing():
    policy = optimal_expected_policy([0.0, 0.0, 0.0], n_ground_truth=10)
    assert policy.n_emitted == 0
    assert policy.expected_pq == 0.0


def test_perfect_scores_emit_everything():
    policy = optimal_expected_policy([1.0] * 10, n_ground_truth=10)
    assert policy.n_emitted == 10
    assert policy.expected_pq == pytest.approx(1.0)


def test_no_candidates():
    policy = optimal_expected_policy([], n_ground_truth=5)
    assert policy.n_emitted == 0
    assert policy.expected_pq == 0.0


def test_rejects_negative_ground_truth_count():
    with pytest.raises(ValueError, match="non-negative"):
        optimal_expected_policy([0.5], n_ground_truth=-1)


# --------------------------------------------------------------------------
# The empirical sweep
# --------------------------------------------------------------------------


def test_sweep_recovers_hand_computed_pq():
    """Three candidates, one threshold, arithmetic done by hand.

    At threshold 0.4 all three are emitted: two clear IoU > 0.5 (0.8, 0.6),
    one does not (0.3) and is a false positive. With 4 ground-truth segments,
    FN = 4 - 2 = 2.
      PQ = (0.8 + 0.6) / (2 + 0.5*1 + 0.5*2) = 1.4 / 3.5 = 0.4
    """
    sweep = sweep_threshold(
        confidences=[0.9, 0.7, 0.5],
        ious=[0.8, 0.6, 0.3],
        n_ground_truth=4,
        grid=[0.4],
    )
    assert sweep.pq[0] == pytest.approx(1.4 / 3.5)


def test_sweep_prefers_dropping_a_hopeless_candidate():
    """Suppressing a candidate that cannot clear IoU 0.5 must raise PQ."""
    sweep = sweep_threshold(
        confidences=[0.9, 0.1],
        ious=[0.8, 0.2],  # the second can never match
        n_ground_truth=1,
        grid=[0.05, 0.5],
    )
    # Emitting both: 0.8 / (1 + 0.5) = 0.5333
    assert sweep.pq[0] == pytest.approx(0.8 / 1.5)
    # Emitting only the good one: 0.8 / 1 = 0.8
    assert sweep.pq[1] == pytest.approx(0.8)
    assert sweep.best_threshold == 0.5


def test_sweep_reports_gain_over_default_threshold():
    """A low-confidence but genuinely matching candidate should be recovered."""
    sweep = sweep_threshold(
        confidences=[0.9, 0.2],
        ious=[0.9, 0.7],  # the low-confidence one *does* match
        n_ground_truth=2,
    )
    # Default 0.5 cut emits one: 0.9 / (1 + 0.5*1) = 0.6
    assert sweep.baseline_pq == pytest.approx(0.9 / 1.5)
    # Emitting both: (0.9 + 0.7) / 2 = 0.8
    assert sweep.best_pq == pytest.approx(1.6 / 2.0)
    assert sweep.gain_over_default > 0
    assert sweep.best_threshold <= 0.2


def test_sweep_rejects_misaligned_inputs():
    with pytest.raises(ValueError, match="align"):
        sweep_threshold([0.5, 0.6], [0.5], n_ground_truth=1)


def test_sweep_default_grid_covers_all_confidences():
    sweep = sweep_threshold([0.1, 0.4, 0.9], [0.6, 0.6, 0.6], n_ground_truth=3)
    assert set(sweep.thresholds) == {0.1, 0.4, 0.9}
    # All three match, so emitting everything is optimal: PQ = 1.8/3 = 0.6
    assert sweep.best_pq == pytest.approx(0.6)
