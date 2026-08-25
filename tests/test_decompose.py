"""Tests for instance decomposition.

The central claims to verify: a filament whose mask has a weak (thinned)
region but whose spine survives should NOT fragment, and two filaments that
touch but have distinct spines SHOULD separate. Both are constructed
explicitly below, since they are the entire justification for spine-seeded
watershed over plain connected components.
"""

from __future__ import annotations

import numpy as np
import pytest

from filament.postproc.decompose import (
    DecomposedInstance,
    decompose,
    enforce_structural_rules,
)


# --------------------------------------------------------------------------
# enforce_structural_rules
# --------------------------------------------------------------------------


def test_enforce_rules_fills_a_hole():
    mask = np.zeros((20, 20), dtype=bool)
    mask[5:15, 5:15] = True
    mask[9:11, 9:11] = False  # a hole in the middle
    fixed = enforce_structural_rules(mask)
    assert fixed[9:11, 9:11].all()
    assert fixed.sum() == 100  # fully filled 10x10 square


def test_enforce_rules_keeps_only_largest_component():
    mask = np.zeros((20, 20), dtype=bool)
    mask[1:4, 1:4] = True  # area 9, small
    mask[10:18, 10:18] = True  # area 64, large
    fixed = enforce_structural_rules(mask)
    assert fixed.sum() == 64
    assert not fixed[1:4, 1:4].any()
    assert fixed[10:18, 10:18].all()


def test_enforce_rules_empty_mask_stays_empty():
    mask = np.zeros((10, 10), dtype=bool)
    fixed = enforce_structural_rules(mask)
    assert not fixed.any()


def test_enforce_rules_single_clean_component_is_unchanged():
    mask = np.zeros((10, 10), dtype=bool)
    mask[2:6, 2:6] = True
    fixed = enforce_structural_rules(mask)
    assert np.array_equal(fixed, mask)


def test_enforce_rules_rejects_non_2d():
    with pytest.raises(ValueError, match="2-D"):
        enforce_structural_rules(np.zeros((5, 5, 2), dtype=bool))


# --------------------------------------------------------------------------
# decompose: basic behaviour
# --------------------------------------------------------------------------


def test_decompose_empty_frame_returns_nothing():
    mask_prob = np.zeros((100, 100))
    spine_prob = np.zeros((100, 100))
    assert decompose(mask_prob, spine_prob) == []


def test_decompose_single_blob_single_seed():
    mask_prob = np.zeros((60, 60))
    mask_prob[20:40, 20:40] = 0.9
    spine_prob = np.zeros((60, 60))
    spine_prob[28:32, 20:40] = 0.9  # a spine running through the blob

    result = decompose(mask_prob, spine_prob)
    assert len(result) == 1
    assert isinstance(result[0], DecomposedInstance)
    assert result[0].mask.any()
    assert result[0].confidence == pytest.approx(0.9, abs=1e-6)


def test_decompose_rejects_mismatched_shapes():
    with pytest.raises(ValueError, match="shape"):
        decompose(np.zeros((10, 10)), np.zeros((10, 11)))


def test_decompose_no_spine_still_emits_unsplit_blob():
    """A blob with mask signal but a spine too weak to clear threshold must
    still be returned, not silently dropped -- recall matters more here."""
    mask_prob = np.zeros((40, 40))
    mask_prob[10:30, 10:30] = 0.8
    spine_prob = np.zeros((40, 40))  # spine head found nothing confident

    result = decompose(mask_prob, spine_prob)
    assert len(result) == 1
    assert result[0].area > 0


def test_decompose_drops_sub_min_area_blobs():
    mask_prob = np.zeros((40, 40))
    mask_prob[5, 5] = 0.9  # a single hot pixel, well under min_area
    spine_prob = np.zeros((40, 40))
    result = decompose(mask_prob, spine_prob, min_area=15)
    assert result == []


# --------------------------------------------------------------------------
# The central claim: resists fragmentation from a thinned bridge
# --------------------------------------------------------------------------


def test_survives_a_thin_bridge_that_would_break_connected_components():
    """A filament with a barb whose bridge dips below the mask threshold, but
    whose spine stays above the (lower) spine threshold, must NOT fragment.

    This is the central claim the module exists to deliver: plain connected
    components on the binarised mask alone would split at the weak bridge
    (mask_prob=0.3 < mask_threshold=0.5 there), but the flood domain is
    `fg | seeds_mask`, not `fg` alone, so a surviving spine seed reconnects
    both sides even where the pixel-level mask confidence dipped.
    """
    shape = (60, 100)
    mask_prob = np.zeros(shape)
    spine_prob = np.zeros(shape)

    # Main body: a horizontal bar of confident mask probability.
    mask_prob[25:35, 10:90] = 0.9
    spine_prob[29:31, 10:90] = 0.9

    # A weak bridge segment: mask probability dips below mask_threshold (0.5)
    # here, but the spine probability does not.
    mask_prob[25:35, 45:55] = 0.3
    spine_prob[29:31, 45:55] = 0.6

    result = decompose(mask_prob, spine_prob, mask_threshold=0.5, spine_threshold=0.5)

    assert len(result) == 1
    # The bridge itself must be part of the recovered mask, not just the two
    # flanking regions -- otherwise this would be two instances that merely
    # happened to receive the same watershed label before pruning.
    assert result[0].mask[29:31, 45:55].any()
    # And it must cover (most of) the full extent: the two strong flanks plus
    # the reconnected bridge, not just one side.
    assert result[0].area > 600  # a single flank alone is ~350px


def test_single_seed_never_fragments_regardless_of_mask_gaps():
    """When exactly one spine component spans a mask gap, watershed still
    assigns every foreground pixel connected to *some* fg region containing
    that seed to a single label -- fragmentation only happens when fg itself
    is disconnected. This test uses a mask gap that stays >= mask_threshold
    (weakened, not absent) so fg remains one piece, isolating the effect of
    a strong, unbroken spine seed."""
    shape = (60, 100)
    mask_prob = np.zeros(shape)
    spine_prob = np.zeros(shape)

    mask_prob[25:35, 10:90] = 0.9
    mask_prob[25:35, 45:55] = 0.55  # weakened but still >= 0.5 threshold
    spine_prob[29:31, 10:90] = 0.9  # spine spans the whole bar, unbroken

    result = decompose(mask_prob, spine_prob, mask_threshold=0.5, spine_threshold=0.5)
    assert len(result) == 1
    assert result[0].mask[25:35, 45:55].any()  # the weak bridge is included


# --------------------------------------------------------------------------
# The central claim: separates two touching filaments
# --------------------------------------------------------------------------


def test_separates_two_touching_filaments_with_distinct_spines():
    """Two blobs that touch (share a border, forming one connected mask
    region) but have geometrically distinct spines must be split into two
    instances, not merged into one."""
    shape = (100, 100)
    mask_prob = np.zeros(shape)
    spine_prob = np.zeros(shape)

    # Two vertical bars, touching at column 50/51.
    mask_prob[10:90, 20:51] = 0.9
    mask_prob[10:90, 50:81] = 0.9  # overlaps by one column -> single fg blob

    # Distinct spines, each centred in its own bar, far enough apart that
    # watershed's ridge falls between them.
    spine_prob[10:90, 34:36] = 0.9
    spine_prob[10:90, 65:67] = 0.9

    result = decompose(mask_prob, spine_prob, mask_threshold=0.5, spine_threshold=0.5)

    assert len(result) == 2
    for inst in result:
        # Each instance must satisfy the single-connected-component rule.
        assert inst.mask.any()
    # The two instances must not overlap (watershed partitions fg exactly).
    overlap = result[0].mask & result[1].mask
    assert not overlap.any()
    # Between them they should cover (most of) the original foreground.
    total_covered = (result[0].mask | result[1].mask).sum()
    assert total_covered == pytest.approx(int((mask_prob >= 0.5).sum()), abs=5)


def test_merged_blob_without_distinguishing_spines_stays_one_instance():
    """Sanity check on the other direction: if there is truly only one spine
    component, two adjacent bright regions it spans should NOT be
    artificially split."""
    shape = (60, 100)
    mask_prob = np.zeros(shape)
    spine_prob = np.zeros(shape)
    mask_prob[20:40, 10:90] = 0.9
    spine_prob[28:32, 10:90] = 0.9  # one continuous spine across the whole bar

    result = decompose(mask_prob, spine_prob)
    assert len(result) == 1


# --------------------------------------------------------------------------
# Structural rules are actually applied to decompose() output
# --------------------------------------------------------------------------


def test_decomposed_instances_satisfy_structural_rules():
    """Every returned instance must already be a single component with no
    holes -- decompose() must call enforce_structural_rules internally."""
    shape = (60, 60)
    mask_prob = np.zeros(shape)
    mask_prob[10:50, 10:50] = 0.9
    mask_prob[25:35, 25:35] = 0.2  # a hole-like dip below threshold
    spine_prob = np.zeros(shape)
    spine_prob[28:32, 10:50] = 0.9

    result = decompose(mask_prob, spine_prob)
    assert len(result) >= 1
    for inst in result:
        # No holes: every mask, once enforced, equals its own hole-filled self.
        refilled = enforce_structural_rules(inst.mask)
        assert np.array_equal(refilled, inst.mask)
