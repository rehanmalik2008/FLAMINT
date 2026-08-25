"""Tests for the multi-task loss functions.

Beyond shape/gradient sanity, these check the specific claim the plan makes:
Focal Tversky with beta > alpha penalises false negatives more than false
positives, i.e. it is biased toward recall -- which is the whole reason it was
chosen over plain BCE or Dice.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from filament.losses.segmentation import (  # noqa: E402
    CombinedLoss,
    focal_tversky_loss,
    offset_loss,
)


# --------------------------------------------------------------------------
# focal_tversky_loss
# --------------------------------------------------------------------------


def test_perfect_prediction_has_near_zero_loss():
    target = (torch.rand(4, 32, 32) > 0.5).float()
    logits = (target * 2 - 1) * 20.0  # saturated logits matching target exactly
    loss = focal_tversky_loss(logits, target)
    assert loss.item() < 1e-3


def test_inverted_prediction_has_high_loss():
    target = (torch.rand(4, 32, 32) > 0.5).float()
    logits = -(target * 2 - 1) * 20.0  # saturated logits, exactly wrong
    loss = focal_tversky_loss(logits, target)
    assert loss.item() > 0.9


def test_beta_greater_than_alpha_penalises_false_negatives_more():
    """With an all-foreground target, a prediction with false negatives should
    be penalised more heavily than one with an equal count of false positives,
    when beta > alpha."""
    target = torch.ones(1, 10, 10)

    # False-negative case: half the pixels predicted as background.
    fn_logits = torch.full((1, 10, 10), -10.0)
    fn_logits[:, :5, :] = 10.0

    # This scenario only has false negatives (target is all-1, so there is no
    # way to construct a pure false-positive case against it); instead compare
    # against a symmetric loss (alpha == beta) to show beta>alpha increases the
    # penalty specifically for this false-negative-heavy prediction.
    loss_recall_weighted = focal_tversky_loss(fn_logits, target, alpha=0.3, beta=0.7)
    loss_symmetric = focal_tversky_loss(fn_logits, target, alpha=0.5, beta=0.5)

    assert loss_recall_weighted.item() > loss_symmetric.item()


def test_alpha_greater_than_beta_penalises_false_positives_more():
    """Mirror check: false positives should be penalised more when
    alpha > beta than when symmetric.

    Uses a target with a small foreground region so tp > 0 and the Tversky
    ratio (not the degenerate all-background fallback, which is alpha/beta
    agnostic by design) is actually exercised.
    """
    target = torch.zeros(1, 10, 10)
    target[:, :2, :2] = 1.0  # a small foreground patch, perfectly predicted

    fp_logits = torch.full((1, 10, 10), -10.0)
    fp_logits[:, :2, :2] = 10.0  # true positives here
    fp_logits[:, 5:, 5:] = 10.0  # false positives elsewhere

    loss_precision_weighted = focal_tversky_loss(fp_logits, target, alpha=0.7, beta=0.3)
    loss_symmetric = focal_tversky_loss(fp_logits, target, alpha=0.5, beta=0.5)

    assert loss_precision_weighted.item() > loss_symmetric.item()


def test_rejects_shape_mismatch():
    with pytest.raises(ValueError, match="shape"):
        focal_tversky_loss(torch.zeros(2, 4, 4), torch.zeros(2, 5, 5))


def test_gradient_flows_through_focal_tversky():
    logits = torch.randn(1, 8, 8, requires_grad=True)
    target = (torch.rand(1, 8, 8) > 0.5).float()
    loss = focal_tversky_loss(logits, target)
    loss.backward()
    assert logits.grad is not None
    assert torch.any(logits.grad != 0)


def test_all_background_target_no_predictions_gives_low_loss():
    """A correctly-empty prediction on an empty target should score well.

    This is the degenerate-Tversky case: tp=0 unconditionally when the target
    is all-background, so the raw ratio collapses toward the epsilon floor
    regardless of prediction quality. The function must detect this and fall
    back to a well-behaved penalty (see Notes in focal_tversky_loss) rather
    than reporting a near-maximal loss for a correct empty prediction.
    """
    target = torch.zeros(1, 16, 16)
    logits = torch.full((1, 16, 16), -10.0)
    loss = focal_tversky_loss(logits, target)
    assert loss.item() < 0.1


def test_all_background_target_spurious_prediction_gives_high_loss():
    """The degenerate-target fallback must still penalise a bad prediction --
    it should not become a free pass for hallucinating foreground."""
    target = torch.zeros(1, 16, 16)
    logits = torch.full((1, 16, 16), 10.0)  # confidently (wrongly) foreground
    loss = focal_tversky_loss(logits, target)
    assert loss.item() > 0.8


def test_degenerate_target_fallback_only_applies_per_sample():
    """In a mixed batch, a background-only sample must use the fallback while
    a sample with real foreground still uses the Tversky ratio -- the two
    must not contaminate each other via a batch-wide reduction."""
    # Sample 0: all background, correctly predicted -> should score low.
    # Sample 1: has foreground, predicted perfectly -> should also score low.
    target = torch.zeros(2, 16, 16)
    target[1, :4, :4] = 1.0
    logits = torch.full((2, 16, 16), -10.0)
    logits[1, :4, :4] = 10.0

    loss = focal_tversky_loss(logits, target)
    assert loss.item() < 0.1


# --------------------------------------------------------------------------
# offset_loss
# --------------------------------------------------------------------------


def test_offset_loss_zero_when_predictions_match_target():
    target_off = torch.randn(2, 2, 16, 16)
    fg = torch.rand(2, 16, 16) > 0.3
    loss = offset_loss(target_off.clone(), target_off, fg)
    assert loss.item() == pytest.approx(0.0, abs=1e-6)


def test_offset_loss_ignores_background_errors():
    """A large error confined entirely to background pixels must not
    contribute to the loss."""
    pred = torch.zeros(1, 2, 8, 8)
    target = torch.zeros(1, 2, 8, 8)
    fg = torch.zeros(1, 8, 8, dtype=torch.bool)
    fg[:, :4, :4] = True  # foreground is the top-left quadrant only

    # Corrupt only the background region.
    pred[:, :, 4:, 4:] = 100.0

    loss = offset_loss(pred, target, fg)
    assert loss.item() == pytest.approx(0.0, abs=1e-6)


def test_offset_loss_all_background_gives_zero_not_nan():
    pred = torch.randn(1, 2, 8, 8, requires_grad=True)
    target = torch.randn(1, 2, 8, 8)
    fg = torch.zeros(1, 8, 8, dtype=torch.bool)

    loss = offset_loss(pred, target, fg)
    assert loss.item() == 0.0
    assert torch.isfinite(loss)

    loss.backward()  # must not raise, even though n_fg == 0
    assert pred.grad is not None


def test_offset_loss_accepts_channel_dim_mask():
    """fg_mask may arrive as (B, 1, H, W) as well as (B, H, W)."""
    target_off = torch.randn(1, 2, 8, 8)
    fg_3d = torch.rand(1, 8, 8) > 0.5
    fg_4d = fg_3d.unsqueeze(1)

    loss_3d = offset_loss(target_off.clone(), target_off, fg_3d)
    loss_4d = offset_loss(target_off.clone(), target_off, fg_4d)
    assert loss_3d.item() == pytest.approx(loss_4d.item())


def test_offset_loss_rejects_shape_mismatch():
    with pytest.raises(ValueError, match="shape"):
        offset_loss(torch.zeros(1, 2, 4, 4), torch.zeros(1, 2, 5, 5), torch.zeros(1, 4, 4))


def test_offset_loss_rejects_wrong_channel_count():
    with pytest.raises(ValueError, match="2, H, W"):
        offset_loss(torch.zeros(1, 3, 4, 4), torch.zeros(1, 3, 4, 4), torch.zeros(1, 4, 4))


# --------------------------------------------------------------------------
# CombinedLoss
# --------------------------------------------------------------------------


def test_combined_loss_returns_all_components():
    combined = CombinedLoss()
    b, h, w = 2, 16, 16
    out = combined(
        mask_logits=torch.randn(b, 1, h, w),
        spine_logits=torch.randn(b, 1, h, w),
        offsets=torch.randn(b, 2, h, w),
        mask_target=(torch.rand(b, 1, h, w) > 0.7).float(),
        spine_target=(torch.rand(b, 1, h, w) > 0.9).float(),
        offset_target=torch.randn(b, 2, h, w),
    )
    assert set(out.keys()) == {"mask", "spine", "offset", "total"}
    for v in out.values():
        assert torch.isfinite(v)


def test_combined_loss_total_is_weighted_sum():
    combined = CombinedLoss(mask_weight=2.0, spine_weight=3.0, offset_weight=0.0)
    b, h, w = 1, 8, 8
    mask_target = (torch.rand(b, 1, h, w) > 0.7).float()
    out = combined(
        mask_logits=torch.randn(b, 1, h, w),
        spine_logits=torch.randn(b, 1, h, w),
        offsets=torch.randn(b, 2, h, w),
        mask_target=mask_target,
        spine_target=(torch.rand(b, 1, h, w) > 0.9).float(),
        offset_target=torch.randn(b, 2, h, w),
    )
    expected = 2.0 * out["mask"] + 3.0 * out["spine"] + 0.0 * out["offset"]
    assert out["total"].item() == pytest.approx(expected.item(), rel=1e-5)


def test_combined_loss_gradient_flows_to_all_inputs():
    combined = CombinedLoss()
    b, h, w = 1, 8, 8
    mask_logits = torch.randn(b, 1, h, w, requires_grad=True)
    spine_logits = torch.randn(b, 1, h, w, requires_grad=True)
    offsets = torch.randn(b, 2, h, w, requires_grad=True)

    out = combined(
        mask_logits=mask_logits,
        spine_logits=spine_logits,
        offsets=offsets,
        mask_target=(torch.rand(b, 1, h, w) > 0.7).float(),
        spine_target=(torch.rand(b, 1, h, w) > 0.9).float(),
        offset_target=torch.randn(b, 2, h, w),
    )
    out["total"].backward()

    assert mask_logits.grad is not None and torch.any(mask_logits.grad != 0)
    assert spine_logits.grad is not None and torch.any(spine_logits.grad != 0)
    assert offsets.grad is not None and torch.any(offsets.grad != 0)
