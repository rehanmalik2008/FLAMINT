"""Loss functions for the multi-task filament decoder.

Three heads, three losses, combined by fixed weights in ``CombinedLoss``:

- **mask**: per-pixel filament probability. Foreground is ~0.5% of a frame
  (MAGFiLO's mean filament is ~2,250 px of 4.2M), so plain BCE would let the
  network minimise loss mostly by predicting background everywhere. Focal
  Tversky is used instead, with beta > alpha, deliberately weighting recall
  over precision -- the literature survey found every published filament
  model overshoots on precision (Flat U-Net: 0.93 / 0.69), which is a poor
  match for Panoptic Quality (0.5 denominator cost per miss, see
  ``filament.postproc.emit_policy``).
- **spine**: per-pixel probability of lying on a filament's 1-D skeleton.
  Even sparser than the mask, so it gets the same class of loss.
- **offset**: a 2-channel vector field pointing each foreground pixel toward
  its own filament's spine, supervised only where the mask is foreground
  (the value is meaningless on background, so it must not contribute there).

All three operate on raw logits, not probabilities, and combine sigmoid with
the loss computation internally for numerical stability (matching
``F.binary_cross_entropy_with_logits`` rather than a manual sigmoid+BCE).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

__all__ = ["focal_tversky_loss", "offset_loss", "CombinedLoss"]


def focal_tversky_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    *,
    alpha: float = 0.3,
    beta: float = 0.7,
    gamma: float = 0.75,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Focal Tversky loss for extreme foreground/background imbalance.

    The Tversky index generalises Dice with independent weights on false
    positives (``alpha``) and false negatives (``beta``); ``beta > alpha``
    penalises missed foreground more than spurious foreground, i.e. trades
    precision for recall. Raising ``(1 - Tversky)`` to ``gamma < 1`` (the
    "focal" part) increases the gradient on already-easy examples relatively
    less than on hard ones, which matters here because most pixels in a
    filament frame are trivially-correct background.

    Parameters
    ----------
    logits, target:
        Same shape, any number of leading dimensions (batch, optionally
        channel). ``target`` is expected in ``{0, 1}`` (or soft labels in
        ``[0, 1]``, which the Tversky index handles without modification).
    alpha, beta:
        False-positive and false-negative weights. Default 0.3/0.7 favours
        recall roughly 2:1, a starting point to be tuned once real validation
        PQ is measurable (P0's data audit); the ratio matters more than the
        absolute values.
    gamma:
        Focusing exponent; 1.0 recovers the plain Tversky loss.

    Notes
    -----
    Region-overlap losses (Tversky, Dice, IoU) are degenerate when a sample's
    target has zero foreground pixels: tp = 0 unconditionally, so the ratio
    collapses toward the epsilon floor -- and hence the loss toward its
    maximum -- almost independently of how good the prediction is. At
    MAGFiLO's ~0.5% foreground density this is not a corner case: tiled or
    cropped training batches will routinely include background-only samples.
    Per-sample, when a target has no foreground (``tp + fn == 0``), this
    function substitutes the mean predicted probability as that sample's loss
    contribution instead of the Tversky ratio -- a plain, well-behaved penalty
    that is near zero for a correctly-empty prediction and grows smoothly with
    spurious foreground, rather than saturating near 1.0 regardless of
    prediction quality.
    """
    if logits.shape != target.shape:
        raise ValueError(
            f"logits and target must share a shape, got {logits.shape} and "
            f"{target.shape}"
        )
    probs = torch.sigmoid(logits)
    dims = tuple(range(1, probs.dim())) if probs.dim() > 1 else (0,)

    tp = (probs * target).sum(dim=dims)
    fp = (probs * (1 - target)).sum(dim=dims)
    fn = ((1 - probs) * target).sum(dim=dims)

    tversky = (tp + eps) / (tp + alpha * fp + beta * fn + eps)
    tversky_loss = torch.pow(1.0 - tversky, gamma)

    # Degenerate-target fallback (see Notes): mean predicted probability over
    # the sample's spatial extent, computed per-item to match tversky_loss.
    mean_prob = probs.mean(dim=dims) if probs.dim() > 1 else probs.mean().unsqueeze(0)
    has_foreground = (tp + fn) > 0
    per_sample_loss = torch.where(has_foreground, tversky_loss, mean_prob)

    return per_sample_loss.mean()


def offset_loss(
    offsets: torch.Tensor, target_offsets: torch.Tensor, fg_mask: torch.Tensor
) -> torch.Tensor:
    """Masked MSE for the spine-offset field, supervised on foreground only.

    Parameters
    ----------
    offsets, target_offsets:
        Shape ``(B, 2, H, W)`` -- predicted and target (row, col) unit vectors
        toward each pixel's nearest spine point.
    fg_mask:
        Shape ``(B, H, W)`` or ``(B, 1, H, W)``, truthy where the ground-truth
        mask is foreground. Background offsets are undefined and must not
        contribute gradient; an all-background batch item contributes zero
        loss and zero gradient rather than NaN.
    """
    if offsets.shape != target_offsets.shape:
        raise ValueError(
            f"offsets and target_offsets must share a shape, got "
            f"{offsets.shape} and {target_offsets.shape}"
        )
    if offsets.dim() != 4 or offsets.shape[1] != 2:
        raise ValueError(f"expected offsets of shape (B, 2, H, W), got {offsets.shape}")

    if fg_mask.dim() == 3:
        fg_mask = fg_mask.unsqueeze(1)
    if fg_mask.shape[1] == 1:
        fg_mask = fg_mask.expand(-1, 2, -1, -1)

    fg_mask = fg_mask.to(offsets.dtype)
    n_fg = fg_mask.sum()
    if n_fg.item() == 0:
        return offsets.sum() * 0.0  # zero, but keeps offsets in the graph

    sq_err = (offsets - target_offsets) ** 2 * fg_mask
    return sq_err.sum() / n_fg


class CombinedLoss(nn.Module):
    """Weighted sum of the mask, spine, and offset losses.

    Weights are fixed at construction rather than learned (e.g. uncertainty
    weighting) so that loss values stay directly interpretable while the loop
    is being validated in P1; revisit once training is actually unstable or
    one head is starved, not before.
    """

    def __init__(
        self,
        mask_weight: float = 1.0,
        spine_weight: float = 0.5,
        offset_weight: float = 0.25,
        tversky_alpha: float = 0.3,
        tversky_beta: float = 0.7,
        tversky_gamma: float = 0.75,
    ):
        super().__init__()
        self.mask_weight = mask_weight
        self.spine_weight = spine_weight
        self.offset_weight = offset_weight
        self.tversky_alpha = tversky_alpha
        self.tversky_beta = tversky_beta
        self.tversky_gamma = tversky_gamma

    def forward(
        self,
        mask_logits: torch.Tensor,
        spine_logits: torch.Tensor,
        offsets: torch.Tensor,
        mask_target: torch.Tensor,
        spine_target: torch.Tensor,
        offset_target: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Returns a dict of the three components plus ``"total"``.

        Returning components (not just the sum) is deliberate: per-head loss
        curves are the fastest way to notice one head starving relative to
        the others during P1's initial training runs.
        """
        mask_loss = focal_tversky_loss(
            mask_logits,
            mask_target,
            alpha=self.tversky_alpha,
            beta=self.tversky_beta,
            gamma=self.tversky_gamma,
        )
        spine_loss = focal_tversky_loss(
            spine_logits,
            spine_target,
            alpha=self.tversky_alpha,
            beta=self.tversky_beta,
            gamma=self.tversky_gamma,
        )
        fg = mask_target > 0.5
        off_loss = offset_loss(offsets, offset_target, fg)

        total = (
            self.mask_weight * mask_loss
            + self.spine_weight * spine_loss
            + self.offset_weight * off_loss
        )
        return {
            "mask": mask_loss,
            "spine": spine_loss,
            "offset": off_loss,
            "total": total,
        }
