"""Quantify the PQ gained by tuning the emit threshold instead of using 0.5.

Every published filament segmentation model we surveyed reports precision far
above recall -- Flat U-Net (arXiv:2502.07259) reports 0.93 / 0.69. That is a
sensible operating point for Dice, and a poor one for Panoptic Quality, which
charges 0.5 for each missed filament.

This script builds a transparent detector model calibrated to that operating
point, then measures PQ at the default 0.5 cut against the PQ-optimal cut. It
makes no claim to predict our final score; it isolates *one* effect, so that
the size of the effect is a number rather than an intuition.

Assumptions, stated so they can be argued with:

  * Each ground-truth filament has a latent "detectability" in [0, 1].
  * Model confidence is detectability plus Gaussian noise.
  * A candidate's IoU rises with detectability; below a floor it cannot clear
    the 0.5 matching threshold no matter how confident the model is.
  * A number of spurious candidates are produced with low confidence and no
    matching ground truth.

Run:  python scripts/quantify_emit_edge.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from filament.postproc.emit_policy import sweep_threshold  # noqa: E402

N_GROUND_TRUTH = 2000
N_SPURIOUS = 400
SEED = 20260825


def simulate(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Return (confidences, true_ious) for all candidate instances."""
    # Detectability: most filaments are clear, a meaningful tail is marginal.
    detectability = rng.beta(a=2.2, b=1.4, size=N_GROUND_TRUTH)

    confidence = np.clip(detectability + rng.normal(0, 0.12, N_GROUND_TRUTH), 0, 1)

    # IoU grows with detectability. The offset is tuned so that marginal
    # filaments land near the 0.5 cliff, which is where the metric is decided.
    iou = np.clip(0.30 + 0.62 * detectability + rng.normal(0, 0.07, N_GROUND_TRUTH), 0, 1)

    # Spurious detections: low confidence, never match.
    spurious_conf = np.clip(rng.beta(1.5, 6.0, N_SPURIOUS), 0, 1)
    spurious_iou = np.zeros(N_SPURIOUS)

    return (
        np.concatenate([confidence, spurious_conf]),
        np.concatenate([iou, spurious_iou]),
    )


def operating_point(conf: np.ndarray, iou: np.ndarray, threshold: float) -> dict:
    emitted = conf >= threshold
    hit = emitted & (iou > 0.5)
    tp = int(hit.sum())
    fp = int(emitted.sum()) - tp
    return {
        "emitted": int(emitted.sum()),
        "tp": tp,
        "fp": fp,
        "fn": N_GROUND_TRUTH - tp,
        "precision": tp / max(int(emitted.sum()), 1),
        "recall": tp / N_GROUND_TRUTH,
    }


def main() -> None:
    rng = np.random.default_rng(SEED)
    conf, iou = simulate(rng)

    sweep = sweep_threshold(conf, iou, n_ground_truth=N_GROUND_TRUTH)

    default = operating_point(conf, iou, 0.5)
    tuned = operating_point(conf, iou, sweep.best_threshold)

    print("Detector calibration check (at the default 0.5 cut)")
    print(f"  instance precision : {default['precision']:.3f}")
    print(f"  instance recall    : {default['recall']:.3f}")
    print("  (literature reports ~0.93 / ~0.69 for pixel-level Dice models)")
    print()

    print(f"{'':22s}{'default 0.5':>14s}{'PQ-optimal':>14s}")
    print("-" * 50)
    print(f"{'threshold':22s}{0.5:>14.3f}{sweep.best_threshold:>14.3f}")
    print(f"{'segments emitted':22s}{default['emitted']:>14d}{tuned['emitted']:>14d}")
    print(f"{'true positives':22s}{default['tp']:>14d}{tuned['tp']:>14d}")
    print(f"{'false positives':22s}{default['fp']:>14d}{tuned['fp']:>14d}")
    print(f"{'false negatives':22s}{default['fn']:>14d}{tuned['fn']:>14d}")
    print(f"{'precision':22s}{default['precision']:>14.3f}{tuned['precision']:>14.3f}")
    print(f"{'recall':22s}{default['recall']:>14.3f}{tuned['recall']:>14.3f}")
    print("-" * 50)
    print(f"{'PQ':22s}{sweep.baseline_pq:>14.4f}{sweep.best_pq:>14.4f}")
    print()

    gain = sweep.gain_over_default
    rel = 100.0 * gain / sweep.baseline_pq if sweep.baseline_pq else float("nan")
    print(f"Absolute PQ gain : {gain:+.4f}")
    print(f"Relative PQ gain : {rel:+.1f}%")
    print()

    # Verify the fixed point. The condition is on p*q for the *marginal*
    # candidate -- those sitting at the cut -- not on raw confidence, which is
    # only a noisy proxy for p. Measure p and q empirically in a narrow band
    # around the optimal threshold and check against 0.5 * PQ*.
    band = np.abs(conf - sweep.best_threshold) < 0.02
    p_marginal = float((iou[band] > 0.5).mean())
    band_hit = band & (iou > 0.5)
    q_marginal = float(iou[band_hit].mean()) if band_hit.any() else 0.0
    predicted = 0.5 * sweep.best_pq

    print("Fixed-point check at the margin")
    print(f"  candidates in band     : {int(band.sum())}")
    print(f"  p (match rate)         : {p_marginal:.4f}")
    print(f"  q (mean IoU if match)  : {q_marginal:.4f}")
    print(f"  p*q observed           : {p_marginal * q_marginal:.4f}")
    print(f"  0.5 * PQ* predicted    : {predicted:.4f}")
    if predicted > 0:
        print(f"  ratio                  : {p_marginal * q_marginal / predicted:.3f}")


if __name__ == "__main__":
    main()
