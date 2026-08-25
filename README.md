# Solar Filament Instance Segmentation — GONG H-alpha

Instance segmentation of solar filaments in full-disk GONG H-alpha
observations, for the [Solar Filament Segmentation Challenge
2026](https://www.kaggle.com/competitions/filament-segmentation-2026)
(IEEE BigData Cup, sponsored by NSF/NSO).

The task is to delineate each filament as a separate instance. Scoring is
**Panoptic Quality** (Kirillov et al., CVPR 2019), which matches predicted to
ground-truth segments at `IoU > 0.5` and penalises fragmentation and
over-merging directly.

## Status

Phase 0 — metric, submission codec, and emit policy implemented and tested.
Model training begins in P1.

## Why this pipeline is shaped the way it is

Three properties of the metric and the dataset drive every design decision.

**1. A near-miss is worse than silence.** A predicted segment at `IoU = 0.49`
is charged as a false positive *and* leaves its ground-truth segment a false
negative — a denominator cost of 1.0 for zero numerator. Emitting nothing costs
0.5. So marginal-quality masks must be pushed above 0.5 IoU or suppressed;
leaving them at 0.49 is the worst available option. See
`tests/test_panoptic.py::test_near_miss_is_strictly_worse_than_abstaining`.

**2. The optimal emit threshold is far below 0.5.** Each emitted candidate
costs the same 0.5 of denominator whether or not it lands, so with `p` the
probability a candidate matches and `q` its expected IoU, emitting pays iff
`p * q > 0.5 * PQ`. At a working PQ of 0.4 that admits candidates with roughly
a 30% chance of being real. Every published filament model we surveyed operates
at precision far above recall — a good choice for Dice, a poor one for PQ.
See `src/filament/postproc/emit_policy.py` and `scripts/quantify_emit_edge.py`.

**3. Fragmentation and over-merging are catastrophic, not gradual.** Splitting
one filament into three equal pieces scores **0.0**, not 0.67. Merging two
equal adjacent filaments produces `IoU = 0.5` exactly, which fails the strict
threshold, and also scores **0.0**. Both are demonstrated as tests. This is why
instance decomposition is treated as the central problem rather than a
post-processing detail.

Ground truth helps here: MAGFiLO annotation rules require every filament mask
to be a **single connected component with no holes**, and exclude anything
beyond ±70° from central meridian. Both are enforced in post-processing rather
than left to the network.

## Layout

```
src/filament/
  data/         COCO parsing, dataset, group- and time-aware splits
  geometry/     solar disk fitting, heliographic coordinates, +/-70 deg mask
  models/       encoder + multi-task decoder (mask / spine / offsets)
  losses/       recall-weighted focal-Tversky, spine, offset
  postproc/     watershed decomposition, hole filling, emit policy
  metrics/      Panoptic Quality, IoU/Dice distributions, fragmentation counts
  submission/   COCO RLE codec and CSV writer
scripts/        analysis and reproduction entry points
tests/          correctness tests; see below
notebooks/      end-to-end pipeline demonstration
report/         technical report source
```

## Install

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

Python 3.11+ is required; the pinned versions in `requirements.txt` are those
verified on Python 3.14.

## Tests

```bash
python -m pytest tests/ -q
```

The suite is the project's correctness gate, not a formality:

- **Panoptic Quality** is checked against values computed by hand in each test
  body, so the suite fails if the implementation drifts *or* if it is "fixed"
  to agree with a buggy reference.
- **The RLE codec** is asserted byte-identical to `pycocotools`, including at
  the full 2048x2048 submission geometry. It is reimplemented in pure NumPy so
  the pipeline does not require a C extension to run.
- **The emit policy's** optimum is verified to satisfy the analytic fixed point
  `threshold = 0.5 * PQ`, and to beat every alternative subset.

## Reproducing the analysis

```bash
python scripts/quantify_emit_edge.py
```

Isolates the PQ gained by tuning the emit threshold rather than using 0.5,
under a detector calibrated to the operating point reported in the literature.

## References

- Kirillov et al., *Panoptic Segmentation*, CVPR 2019 — the evaluation metric.
- Ahmadzadeh et al., *A dataset of manually annotated filaments from H-alpha
  observations*, Scientific Data 11, 2024
  ([10.1038/s41597-024-03876-y](https://doi.org/10.1038/s41597-024-03876-y)) —
  MAGFiLO, the ground truth.
- Solomon et al., *EdgeAttNet: Towards Barb-Aware Filament Segmentation*,
  [arXiv:2509.02964](https://arxiv.org/abs/2509.02964) — the strongest
  published baseline on MAGFiLO.
- Zhu et al., *Flat U-Net*, ApJ 2025
  ([arXiv:2502.07259](https://arxiv.org/abs/2502.07259)) — ultralightweight
  filament segmentation.
- Hu et al., *A Modern ConvNet for Solar Filament Detection*,
  [arXiv:2607.24525](https://arxiv.org/abs/2607.24525) — MORDEN; uses DBSCAN
  clustering to reassemble fragmented predictions.

## Licence

Released under the MIT Licence, per the challenge's open-access policy.
