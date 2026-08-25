# Instructions for the Kaggle-connected session: real GPU training run

Paste this whole message into the other Claude session that has Kaggle MCP access.

---

I need you to create and run a Kaggle Notebook (kernel) that trains a model
for the `filament-segmentation-2026` competition on GPU. This is a **training
notebook**, separate from any future submission notebook — internet access
and full compute are fine here; the no-internet restriction only applies to
the notebook that generates the final `submission.csv`.

## Setup

1. Create a new Kaggle Notebook.
2. **Settings → Accelerator: GPU T4 x2** (or P100, whichever is offered).
3. **Settings → Internet: On** (needed to `git clone` from GitHub and `pip install`).
4. **Add Data → search "filament-segmentation-2026"** and attach the
   competition dataset as a notebook input.

## Notebook cells

**Cell 1 — clone and install:**
```python
!git clone https://github.com/rehanmalik2008/FLAMINT.git
%cd FLAMINT
!pip install -e . --quiet
```

**Cell 2 — check the actual mounted data path** (Kaggle's input mount name
can vary slightly from the dataset slug — run this first to confirm):
```python
!ls /kaggle/input/
!find /kaggle/input -iname "*.json" | head -5
!find /kaggle/input -iname "train_images" -type d
```
If the printed paths don't match `/kaggle/input/filament-segmentation-2026/`,
edit `KAGGLE_INPUT_ROOT` at the top of `notebooks/kaggle_train.py` to match
what you actually see, before running Cell 4.

**Cell 3 — precompute the training targets** (one-time cost, ~5-10 min on
Kaggle's CPU — this step doesn't use the GPU):
```python
!python scripts/precompute_targets.py --resolution 1024
```

**Cell 4 — the real training run:**
```python
!python notebooks/kaggle_train.py
```

This trains `FilamentUNet` at its real configuration — full ~1,150-image
training set, `base_channels=32`, 60 epochs — and checkpoints the
best-validation-loss model to `outputs/baseline_kaggle/best.pt`, plus a
`history.json` with the full loss curve.

## What I need back

1. **The full console output** of Cell 4 — every epoch's train/val loss line.
2. **How long the whole run actually took** (wall-clock), so I can plan
   future runs against Kaggle's 30 GPU-hours/week quota.
3. **The saved checkpoint file** (`outputs/baseline_kaggle/best.pt`) —
   either attach it directly, or if that's awkward, tell me its file size
   and I'll figure out the transfer from there (e.g. via a Kaggle Dataset
   output, or Kaggle's notebook "Save Version" output files).
4. If anything errors — paste the **full traceback**, not just the error
   line; several bugs in this codebase were only diagnosable from the full
   stack trace.

## If it runs too long or crashes on GPU memory

- Reduce `batch_size` (currently 8) in `notebooks/kaggle_train.py`'s
  `TrainConfig` first — that's the most memory-sensitive parameter.
- If it's `base_channels=32` causing a CUDA OOM, drop to 16 as a fallback,
  but flag this to me — it directly reduces the model's capacity.
- Checkpoints save every time validation loss improves, so a mid-run crash
  doesn't lose everything — the `best.pt` up to that point is still usable.
