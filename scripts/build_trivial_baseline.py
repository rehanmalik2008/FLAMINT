"""Build a throwaway, content-free baseline submission.

Purpose: NOT to score well. This has no access to actual pixel data -- only
the test set's file listing -- so it cannot look at the images at all. Its
only job is to close P0's submission-plumbing gate:

  1. Confirm the RLE/CSV format is accepted by the real Kaggle scorer.
  2. Get back one real, calibrated PQ number to check filament.metrics.panoptic
     against (see filament/metrics/panoptic.py's module warning about the
     aggregation ambiguity -- a real score is the fastest way to notice if our
     assumption about per-image-vs-pooled aggregation, or the IoU threshold
     convention, is wrong).
  3. Exercise filament.submission.rle end to end against the real scorer, not
     just against pycocotools locally.

Strategy: predict one fixed-size, fixed-position elliptical blob near the
image centre for every test image, using the dataset's own reported mean
filament dimensions (bbox ~13,100 px^2 measured from the real training file
in P0's audit) as the blob's rough scale. This is expected to score very
close to zero PQ -- a blob that happens to land on a real filament by pure
chance, at random orientation and location, is a low-probability event -- and
that is fine. A deliberately bad, diagnostic-only submission, per the
project's stated strategy: don't optimise the first submission, use it to
validate the pipeline.

Run:  python scripts/build_trivial_baseline.py [--test-list PATH] [--out PATH]
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from filament.submission.rle import MASK_SHAPE, mask_to_rle  # noqa: E402

DEFAULT_TEST_LIST = Path(__file__).resolve().parents[1] / "data" / "test_images_list.csv"
DEFAULT_OUT = Path(__file__).resolve().parents[1] / "data" / "submission_baseline.csv"

# A fixed ellipse near frame centre, sized from the real training file's
# measured mean bbox area (~13,100 px^2, P0 audit) -- semi-axes chosen so
# pi*a*b matches that area at a roughly 2:1 aspect ratio (filaments are
# elongated, not circular).
_H, _W = MASK_SHAPE
_CENTER = (_H // 2, _W // 2)
_SEMI_MAJOR = 91  # along columns
_SEMI_MINOR = 46  # along rows
# pi * 91 * 46 =~ 13,150 px^2, matching the measured mean bbox area.


def fixed_blob_mask() -> np.ndarray:
    rows, cols = np.indices(MASK_SHAPE)
    r0, c0 = _CENTER
    normalized = ((rows - r0) / _SEMI_MINOR) ** 2 + ((cols - c0) / _SEMI_MAJOR) ** 2
    return normalized <= 1.0


def parse_stems(test_list_path: Path) -> list[str]:
    """Extract the bare filament-id stem (filename without extension) from
    each row of the Kaggle-provided file listing. The listing's first column
    is a path like 'MAGFiLO_1.0_Kaggle_2026/test/test_images/<stem>.jpeg'."""
    stems = []
    with open(test_list_path, "r", encoding="utf-8", newline="") as f:
        for row in csv.reader(f):
            if not row:
                continue
            path = row[0].strip()
            stem = Path(path).stem
            stems.append(stem)
    if not stems:
        raise ValueError(f"no rows found in {test_list_path}")
    return stems


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-list", type=Path, default=DEFAULT_TEST_LIST)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    stems = parse_stems(args.test_list)
    mask = fixed_blob_mask()
    rle = mask_to_rle(mask)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filament_id", "segmentation_rle"])
        for stem in stems:
            writer.writerow([f"{stem}_1", rle])

    print(f"wrote {len(stems)} rows to {args.out}")
    print(f"blob area: {int(mask.sum())} px  (target ~13,100 px^2 mean bbox area)")
    print(
        "This is a diagnostic-only baseline -- expect a very low PQ. Its "
        "purpose is validating the submission format and getting one real "
        "score back to check filament.metrics.panoptic against."
    )


if __name__ == "__main__":
    main()
