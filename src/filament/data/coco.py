"""Parsing MAGFiLO's COCO-style annotation JSON.

Reconciled against the real competition file
(``MAGFiLO_1.0_Annotations_kaggle2026_train.json``, 1,154 images / 8,199
annotations) during P0's data audit. Several details differ from what the
Nature/PMC paper summary implied, and are recorded here rather than silently
patched over:

- **IDs are strings, not integers.** Image ids look like
  ``"040301-20140609195854Bh"`` and annotation ids are UUIDs. The leading
  numeric prefix before the ``-`` is a group/reviewer code: three images
  ``"050101-20111116063134Lh"``, ``"050102-...-Lh"``, ``"050103-...-Lh"``
  were observed sharing one timestamp suffix, i.e. one physical frame
  independently re-annotated three times. This is the real mechanism behind
  the duplicate-observation risk the project plan flagged from the paper.
- **``date_captured`` is present directly** (``"2014-06-09 19:58:54"``), so
  observation grouping and time-ordering no longer need to be inferred from a
  filename regex -- it is still kept as a fallback for robustness, but
  ``date_captured`` is authoritative when present.
- **Four categories, not three**: ``Left`` (1), ``Right`` (2),
  ``Unidentifiable`` (3), and ``Ambiguous`` (4) -- the paper's summary did not
  mention the fourth. Unused for mask training either way; retained per
  record for a possible future auxiliary signal.
- **Annotations carry ``area``, ``iscrowd``, and ``spine`` fields** in
  addition to ``segmentation``, ``bbox``, ``category_id``. ``spine`` is a
  flat polyline in the same coordinate format as ``segmentation`` -- this is
  the ground-truth spine data the rules-compliance note in
  ``filament.data.targets`` and ``filament.postproc.decompose`` flagged as
  ambiguous to train on. It is parsed and retained here (so it is available
  for analysis, e.g. comparing against our own mask-derived skeletons) but is
  **not** used to build training targets by default; see those modules.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

__all__ = [
    "ImageRecord",
    "FilamentAnnotation",
    "MagfiloDataset",
    "load_magfilo",
    "polygon_to_mask",
]

#: GONG H-alpha filenames encode a UTC timestamp, e.g. "20140609195854Bh.jpeg"
#: (YYYYMMDDHHMMSS + a station/instrument code). Confirmed against the real
#: file's image records during P0's data audit.
_TIMESTAMP_RE = re.compile(r"(\d{14})")

#: The reference format for the `date_captured` field, confirmed against the
#: real file: "2014-06-09 19:58:54".
_DATE_CAPTURED_FMT = "%Y-%m-%d %H:%M:%S"


@dataclass(frozen=True)
class ImageRecord:
    """One entry from the COCO ``images`` collection.

    ``image_id`` is the JSON's own id string (e.g.
    ``"040301-20140609195854Bh"``), used verbatim rather than re-keyed to an
    integer, so it round-trips unambiguously back to the source file.
    """

    image_id: str
    file_name: str
    height: int
    width: int
    date_captured: datetime | None
    observation_key: str  # stable across duplicate re-annotations of one frame


@dataclass(frozen=True)
class FilamentAnnotation:
    """One entry from the COCO ``annotations`` collection."""

    annotation_id: str
    image_id: str
    category_id: int  # 1=Left, 2=Right, 3=Unidentifiable, 4=Ambiguous (chirality)
    segmentation: list[list[float]]  # polygon(s), COCO [x0,y0,x1,y1,...] format
    area: float | None = None
    iscrowd: int | None = None
    bbox: tuple[float, float, float, float] | None = None
    spine: list[float] | None = None  # flat polyline [x0,y0,x1,y1,...]; see module docstring


@dataclass(frozen=True)
class MagfiloDataset:
    """Parsed COCO annotation file, indexed for fast lookup."""

    images: dict[str, ImageRecord]
    annotations_by_image: dict[str, list[FilamentAnnotation]]

    def __len__(self) -> int:
        return len(self.images)

    def n_annotations(self) -> int:
        return sum(len(v) for v in self.annotations_by_image.values())

    def observation_keys(self) -> dict[str, str]:
        """image_id -> observation_key, for building leak-free splits."""
        return {img_id: img.observation_key for img_id, img in self.images.items()}


def _parse_date_captured(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.strptime(value, _DATE_CAPTURED_FMT)
    except ValueError:
        return None  # unrecognised format; fall back to filename parsing


def _observation_key(file_name: str, date_captured: datetime | None) -> str:
    """Extract a key stable across duplicate re-annotations of one frame.

    Prefers ``date_captured`` (ISO-formatted, so it also sorts chronologically
    as a string) when parseable; falls back to a timestamp regex on the file
    name, and finally to the bare file stem, so parsing never silently drops
    an image -- worst case it just fails to deduplicate or time-order it,
    which ``filament.data.splits.assert_no_leakage`` would then catch.
    """
    if date_captured is not None:
        return date_captured.strftime(_DATE_CAPTURED_FMT)
    match = _TIMESTAMP_RE.search(file_name)
    if match:
        return match.group(1)
    return Path(file_name).stem.lower()


def load_magfilo(json_path: str | Path) -> MagfiloDataset:
    """Parse a MAGFiLO-format COCO JSON file.

    Raises
    ------
    KeyError
        If the top-level ``images`` or ``annotations`` keys are missing --
        surfaced immediately rather than producing a silently empty dataset,
        since a wrong/renamed key is the most likely first failure mode
        against a differently-versioned file.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        raw: dict[str, Any] = json.load(f)

    if "images" not in raw or "annotations" not in raw:
        raise KeyError(
            f"expected top-level 'images' and 'annotations' keys, got "
            f"{sorted(raw.keys())}"
        )

    images: dict[str, ImageRecord] = {}
    for img in raw["images"]:
        date_captured = _parse_date_captured(img.get("date_captured"))
        images[img["id"]] = ImageRecord(
            image_id=img["id"],
            file_name=img["file_name"],
            height=img["height"],
            width=img["width"],
            date_captured=date_captured,
            observation_key=_observation_key(img["file_name"], date_captured),
        )

    annotations_by_image: dict[str, list[FilamentAnnotation]] = {
        img_id: [] for img_id in images
    }
    for ann in raw["annotations"]:
        record = FilamentAnnotation(
            annotation_id=ann["id"],
            image_id=ann["image_id"],
            category_id=ann.get("category_id", 0),
            segmentation=ann["segmentation"],
            area=ann.get("area"),
            iscrowd=ann.get("iscrowd"),
            bbox=tuple(ann["bbox"]) if "bbox" in ann else None,
            spine=ann.get("spine"),
        )
        annotations_by_image.setdefault(record.image_id, []).append(record)

    return MagfiloDataset(images=images, annotations_by_image=annotations_by_image)


def polygon_to_mask(
    segmentation: list[list[float]], height: int, width: int
) -> np.ndarray:
    """Rasterise a COCO polygon list into a boolean mask.

    Implemented with a scanline point-in-polygon fill (even-odd rule) rather
    than depending on ``pycocotools.mask.frPyObjects`` for this step, so
    dataset loading has no hard dependency on the C extension being present
    -- consistent with the rest of the codebase's dependency-light stance.
    Cross-validated against ``pycocotools`` in
    ``tests/test_coco.py::test_polygon_to_mask_matches_pycocotools`` and
    against the real file's own ``area`` field in
    ``tests/test_coco_real_data.py``.

    A filament with multiple polygon parts (disjoint pieces in the raw
    annotation) is unioned into one mask; MAGFiLO's own annotation rule
    requires each filament to be single-piece, so multiple parts should not
    occur in practice -- if they do, this is a reasonable, size-order-
    independent way to still produce something coherent to train against
    rather than raising.
    """
    mask = np.zeros((height, width), dtype=bool)
    for poly in segmentation:
        coords = np.asarray(poly, dtype=np.float64).reshape(-1, 2)
        if coords.shape[0] < 3:
            continue
        mask |= _rasterize_polygon(coords, height, width)
    return mask


def _rasterize_polygon(coords: np.ndarray, height: int, width: int) -> np.ndarray:
    """Even-odd scanline fill for a single polygon, coords as (x, y) pairs."""
    mask = np.zeros((height, width), dtype=bool)
    xs, ys = coords[:, 0], coords[:, 1]
    y_min = max(int(np.floor(ys.min())), 0)
    y_max = min(int(np.ceil(ys.max())), height - 1)

    n = len(coords)
    for row in range(y_min, y_max + 1):
        y = row + 0.5  # sample at pixel centre
        xs_hit = []
        for i in range(n):
            x0, y0 = xs[i], ys[i]
            x1, y1 = xs[(i + 1) % n], ys[(i + 1) % n]
            if (y0 <= y < y1) or (y1 <= y < y0):
                t = (y - y0) / (y1 - y0)
                xs_hit.append(x0 + t * (x1 - x0))
        xs_hit.sort()
        for j in range(0, len(xs_hit) - 1, 2):
            x_start = max(int(np.ceil(xs_hit[j] - 0.5)), 0)
            x_end = min(int(np.floor(xs_hit[j + 1] - 0.5)), width - 1)
            if x_start <= x_end:
                mask[row, x_start : x_end + 1] = True
    return mask
