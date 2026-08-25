"""COCO run-length encoding, implemented directly from the cocoapi spec.

The challenge requires each predicted filament as one row carrying only the RLE
*counts* string (the size is fixed at 2048x2048). `pycocotools` is the
reference implementation, but it is a C extension that is awkward to install on
some platforms and is not always present in a judge's environment. This module
is a dependency-free equivalent; `tests/test_rle.py` cross-validates it against
`pycocotools` whenever that package is importable, so the two cannot silently
diverge.

Two details of the format are easy to get wrong and are worth stating:

1. Masks are flattened in **column-major (Fortran) order**, not row-major.
2. Counts are delta-encoded against the value two positions back, then written
   in a variable-length base-32 scheme with a continuation bit. Deltas may be
   negative, so the shift used during decoding must be arithmetic.

Reference: ``rleToString`` / ``rleFrString`` in ``cocoapi/common/maskApi.c``.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "MASK_SHAPE",
    "counts_to_string",
    "string_to_counts",
    "mask_to_rle",
    "rle_to_mask",
    "pixels_to_rle",
    "rle_to_pixels",
]

#: Every image in this challenge is 2048x2048, per the submission spec.
MASK_SHAPE: tuple[int, int] = (2048, 2048)


def counts_to_string(counts: list[int]) -> str:
    """Encode a run-length count list as a COCO counts string."""
    out: list[str] = []
    for i, count in enumerate(counts):
        x = int(count)
        if i > 2:
            x -= int(counts[i - 2])
        more = True
        while more:
            c = x & 0x1F
            x >>= 5  # Python's >> is arithmetic, matching signed C shifts
            more = (x != -1) if (c & 0x10) else (x != 0)
            if more:
                c |= 0x20
            out.append(chr(c + 48))
    return "".join(out)


def string_to_counts(s: str) -> list[int]:
    """Decode a COCO counts string back into a run-length count list."""
    counts: list[int] = []
    p, n = 0, len(s)
    while p < n:
        x, k, more = 0, 0, True
        while more:
            c = ord(s[p]) - 48
            x |= (c & 0x1F) << (5 * k)
            more = bool(c & 0x20)
            p += 1
            k += 1
            if not more and (c & 0x10):
                x |= -1 << (5 * k)  # sign-extend a negative delta
        if len(counts) > 2:
            x += counts[len(counts) - 2]
        counts.append(x)
    return counts


def mask_to_rle(mask: np.ndarray) -> str:
    """Encode a 2-D binary mask as a COCO counts string.

    The first run always counts zeros, so a mask whose first pixel (in
    column-major order) is set begins with an explicit zero-length run.
    """
    if mask.ndim != 2:
        raise ValueError(f"expected a 2-D mask, got shape {mask.shape}")

    flat = np.asarray(mask, dtype=bool).ravel(order="F")
    if flat.size == 0:
        return counts_to_string([])

    change = np.flatnonzero(np.diff(flat)) + 1
    bounds = np.concatenate(([0], change, [flat.size]))
    runs = np.diff(bounds).tolist()

    if flat[0]:
        runs = [0] + runs
    return counts_to_string(runs)


def rle_to_mask(counts: str, shape: tuple[int, int] = MASK_SHAPE) -> np.ndarray:
    """Decode a COCO counts string into a 2-D boolean mask."""
    runs = string_to_counts(counts)
    total = shape[0] * shape[1]

    flat = np.zeros(total, dtype=bool)
    position, value = 0, False
    for run in runs:
        if run < 0:
            raise ValueError(f"negative run length {run} in decoded counts")
        end = position + run
        if end > total:
            raise ValueError(
                f"RLE decodes to more than {total} pixels for shape {shape}"
            )
        if value:
            flat[position:end] = True
        position, value = end, not value

    if position != total:
        raise ValueError(
            f"RLE decodes to {position} pixels, expected {total} for shape {shape}"
        )
    return flat.reshape(shape, order="F")


def pixels_to_rle(
    pixels: np.ndarray, shape: tuple[int, int] = MASK_SHAPE
) -> str:
    """Encode a sparse segment (flat column-major indices) as a counts string.

    This is the bridge between the metric module -- which represents segments
    sparsely -- and the submission format.
    """
    mask = np.zeros(shape[0] * shape[1], dtype=bool)
    mask[np.asarray(pixels, dtype=np.int64)] = True
    return mask_to_rle(mask.reshape(shape, order="F"))


def rle_to_pixels(counts: str, shape: tuple[int, int] = MASK_SHAPE) -> np.ndarray:
    """Decode a counts string into sorted flat column-major pixel indices."""
    mask = rle_to_mask(counts, shape)
    return np.flatnonzero(mask.ravel(order="F")).astype(np.int64)
