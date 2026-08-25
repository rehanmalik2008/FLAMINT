"""Round-trip and cross-validation tests for the COCO RLE codec.

The decisive test is `test_matches_pycocotools`, which asserts byte-identical
output against the reference C implementation. It is skipped when
`pycocotools` is unavailable, so the rest of the suite still runs on a bare
environment -- but CI must have it installed for the comparison to mean
anything.
"""

from __future__ import annotations

import numpy as np
import pytest

from filament.submission.rle import (
    counts_to_string,
    mask_to_rle,
    pixels_to_rle,
    rle_to_mask,
    rle_to_pixels,
    string_to_counts,
)

try:  # pragma: no cover - environment dependent
    from pycocotools import mask as pycocotools_mask  # type: ignore
    HAS_PYCOCOTOOLS = True
except ImportError:  # pragma: no cover - environment dependent
    HAS_PYCOCOTOOLS = False


SMALL = (16, 24)  # deliberately non-square, to catch F/C order mistakes


def random_mask(shape, density=0.1, seed=0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.random(shape) < density


def blobby_mask(shape, seed=0) -> np.ndarray:
    """A mask with long runs, closer to real filaments than salt-and-pepper."""
    rng = np.random.default_rng(seed)
    mask = np.zeros(shape, dtype=bool)
    for _ in range(4):
        r = rng.integers(0, shape[0] - 4)
        c = rng.integers(0, shape[1] - 6)
        mask[r : r + 4, c : c + 6] = True
    return mask


# --------------------------------------------------------------------------
# Counts-string codec
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "counts",
    [
        [],
        [0],
        [5, 3, 5],
        [0, 10, 20, 30, 40],
        [1, 1, 1, 1, 1, 1, 1, 1],
        [100000, 1, 99999, 2],
        [0, 1, 0, 1, 0, 1],
        [7, 2**20, 3, 2**16, 5],
    ],
)
def test_counts_string_round_trip(counts):
    assert string_to_counts(counts_to_string(counts)) == counts


def test_counts_string_handles_negative_deltas():
    """Delta encoding produces negatives when runs shrink; sign must survive."""
    counts = [1000, 1000, 5, 5, 1, 1]  # deltas go sharply negative at index 4
    encoded = counts_to_string(counts)
    assert string_to_counts(encoded) == counts


# --------------------------------------------------------------------------
# Mask round-trip
# --------------------------------------------------------------------------


@pytest.mark.parametrize("seed", range(5))
def test_mask_round_trip_random(seed):
    mask = random_mask(SMALL, seed=seed)
    assert np.array_equal(rle_to_mask(mask_to_rle(mask), SMALL), mask)


@pytest.mark.parametrize("seed", range(5))
def test_mask_round_trip_blobby(seed):
    mask = blobby_mask(SMALL, seed=seed)
    assert np.array_equal(rle_to_mask(mask_to_rle(mask), SMALL), mask)


def test_mask_round_trip_all_zeros_and_all_ones():
    for mask in (np.zeros(SMALL, bool), np.ones(SMALL, bool)):
        assert np.array_equal(rle_to_mask(mask_to_rle(mask), SMALL), mask)


def test_first_pixel_set_emits_leading_zero_run():
    """A mask starting with a set pixel must begin with a zero-length run."""
    mask = np.zeros(SMALL, bool)
    mask[0, 0] = True
    assert string_to_counts(mask_to_rle(mask))[0] == 0
    assert np.array_equal(rle_to_mask(mask_to_rle(mask), SMALL), mask)


def test_encoding_is_column_major():
    """Setting a full column must give a single run, unlike a full row."""
    column = np.zeros(SMALL, bool)
    column[:, 0] = True
    # [0-run, 16-run of ones, remainder] -> the ones are contiguous in F order
    assert string_to_counts(mask_to_rle(column))[1] == SMALL[0]

    row = np.zeros(SMALL, bool)
    row[0, :] = True
    # In F order a full row is maximally fragmented: many alternating runs.
    assert len(string_to_counts(mask_to_rle(row))) > SMALL[1]


def test_pixels_round_trip():
    pixels = np.array([0, 1, 2, 100, 101, 383], dtype=np.int64)
    counts = pixels_to_rle(pixels, SMALL)
    assert np.array_equal(rle_to_pixels(counts, SMALL), pixels)


def test_rle_to_mask_rejects_wrong_total():
    bad = counts_to_string([5, 5])  # only 10 pixels, shape needs 384
    with pytest.raises(ValueError, match="expected"):
        rle_to_mask(bad, SMALL)


def test_mask_to_rle_rejects_non_2d():
    with pytest.raises(ValueError, match="2-D"):
        mask_to_rle(np.zeros((4, 4, 3), bool))


# --------------------------------------------------------------------------
# Cross-validation against the reference implementation
# --------------------------------------------------------------------------


@pytest.mark.skipif(not HAS_PYCOCOTOOLS, reason="pycocotools not installed")
@pytest.mark.parametrize("seed", range(8))
def test_matches_pycocotools(seed):
    """Our counts string must be byte-identical to cocoapi's."""
    mask = blobby_mask(SMALL, seed=seed)
    reference = pycocotools_mask.encode(np.asfortranarray(mask.astype(np.uint8)))
    assert mask_to_rle(mask) == reference["counts"].decode("ascii")


@pytest.mark.skipif(not HAS_PYCOCOTOOLS, reason="pycocotools not installed")
def test_matches_pycocotools_at_full_resolution():
    """Exercise the real 2048x2048 submission geometry, not just a toy."""
    mask = np.zeros((2048, 2048), dtype=bool)
    mask[900:1000, 1000:1100] = True
    mask[500:520, 300:900] = True  # a long thin structure, like a filament
    reference = pycocotools_mask.encode(np.asfortranarray(mask.astype(np.uint8)))
    assert mask_to_rle(mask) == reference["counts"].decode("ascii")
