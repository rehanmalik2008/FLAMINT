"""Tests for time-grouped, leak-free splitting.

The central property under test: no observation_key (a proxy for "the same
physical GONG frame") may appear in more than one split, even when it has
multiple annotation rows under different image ids.
"""

from __future__ import annotations

import pytest

from filament.data.coco import ImageRecord, MagfiloDataset
from filament.data.splits import assert_no_leakage, time_grouped_split


def make_dataset(n_unique: int, duplicates_at: set[int] | None = None) -> MagfiloDataset:
    """Build a synthetic dataset with `n_unique` distinct timestamps, evenly
    spaced by one day, where `duplicates_at` indices get a second image id
    for the same frame (simulating MAGFiLO's re-annotation duplicates)."""
    duplicates_at = duplicates_at or set()
    images: dict[int, ImageRecord] = {}
    next_id = 1
    for i in range(n_unique):
        ts = f"201501{(i % 28) + 1:02d}120000"  # a fake but unique-per-i timestamp
        # Ensure genuine lexicographic ordering across i by encoding i into
        # the month/day/seconds fields distinctly.
        ts = f"{2015 + i // 300:04d}{(i // 25) % 12 + 1:02d}{(i % 25) + 1:02d}120000"
        fname_a = f"{ts}Mh_a.jpg"
        images[next_id] = ImageRecord(next_id, fname_a, 100, 100, ts)
        next_id += 1
        if i in duplicates_at:
            fname_b = f"{ts}Mh_b.jpg"
            images[next_id] = ImageRecord(next_id, fname_b, 100, 100, ts)
            next_id += 1

    return MagfiloDataset(images=images, annotations_by_image={i: [] for i in images})


# --------------------------------------------------------------------------
# time_grouped_split
# --------------------------------------------------------------------------


def test_split_sizes_approximately_match_requested_fractions():
    ds = make_dataset(n_unique=100)
    assignment = time_grouped_split(ds, val_fraction=0.15, test_fraction=0.15)

    n_train = len(assignment.image_ids("train"))
    n_val = len(assignment.image_ids("val"))
    n_test = len(assignment.image_ids("test"))

    assert n_train + n_val + n_test == 100
    assert n_val == pytest.approx(15, abs=2)
    assert n_test == pytest.approx(15, abs=2)


def test_split_is_time_ordered_not_interleaved():
    """Every train observation_key must sort earlier than every val key,
    which must sort earlier than every test key."""
    ds = make_dataset(n_unique=50)
    assignment = time_grouped_split(ds, val_fraction=0.2, test_fraction=0.2)

    train_keys = {ds.images[i].observation_key for i in assignment.image_ids("train")}
    val_keys = {ds.images[i].observation_key for i in assignment.image_ids("val")}
    test_keys = {ds.images[i].observation_key for i in assignment.image_ids("test")}

    assert max(train_keys) < min(val_keys)
    assert max(val_keys) < min(test_keys)


def test_duplicate_observations_stay_in_one_split():
    """The central leakage-prevention property: both image ids of a
    duplicate-annotated frame must land in the same split."""
    ds = make_dataset(n_unique=100, duplicates_at={10, 50, 90})
    assignment = time_grouped_split(ds, val_fraction=0.15, test_fraction=0.15)

    # Find the duplicate pairs and check they agree.
    by_key: dict[str, list[int]] = {}
    for img_id, img in ds.images.items():
        by_key.setdefault(img.observation_key, []).append(img_id)

    duplicated = {k: v for k, v in by_key.items() if len(v) > 1}
    assert len(duplicated) == 3  # sanity: the fixture actually has duplicates

    for key, img_ids in duplicated.items():
        splits = {assignment.image_id_to_split[i] for i in img_ids}
        assert len(splits) == 1, f"observation {key} split across {splits}"


def test_rejects_invalid_fractions():
    ds = make_dataset(n_unique=10)
    with pytest.raises(ValueError, match="0, 1"):
        time_grouped_split(ds, val_fraction=0.0, test_fraction=0.15)
    with pytest.raises(ValueError, match="< 1"):
        time_grouped_split(ds, val_fraction=0.6, test_fraction=0.6)


def test_every_image_is_assigned_exactly_one_split():
    ds = make_dataset(n_unique=40, duplicates_at={5, 20})
    assignment = time_grouped_split(ds, val_fraction=0.2, test_fraction=0.2)
    assigned = set(assignment.image_id_to_split.keys())
    assert assigned == set(ds.images.keys())


# --------------------------------------------------------------------------
# assert_no_leakage
# --------------------------------------------------------------------------


def test_assert_no_leakage_passes_on_a_correct_split():
    ds = make_dataset(n_unique=60, duplicates_at={3, 30, 55})
    assignment = time_grouped_split(ds, val_fraction=0.15, test_fraction=0.15)
    assert_no_leakage(ds, assignment)  # must not raise


def test_assert_no_leakage_catches_an_injected_leak():
    ds = make_dataset(n_unique=60, duplicates_at={3, 30, 55})
    assignment = time_grouped_split(ds, val_fraction=0.15, test_fraction=0.15)

    # Sabotage: find a duplicated observation and force its two image ids
    # into different splits, simulating a bug in a future refactor.
    by_key: dict[str, list[int]] = {}
    for img_id, img in ds.images.items():
        by_key.setdefault(img.observation_key, []).append(img_id)
    dup_key, dup_ids = next((k, v) for k, v in by_key.items() if len(v) > 1)

    broken = dict(assignment.image_id_to_split)
    broken[dup_ids[0]] = "train"
    broken[dup_ids[1]] = "test"
    sabotaged = assignment.__class__(
        image_id_to_split=broken,
        observation_key_to_split=assignment.observation_key_to_split,
    )

    with pytest.raises(AssertionError, match="leakage"):
        assert_no_leakage(ds, sabotaged)
