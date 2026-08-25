"""Leak-free, time-aware train/val/test splitting for MAGFiLO.

Two properties of the dataset make a naive random split unsafe:

1. **Duplicate observations.** The paper reports 1,593 annotation rows over
   only 958 unique observations -- some frames were independently
   re-annotated by more than one reviewer group. A random split at the
   *annotation* level can put two annotations of the *same* frame on
   opposite sides of train/val, which is direct leakage: the model would be
   validated on an image (or a near-duplicate of one) it effectively trained
   on.
2. **Temporal structure.** GONG runs continuous 60-second-cadence
   observations across 2011-2022, spanning solar cycle 24's max, its
   subsequent minimum, and the rise of cycle 25. Filament abundance and
   morphology vary substantially with the solar cycle. A random split
   distributes cycle phases evenly across train/val/test and will therefore
   *overstate* generalisation relative to time-ordered real-world deployment
   (and relative to how the competition's own held-out test set was almost
   certainly constructed).

This module groups by ``observation_key`` (see ``filament.data.coco``) so
that all annotations of one frame land in the same split, and splits by time
so validation reflects a genuinely later period than training.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from filament.data.coco import MagfiloDataset

__all__ = ["SplitAssignment", "time_grouped_split", "assert_no_leakage"]


@dataclass(frozen=True)
class SplitAssignment:
    """image_id -> split name ('train' / 'val' / 'test'), plus the audit trail."""

    image_id_to_split: dict[str, str]
    observation_key_to_split: dict[str, str]

    def image_ids(self, split: str) -> list[str]:
        return sorted(
            img_id
            for img_id, s in self.image_id_to_split.items()
            if s == split
        )


def time_grouped_split(
    dataset: MagfiloDataset,
    *,
    val_fraction: float = 0.15,
    test_fraction: float = 0.15,
) -> SplitAssignment:
    """Split by observation_key, ordered by key (a proxy for acquisition time).

    ``observation_key`` (see ``filament.data.coco._observation_key``) prefers
    the real file's ``date_captured`` field, formatted so lexicographic and
    chronological order coincide; confirmed against the actual competition
    JSON during P0's data audit, including the duplicate-observation case
    (three image ids sharing one timestamp for one physical frame). If
    ``date_captured`` is ever missing, it falls back to a filename-timestamp
    regex and finally the bare file stem -- in that degraded case ordering is
    no longer guaranteed chronological, but the leakage guarantee still holds,
    since it depends only on grouping, not on correct time order.

    The split boundary is chosen so that **train is strictly earlier than
    val, which is strictly earlier than test** -- not shuffled within a time
    window -- so that validation performance is not inflated by information
    "from the future" relative to what a deployed model would have had.

    Parameters
    ----------
    val_fraction, test_fraction:
        Fraction of unique observation_keys assigned to each split. Applied
        to the count of unique keys, not annotation rows, so a
        heavily-re-annotated frame does not get overweighted in the split
        sizing.
    """
    if not (0 < val_fraction < 1) or not (0 < test_fraction < 1):
        raise ValueError("val_fraction and test_fraction must be in (0, 1)")
    if val_fraction + test_fraction >= 1:
        raise ValueError("val_fraction + test_fraction must be < 1")

    obs_to_images: dict[str, list[str]] = {}
    for img_id, img in dataset.images.items():
        obs_to_images.setdefault(img.observation_key, []).append(img_id)

    ordered_keys = sorted(obs_to_images.keys())
    n = len(ordered_keys)
    n_test = max(int(round(n * test_fraction)), 1) if n > 0 else 0
    n_val = max(int(round(n * val_fraction)), 1) if n > 0 else 0
    n_train = n - n_val - n_test
    if n_train <= 0:
        raise ValueError(
            f"val_fraction + test_fraction leave no observations for "
            f"training ({n} unique observations, {n_val} val, {n_test} test)"
        )

    train_keys = ordered_keys[:n_train]
    val_keys = ordered_keys[n_train : n_train + n_val]
    test_keys = ordered_keys[n_train + n_val :]

    key_to_split: dict[str, str] = {}
    key_to_split.update({k: "train" for k in train_keys})
    key_to_split.update({k: "val" for k in val_keys})
    key_to_split.update({k: "test" for k in test_keys})

    image_to_split: dict[str, str] = {}
    for key, img_ids in obs_to_images.items():
        split = key_to_split[key]
        for img_id in img_ids:
            image_to_split[img_id] = split

    return SplitAssignment(
        image_id_to_split=image_to_split, observation_key_to_split=key_to_split
    )


def assert_no_leakage(dataset: MagfiloDataset, assignment: SplitAssignment) -> None:
    """Raise AssertionError if any observation_key spans more than one split.

    Intended to run as an explicit, named check in the training entry point
    and in CI -- not just implicitly relied upon -- so a future change to the
    splitting logic that reintroduces leakage fails loudly rather than
    silently inflating validation scores.
    """
    keys_by_split: dict[str, set[str]] = {"train": set(), "val": set(), "test": set()}
    for img_id, img in dataset.images.items():
        split = assignment.image_id_to_split[img_id]
        keys_by_split[split].add(img.observation_key)

    train_val = keys_by_split["train"] & keys_by_split["val"]
    train_test = keys_by_split["train"] & keys_by_split["test"]
    val_test = keys_by_split["val"] & keys_by_split["test"]

    leaks = []
    if train_val:
        leaks.append(f"train/val: {sorted(train_val)[:5]}")
    if train_test:
        leaks.append(f"train/test: {sorted(train_test)[:5]}")
    if val_test:
        leaks.append(f"val/test: {sorted(val_test)[:5]}")

    if leaks:
        raise AssertionError(
            "observation_key leakage across splits: " + "; ".join(leaks)
        )
