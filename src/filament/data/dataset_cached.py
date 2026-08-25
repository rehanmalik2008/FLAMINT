"""The fast-path Dataset: reads precomputed targets instead of computing
them live.

Pairs with ``scripts/precompute_targets.py``, which writes one .npz per image
containing the resized image, mask, spine, and offset targets -- see that
script's docstring for why (skeletonize + distance transform are too slow to
run inside a training loop, as measured against real data in P0's audit).
``FilamentDataset`` (``filament.data.dataset``) remains the correct reference
implementation and is what the precompute script's per-image logic mirrors;
this module trades that generality for training-loop speed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from filament.data.dataset import FilamentSample

__all__ = ["CachedFilamentDataset"]


class CachedFilamentDataset(Dataset):
    """Loads precomputed .npz files produced by scripts/precompute_targets.py.

    Parameters
    ----------
    cache_dir:
        Directory of ``<image_id>.npz`` files.
    image_ids:
        Which cached images this instance should serve (a split's worth).
        Ids with slashes are sanitised the same way the precompute script
        sanitises them for the file name (``/`` -> ``_``); real MAGFiLO ids
        do not contain slashes, but this keeps the two sides consistent if
        that ever changes.
    """

    def __init__(self, cache_dir: str | Path, image_ids: list[str]):
        self.cache_dir = Path(cache_dir)
        self.image_ids = list(image_ids)

    def __len__(self) -> int:
        return len(self.image_ids)

    def _path_for(self, image_id: str) -> Path:
        return self.cache_dir / f"{image_id.replace('/', '_')}.npz"

    def __getitem__(self, idx: int) -> FilamentSample:
        image_id = self.image_ids[idx]
        path = self._path_for(image_id)
        if not path.exists():
            raise FileNotFoundError(
                f"no cached target file for image_id={image_id!r} at {path} -- "
                "run scripts/precompute_targets.py first"
            )

        with np.load(path) as data:
            image = data["image"]
            size_h, size_w = int(data["mask_shape"][0]), int(data["mask_shape"][1])
            mask = np.unpackbits(data["mask"])[: size_h * size_w].reshape(size_h, size_w)
            spine = np.unpackbits(data["spine"])[: size_h * size_w].reshape(size_h, size_w)
            offsets = data["offsets"].astype(np.float32)

        image_t = torch.from_numpy(image.astype(np.float32) / 255.0)[None, :, :]
        mask_t = torch.from_numpy(mask.astype(np.float32))[None, :, :]
        spine_t = torch.from_numpy(spine.astype(np.float32))[None, :, :]
        offsets_t = torch.from_numpy(offsets)

        return FilamentSample(
            image=image_t, mask=mask_t, spine=spine_t, offsets=offsets_t, image_id=image_id
        )
