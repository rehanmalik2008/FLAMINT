"""The torch Dataset that ties image loading, GT masks, and training targets
together into what the training loop actually iterates over.

Pipeline per item: load a JPEG -> rasterise all of that image's filament
polygons into one union mask (for the semantic mask head) -> derive spine and
offset targets from that mask (``filament.data.targets``) -> return tensors.

Deliberately does NOT attempt per-instance offset targets here (i.e. does not
try to make each pixel point toward *its own* filament's spine when several
filaments share a frame) -- see ``FilamentDataset``'s docstring for why, and
``tests/test_dataset.py`` for the multi-filament case this affects.

PERFORMANCE NOTE (measured against real data in P0's audit): skeletonize and
the Euclidean distance transform at full 2048x2048 resolution take on the
order of a few seconds per image (7 real-image tests covering ~20 samples in
``tests/test_dataset_real_data.py`` took 137s). That is much too slow to run
live inside a training loop at any reasonable batch size. Before P1 training
starts, spine/offset targets should be precomputed once per image (e.g. to a
cached .npy alongside each JPEG, or generated at the training resolution --
1024x1024 per the project plan -- rather than at full 2048x2048, which would
also directly cut this cost roughly 4x). This module is correct as the
reference implementation; it is not yet the fast path.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from filament.data.coco import MagfiloDataset, polygon_to_mask
from filament.data.targets import build_offset_target, build_spine_target
from filament.geometry.disk import DiskGeometry, fit_disk

__all__ = ["FilamentSample", "FilamentDataset"]


@dataclass(frozen=True)
class FilamentSample:
    """One training item, as tensors ready for the model and loss.

    image   : (C, H, W) float32, normalised to roughly [0, 1]
    mask    : (1, H, W) float32 {0, 1} -- union of all filaments in the frame
    spine   : (1, H, W) float32 {0, 1}
    offsets : (2, H, W) float32
    image_id: the source image's id, for debugging/inspection
    """

    image: torch.Tensor
    mask: torch.Tensor
    spine: torch.Tensor
    offsets: torch.Tensor
    image_id: str


class FilamentDataset(Dataset):
    """A MagfiloDataset plus a directory of JPEGs, as a torch Dataset.

    Parameters
    ----------
    dataset:
        Parsed annotations (``filament.data.coco.load_magfilo``).
    image_dir:
        Directory containing the JPEGs named by ``ImageRecord.file_name``.
    image_ids:
        Which images (by id) this instance should serve -- typically one
        split's worth from ``filament.data.splits.time_grouped_split``.
    include_geometry_channels:
        If True, appends two extra input channels: normalised disk radius
        (r/R_sun) and heliographic longitude/90, from
        ``filament.geometry.disk`` (Edge 3). Off by default because disk
        fitting has an extra failure mode (a mask/threshold that doesn't
        resolve to a sane circle) that intensity-only training does not need
        to depend on while the geometry module's real-image behaviour is
        still unverified (see disk.py's module docstring).
    dilation_radius:
        Passed through to ``build_spine_target``; see that function.

    Notes on the offset target and multiple filaments per frame
    -------------------------------------------------------------
    A frame can contain several filaments (MAGFiLO: up to 26 in one
    observation, per the dataset paper). This dataset builds the offset
    target from the **union** mask and the **union** skeleton -- so a
    foreground pixel belonging to filament A can end up pointing toward
    filament B's spine if B's nearest skeleton point happens to be closer in
    pure Euclidean distance (e.g. two filaments passing near each other).

    This is a known, deliberate simplification for the first working
    pipeline, not an oversight: computing a true per-instance offset field at
    dataset-build time would require resolving which polygon each pixel
    belongs to when they are close/touching, which is exactly the ambiguity
    ``filament.postproc.decompose`` exists to resolve at *inference* time from
    a *predicted* spine map -- doing it perfectly in the *target* generator
    would be circular. In practice most filaments are well-separated (this is
    exactly what makes the watershed step at inference time work), so the
    mislabelled-pixel fraction is expected to be small; ``tests/test_dataset.
    py`` measures it directly on a synthetic two-filament case so the
    magnitude is known rather than assumed, and this is flagged as a
    candidate refinement if P1's real training shows the offset head
    underperforming on multi-filament frames specifically.
    """

    def __init__(
        self,
        dataset: MagfiloDataset,
        image_dir: str | Path,
        image_ids: list[str],
        *,
        include_geometry_channels: bool = False,
        dilation_radius: int = 1,
    ):
        self.dataset = dataset
        self.image_dir = Path(image_dir)
        self.image_ids = list(image_ids)
        self.include_geometry_channels = include_geometry_channels
        self.dilation_radius = dilation_radius

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, idx: int) -> FilamentSample:
        image_id = self.image_ids[idx]
        record = self.dataset.images[image_id]

        image = self._load_image(record.file_name)
        mask = self._build_union_mask(image_id, record.height, record.width)

        spine = build_spine_target(mask, dilation_radius=self.dilation_radius)
        offsets = build_offset_target(mask, self._undilated_skeleton(mask))

        if self.include_geometry_channels:
            image = self._append_geometry_channels(image)

        return FilamentSample(
            image=torch.from_numpy(image),
            mask=torch.from_numpy(mask.astype(np.float32))[None, :, :],
            spine=torch.from_numpy(spine)[None, :, :],
            offsets=torch.from_numpy(offsets),
            image_id=image_id,
        )

    def _load_image(self, file_name: str) -> np.ndarray:
        """Load a grayscale JPEG as a (1, H, W) float32 array in [0, 1]."""
        path = self.image_dir / file_name
        with Image.open(path) as img:
            arr = np.asarray(img.convert("L"), dtype=np.float32) / 255.0
        return arr[np.newaxis, :, :]

    def _build_union_mask(self, image_id: str, height: int, width: int) -> np.ndarray:
        anns = self.dataset.annotations_by_image.get(image_id, [])
        mask = np.zeros((height, width), dtype=bool)
        for ann in anns:
            mask |= polygon_to_mask(ann.segmentation, height, width)
        return mask

    @staticmethod
    def _undilated_skeleton(mask: np.ndarray) -> np.ndarray:
        from skimage.morphology import skeletonize

        return skeletonize(mask)

    def _append_geometry_channels(self, image: np.ndarray) -> np.ndarray:
        """Add r/R_sun and longitude/90 channels, from a coarse disk fit on
        the intensity image itself (bright disk vs. dark background/off-disk)."""
        intensity = image[0]
        # A generous percentile threshold rather than a fixed value, so it
        # adapts to per-frame exposure variation instead of assuming a
        # specific normalised brightness level.
        is_bright = intensity > np.percentile(intensity, 60)
        try:
            geometry: DiskGeometry = fit_disk(is_bright)
        except ValueError:
            # Degenerate frame (e.g. mostly uniform); fall back to a
            # frame-centred, half-frame-radius disk rather than raising and
            # losing the sample entirely.
            h, w = intensity.shape
            geometry = DiskGeometry(h / 2, w / 2, min(h, w) / 2, (h, w))

        rows, cols = np.indices(intensity.shape)
        r_norm = geometry.normalized_radius(rows, cols).astype(np.float32)
        lon = (geometry.longitude(rows, cols) / 90.0).astype(np.float32)
        return np.concatenate([image, r_norm[None], lon[None]], axis=0)
