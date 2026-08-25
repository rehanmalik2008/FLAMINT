"""Precompute and cache resized images, masks, and training targets.

Motivated directly by the finding in P0's real-data audit
(``filament.data.dataset``'s module docstring): skeletonize + the Euclidean
distance transform at full 2048x2048 resolution cost a few seconds per image,
which is far too slow to run live inside a training loop.

This script does that work once, at the project's actual training resolution
(1024x1024, per the plan), and caches the result as compact .npy files. The
training-time Dataset (``filament.data.dataset_cached.CachedFilamentDataset``)
then just loads arrays -- no JPEG decode, no skeletonize, no distance
transform, in the hot loop.

Storage choices, to keep the cache small: image as uint8 (not float), mask
and spine as packed bool via np.packbits (8x smaller than uint8), offsets as
float16 (displacement components are unit-vector-scale, float16's precision
is more than adequate and halves size vs float32).

Run:  python scripts/precompute_targets.py [--resolution 1024] [--limit N]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from filament.data.coco import load_magfilo, polygon_to_mask  # noqa: E402
from filament.data.targets import build_offset_target, build_spine_target  # noqa: E402
from skimage.morphology import skeletonize  # noqa: E402
from skimage.transform import resize as sk_resize  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_JSON = ROOT / "data" / "MAGFiLO_1.0_Annotations_kaggle2026_train.json"
DEFAULT_IMAGE_DIR = ROOT / "data" / "train_images"
DEFAULT_CACHE_DIR = ROOT / "data" / "cache"


def resize_mask(mask: np.ndarray, size: int) -> np.ndarray:
    """Nearest-neighbour resize for a boolean mask (order=0 preserves the
    {0,1} structure; anything else would introduce fractional edge values
    that then need re-thresholding, an unnecessary extra step here)."""
    return sk_resize(mask, (size, size), order=0, preserve_range=True, anti_aliasing=False).astype(bool)


def process_one(
    image_path: Path, segmentations: list[list[list[float]]], orig_h: int, orig_w: int, size: int
) -> dict[str, np.ndarray]:
    with Image.open(image_path) as img:
        image = np.asarray(img.convert("L").resize((size, size), Image.BILINEAR), dtype=np.uint8)

    mask_full = np.zeros((orig_h, orig_w), dtype=bool)
    for seg in segmentations:
        mask_full |= polygon_to_mask(seg, orig_h, orig_w)
    mask = resize_mask(mask_full, size)

    # Skeleton and offsets computed at the CACHE resolution, not full res --
    # this is the actual speedup: 1024x1024 is 1/4 the pixels of 2048x2048.
    spine = build_spine_target(mask, dilation_radius=1)
    undilated_skeleton = skeletonize(mask)
    offsets = build_offset_target(mask, undilated_skeleton)

    return {
        "image": image,  # uint8, (size, size)
        "mask": np.packbits(mask),  # packed bool
        "mask_shape": np.array([size, size], dtype=np.int32),
        "spine": np.packbits(spine.astype(bool)),
        "offsets": offsets.astype(np.float16),  # (2, size, size)
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGE_DIR)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument("--limit", type=int, default=None, help="process only N images (debug)")
    args = parser.parse_args()

    ds = load_magfilo(args.json)
    args.cache_dir.mkdir(parents=True, exist_ok=True)

    available = {p.name for p in args.image_dir.iterdir()}
    image_ids = [iid for iid, rec in ds.images.items() if rec.file_name in available]
    if args.limit:
        image_ids = image_ids[: args.limit]

    print(f"processing {len(image_ids)} images at {args.resolution}x{args.resolution} ...")
    t0 = time.time()
    for i, image_id in enumerate(image_ids):
        record = ds.images[image_id]
        anns = ds.annotations_by_image.get(image_id, [])
        segmentations = [a.segmentation for a in anns]

        out_path = args.cache_dir / f"{image_id.replace('/', '_')}.npz"
        if out_path.exists():
            continue

        data = process_one(
            args.image_dir / record.file_name,
            segmentations,
            record.height,
            record.width,
            args.resolution,
        )
        np.savez_compressed(out_path, **data)

        if (i + 1) % 50 == 0 or (i + 1) == len(image_ids):
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            print(f"  {i + 1}/{len(image_ids)}  ({rate:.1f} img/s, {elapsed:.0f}s elapsed)")

    print(f"done in {time.time() - t0:.0f}s -> {args.cache_dir}")


if __name__ == "__main__":
    main()
