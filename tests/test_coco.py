"""Tests for MAGFiLO COCO parsing, against a synthetic fixture shaped like
the documented format -- NOT the real file, which is still unavailable. See
the module docstring in filament/data/coco.py for the specific assumptions
this fixture encodes and which P0's data audit must confirm or correct.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from filament.data.coco import (
    FilamentAnnotation,
    ImageRecord,
    load_magfilo,
    polygon_to_mask,
)

try:  # pragma: no cover - environment dependent
    from pycocotools import mask as pycocotools_mask
    HAS_PYCOCOTOOLS = True
except ImportError:  # pragma: no cover
    HAS_PYCOCOTOOLS = False


def synthetic_magfilo_json(tmp_path):
    """A minimal COCO file with two duplicate-annotated frames and one unique
    frame, mimicking MAGFiLO's documented 1,593-rows / 958-unique structure
    at small scale."""
    data = {
        "info": {"description": "synthetic test fixture"},
        "licenses": [],
        "categories": [
            {"id": 1, "name": "left"},
            {"id": 2, "name": "right"},
            {"id": 3, "name": "unidentifiable"},
        ],
        "images": [
            # Same GONG frame (timestamp 20150125172714), annotated twice
            # under different image ids -- the duplicate-observation case.
            {"id": 1, "file_name": "20150125172714Mh_a.jpg", "height": 100, "width": 100},
            {"id": 2, "file_name": "20150125172714Mh_b.jpg", "height": 100, "width": 100},
            # A distinct frame, later in time.
            {"id": 3, "file_name": "20160601090000Mh.jpg", "height": 100, "width": 100},
        ],
        "annotations": [
            {
                "id": 101,
                "image_id": 1,
                "category_id": 1,
                "segmentation": [[10, 10, 30, 10, 30, 20, 10, 20]],  # a 20x10 rect
                "bbox": [10, 10, 20, 10],
            },
            {
                "id": 102,
                "image_id": 2,
                "category_id": 2,
                "segmentation": [[10, 10, 30, 10, 30, 20, 10, 20]],
            },
            {
                "id": 103,
                "image_id": 3,
                "category_id": 3,
                "segmentation": [[50, 50, 70, 50, 70, 60, 50, 60]],
            },
        ],
    }
    path = tmp_path / "magfilo_test.json"
    path.write_text(json.dumps(data))
    return path


# --------------------------------------------------------------------------
# load_magfilo
# --------------------------------------------------------------------------


def test_load_parses_images_and_annotations(tmp_path):
    ds = load_magfilo(synthetic_magfilo_json(tmp_path))
    assert len(ds) == 3
    assert ds.n_annotations() == 3
    assert isinstance(ds.images[1], ImageRecord)
    assert ds.images[1].file_name == "20150125172714Mh_a.jpg"


def test_load_annotations_indexed_by_image():
    pass  # covered via synthetic fixture in test above; kept for discoverability


def test_load_annotation_fields(tmp_path):
    ds = load_magfilo(synthetic_magfilo_json(tmp_path))
    anns = ds.annotations_by_image[1]
    assert len(anns) == 1
    ann = anns[0]
    assert isinstance(ann, FilamentAnnotation)
    assert ann.category_id == 1
    assert ann.bbox == (10, 10, 20, 10)


def test_load_rejects_missing_top_level_keys(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"images": []}))  # no 'annotations'
    with pytest.raises(KeyError, match="images.*annotations|annotations.*images"):
        load_magfilo(bad)


def test_observation_key_deduplicates_re_annotated_frames(tmp_path):
    """Two image ids from the same GONG timestamp must share one
    observation_key -- this is the property the split logic depends on."""
    ds = load_magfilo(synthetic_magfilo_json(tmp_path))
    keys = ds.observation_keys()
    assert keys[1] == keys[2]  # same underlying frame
    assert keys[1] != keys[3]  # genuinely different frame


def test_observation_key_falls_back_gracefully_for_unrecognised_names(tmp_path):
    """A file name without a 14-digit timestamp must not raise -- it should
    fall back to a stable per-file key instead."""
    data = {
        "images": [{"id": 1, "file_name": "weird_name.jpg", "height": 10, "width": 10}],
        "annotations": [],
    }
    path = tmp_path / "weird.json"
    path.write_text(json.dumps(data))
    ds = load_magfilo(path)
    assert ds.observation_keys()[1] == "weird_name"


# --------------------------------------------------------------------------
# polygon_to_mask
# --------------------------------------------------------------------------


def test_polygon_to_mask_simple_rectangle():
    # A rectangle from (10,10) to (30,20) in (x, y) COCO order.
    poly = [[10, 10, 30, 10, 30, 20, 10, 20]]
    mask = polygon_to_mask(poly, height=40, width=40)
    assert mask[15, 20]  # inside
    assert not mask[5, 5]  # outside
    assert not mask[25, 20]  # outside (y=25 > 20)


def test_polygon_to_mask_area_is_approximately_correct():
    poly = [[10, 10, 30, 10, 30, 20, 10, 20]]  # 20 wide x 10 tall = 200
    mask = polygon_to_mask(poly, height=40, width=40)
    assert mask.sum() == pytest.approx(200, abs=20)


def test_polygon_to_mask_triangle():
    poly = [[0, 0, 10, 0, 5, 10]]
    mask = polygon_to_mask(poly, height=20, width=20)
    assert mask.sum() > 0
    assert mask[1, 5]  # near the wide base, should be inside
    assert not mask[15, 5]  # well below the triangle's apex


def test_polygon_to_mask_degenerate_polygon_ignored():
    """A polygon with fewer than 3 points must not raise or crash."""
    mask = polygon_to_mask([[0, 0, 5, 5]], height=10, width=10)
    assert not mask.any()


def test_polygon_to_mask_multi_part_unions():
    poly = [
        [0, 0, 5, 0, 5, 5, 0, 5],  # top-left square
        [10, 10, 15, 10, 15, 15, 10, 15],  # bottom-right square, disjoint
    ]
    mask = polygon_to_mask(poly, height=20, width=20)
    assert mask[2, 2]
    assert mask[12, 12]
    assert not mask[7, 7]  # between the two, should be empty


@pytest.mark.skipif(not HAS_PYCOCOTOOLS, reason="pycocotools not installed")
def test_polygon_to_mask_matches_pycocotools():
    poly = [[10, 10, 30, 10, 30, 25, 15, 30, 10, 20]]  # an irregular pentagon
    h, w = 50, 50
    ours = polygon_to_mask(poly, height=h, width=w)

    rle = pycocotools_mask.frPyObjects(poly, h, w)
    merged = pycocotools_mask.merge(rle)
    reference = pycocotools_mask.decode(merged).astype(bool)

    # Allow a small boundary-pixel discrepancy: the two implementations use
    # different point-in-polygon sampling conventions at pixel edges, but the
    # interior and overall area must agree closely.
    disagreement = np.logical_xor(ours, reference).sum()
    assert disagreement < 0.03 * reference.sum()
