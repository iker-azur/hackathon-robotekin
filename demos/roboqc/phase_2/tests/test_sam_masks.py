"""SAM export tests with handmade mask fixtures; never load or run a model."""

import json

import cv2
import numpy as np
import pytest

from roboqc.industrial.sam_masks import save_candidates


def test_preserves_raw_holes_and_overlaps_and_reports_filtering(tmp_path):
    rgb = np.full((40,60,3),200,np.uint8)
    ring = np.zeros((40,60),bool); ring[5:30,5:30] = True; ring[10:20,10:20] = False
    nested = np.zeros_like(ring); nested[5:9,5:9] = True
    background = np.ones_like(ring)
    tiny = np.zeros_like(ring); tiny[0,0] = True
    annotations = [{"segmentation":m,"predicted_iou":.9,"stability_score":.95} for m in [ring,nested,background,tiny]]
    out = tmp_path / "export"
    report,overlay = save_candidates(rgb,annotations,out,minimum_area=4,maximum_fraction=.85)
    assert report["raw_masks"] == 4 and report["kept_candidates"] == 2
    assert report["accuracy"] == "not_measured"
    rows = report["candidates"]
    assert rows[0]["bbox_xywh"] == [5,5,25,25]
    assert rows[2]["filter_reason"] == "above_maximum_fraction"
    assert rows[3]["filter_reason"] == "below_minimum_area"
    assert rows[0]["category"] == rows[0]["quality"] == "unknown"
    assert not rows[0]["grasp_candidates"][0]["validated"]
    mask = cv2.imread(str(out / rows[0]["mask_file"]),0)
    assert mask[15,15] == 0 and mask[6,6] == 255
    with np.load(out / "raw_masks.npz") as archive:
        assert archive["masks"].shape == (4,40,60)
        np.testing.assert_array_equal(archive["masks"][0],ring)
        assert np.logical_and(archive["masks"][0],archive["masks"][1]).sum() == 16
    assert overlay.shape == rgb.shape and overlay.dtype == np.uint8
    assert json.loads((out / "candidates.json").read_text())["schema_version"] == "sam-candidates-1.0"
    with pytest.raises(FileExistsError): save_candidates(rgb,annotations,out)


def test_empty_predictions_are_reported_without_inventing_objects(tmp_path):
    image = np.full((20,30,3),80,np.uint8)
    report,overlay = save_candidates(image,[],tmp_path / "empty")
    assert report["raw_masks"] == report["kept_candidates"] == 0
    np.testing.assert_array_equal(overlay,image)
    with np.load(tmp_path / "empty/raw_masks.npz") as archive:
        assert archive["masks"].shape == (0,20,30)


def test_invalid_masks_and_parameters_fail_before_writing(tmp_path):
    image = np.zeros((20,30,3),np.uint8)
    out = tmp_path / "bad"
    with pytest.raises(ValueError,match="shape"):
        save_candidates(image,[{"segmentation":np.ones((10,10)),"predicted_iou":1}],out)
    with pytest.raises(ValueError,match="minimum_area"):
        save_candidates(image,[],out,maximum_fraction=2)
    assert not out.exists()
