"""YOLOE contract tests with fake weights/predictions; no model downloads."""

import json
import sys
import types

import numpy as np
import pytest

from roboqc.industrial.yoloe import DEFAULT_PROMPTS, compare_instances, find_images, inspect_images, prepare_bundle
from roboqc.industrial.vision_inference import CLASSES, Segmenter


def test_export_class_order_prompt_mapping_and_no_training(tmp_path,monkeypatch):
    calls = []
    class Model:
        names = {}
        def __init__(self,path,task): calls.append(("load",path))
        def get_text_pe(self,texts): return "embeddings"
        def set_classes(self,texts,embeddings):
            assert embeddings == "embeddings"
            self.names = dict(enumerate(texts))
            calls.append(("prompts",texts))
        def save(self,path): Path(path).write_bytes(b"fake")
        def train(self,*args,**kwargs): raise AssertionError("Must never train")
    from pathlib import Path
    monkeypatch.setitem(sys.modules,"ultralytics",types.SimpleNamespace(YOLOE=Model))
    monkeypatch.setattr("roboqc.industrial.yoloe.Segmenter",lambda *a,**kw:types.SimpleNamespace(predict=lambda image:[]))
    monkeypatch.setattr("roboqc.industrial.yoloe.cv2.imread",lambda p:np.zeros((20,20,3),np.uint8))
    prompts = dict(reversed(list(DEFAULT_PROMPTS.items())))
    out = tmp_path / "bundle"
    manifest = prepare_bundle(out,["image.png"],prompts=prompts,export_onnx=False)
    assert manifest["classes"] == CLASSES
    assert ("prompts",[DEFAULT_PROMPTS[c] for c in CLASSES]) in calls
    assert manifest["training_performed"] is False and manifest["quality_enabled"] is False
    assert manifest["export_verified"] is True
    assert calls[-1] == ("load",str(out / "segmentation.pt"))
    with pytest.raises(FileExistsError):
        prepare_bundle(out,["image.png"],export_onnx=False)


def test_invalid_prompts_and_missing_photos_fail_before_downloading(tmp_path):
    with pytest.raises(ValueError,match="nonempty"):
        prepare_bundle(tmp_path / "a",["photo.png"],prompts={"nut":"nut"})
    with pytest.raises(ValueError,match="distinguish"):
        prepare_bundle(tmp_path / "a",["photo.png"],prompts={c:"part" for c in CLASSES})
    with pytest.raises(ValueError,match="photo"):
        prepare_bundle(tmp_path / "a",[])
    with pytest.raises(ValueError,match="ordinary"):
        find_images(tmp_path)
    assert not (tmp_path / "a").exists()


def test_export_matching_checks_masks_and_class_not_output_order():
    mask = np.eye(10,dtype=np.uint8)
    a = {"type":"nut","mask":mask}
    b = {"type":"bolt","mask":np.fliplr(mask)}
    assert compare_instances([a,b],[b,a]) == [1.0,1.0]
    with pytest.raises(AssertionError): compare_instances([a],[b])
    with pytest.raises(AssertionError): compare_instances([a],[])


def test_export_mismatch_saves_diagnostics_and_never_marks_bundle_verified(tmp_path,monkeypatch):
    from pathlib import Path
    class Model:
        def __init__(self,path,task): self.path = Path(path)
        def get_text_pe(self,texts): return "embeddings"
        def set_classes(self,texts,embeddings): self.names = dict(enumerate(texts))
        def save(self,path): Path(path).write_bytes(b"fake weights")
        def export(self,**kwargs):
            path = self.path.with_suffix(".onnx")
            path.write_bytes(b"fake export")
            return str(path)
    obj = {"type":"bolt","confidence":.9,"bbox":[1,1,8,8],"mask":np.ones((10,10),np.uint8)}
    def segmenter(bundle,prefer_onnx=True,**kwargs):
        return types.SimpleNamespace(predict=lambda frame:[] if prefer_onnx else [obj])
    monkeypatch.setitem(sys.modules,"ultralytics",types.SimpleNamespace(YOLOE=Model))
    monkeypatch.setattr("roboqc.industrial.yoloe.Segmenter",segmenter)
    monkeypatch.setattr("roboqc.industrial.yoloe.cv2.imread",lambda p:np.zeros((10,10,3),np.uint8))
    out = tmp_path / "bundle"
    with pytest.raises(AssertionError,match="PyTorch=1, ONNX=0"):
        prepare_bundle(out,["photo.png"])
    report = json.loads((out / "export_equivalence.json").read_text())
    assert report["passed"] is False and report["inference"]["rect"] is False
    assert report["images"][0]["pytorch"][0]["type"] == "bolt"
    assert report["images"][0]["onnx_predicted_instances"] == 0
    assert json.loads((out / "vision_manifest.json").read_text())["export_verified"] is False


@pytest.mark.parametrize("prefer_onnx",[False,True])
def test_pretrained_segmenter_loads_baked_prompts_and_returns_canonical_ids(tmp_path,monkeypatch,prefer_onnx):
    pytest.importorskip("torch")
    mask = np.zeros((1,60,80),np.float32); mask[:,10:40,10:50] = 1
    class Array:
        def __init__(self,value): self.value = np.array(value)
        def cpu(self): return self
        def numpy(self): return self.value
    prediction = types.SimpleNamespace(masks=types.SimpleNamespace(data=Array(mask)),boxes=types.SimpleNamespace(
        xyxy=Array([[10,10,50,40]]),cls=Array([1]),conf=Array([.91])))
    used = []
    class Model:
        def __init__(self,path,task): used.append((path,task))
        def predict(self,frame,**kwargs):
            assert kwargs["device"] == "cpu"
            # Static ONNX and PT must receive identical padding and NMS options.
            assert kwargs["rect"] is False
            assert kwargs["agnostic_nms"] is True
            assert kwargs["iou"] == .7 and kwargs["max_det"] == 300
            assert kwargs["retina_masks"] is False and kwargs["augment"] is False
            return [prediction]
        def set_classes(self,*args): raise AssertionError("No text encoder on local load")
    monkeypatch.setitem(sys.modules,"ultralytics",types.SimpleNamespace(YOLO=Model,YOLOE=Model))
    monkeypatch.setitem(sys.modules,"ultralytics.utils",types.ModuleType("ultralytics.utils"))
    monkeypatch.setitem(sys.modules,"ultralytics.utils.ops",types.SimpleNamespace(scale_masks=lambda tensor,size:tensor))
    (tmp_path / "vision_manifest.json").write_text(json.dumps({"schema_version":"industrial-vision-1.0","classes":CLASSES,"model_family":"yoloe","quality_enabled":False}))
    (tmp_path / "segmentation.pt").write_bytes(b"fake")
    (tmp_path / "segmentation.onnx").write_bytes(b"fake")
    segmenter = Segmenter(tmp_path,prefer_onnx=prefer_onnx)
    frame = np.zeros((60,80,3),np.uint8); frame[10:40,10:50] = [0,0,255]
    obj = segmenter.predict(frame)[0]
    assert obj["type"] == "nut" and obj["color"] == "red"
    assert obj["quality"] == "unknown" and obj["orientation_deg"] is None
    assert obj["mask"].shape == (60,80)
    prediction.boxes.cls = Array([8])
    with pytest.raises(ValueError,match="class ID"):
        segmenter.predict(frame)


def test_unlabelled_report_never_claims_accuracy(tmp_path,monkeypatch):
    pytest.importorskip("torch")
    import cv2
    image = tmp_path / "photo.png"
    cv2.imwrite(str(image),np.zeros((20,20,3),np.uint8))
    monkeypatch.setattr("roboqc.industrial.yoloe.Segmenter",lambda *a,**kw:types.SimpleNamespace(predict=lambda frame:[]))
    report,rows = inspect_images("unused",[image],tmp_path / "inspection")
    assert report["accuracy"]["status"] == "not_measured"
    assert report["quality"]["status"] == "unknown"
    assert rows[0]["objects"] == [] and report["images"] == 1


def test_demo_photos_download_credited_and_reuse_only_verified_files(tmp_path,monkeypatch):
    import hashlib
    import io
    import cv2
    import urllib.request
    import roboqc.industrial.yoloe as module
    monkeypatch.setattr(module.time,"sleep",lambda seconds:None)
    ok,encoded = cv2.imencode(".jpg",np.zeros((20,30,3),np.uint8))
    assert ok
    photo = encoded.tobytes()
    examples = [dict(item,sha1=hashlib.sha1(photo).hexdigest()) for item in module.DEMO_IMAGES]
    monkeypatch.setattr(module,"DEMO_IMAGES",examples)
    calls = []
    def download(request,timeout):
        calls.append(request.full_url)
        assert timeout == 45
        return io.BytesIO(photo)
    monkeypatch.setattr(urllib.request,"urlopen",download)
    out = module.download_demo_images(tmp_path / "photos")
    assert len(module.find_images(out)) == len(examples) and len(calls) == len(examples)
    credits = (out / "ATTRIBUTION.txt").read_text(encoding="utf-8")
    assert "Sbrennan83" in credits and "CC BY-SA 4.0" in credits and "Fructibus" in credits
    assert "https://creativecommons.org/" in credits and "Changes: none" in credits
    module.download_demo_images(out)
    assert len(calls) == len(examples)  # Rerunning never downloads or replaces existing images.
    changed = out / examples[0]["filename"]
    changed.write_bytes(b"user image")
    with pytest.raises(ValueError,match="checksum"):
        module.download_demo_images(out)
    assert changed.read_bytes() == b"user image"


def test_demo_catalog_contains_credited_formed_bracket_photos():
    import re
    from roboqc.industrial.yoloe import DEMO_IMAGES
    by_name = {i["filename"]:i for i in DEMO_IMAGES}
    assert len(by_name) == len(DEMO_IMAGES) == 5
    assert {"four_gray_brackets.jpg","amplifier_bracket.jpg","channel_connector.jpg"} <= by_name.keys()
    for item in DEMO_IMAGES:
        assert re.fullmatch("[0-9a-f]{40}",item["sha1"])
        assert item["author"] and item["title"] and item["source"].startswith("https://commons.wikimedia.org/wiki/File:")
        assert item["license_url"].startswith("https://creativecommons.org/")
    assert "Stereo" in by_name["amplifier_bracket.jpg"]["notes"]
    assert "Assembled" in by_name["channel_connector.jpg"]["notes"]


def test_demo_photos_reject_wrong_remote_content_and_explain_network_failure(tmp_path,monkeypatch):
    import io
    import urllib.request
    from roboqc.industrial.yoloe import download_demo_images
    monkeypatch.setattr("roboqc.industrial.yoloe.time.sleep",lambda seconds:None)
    monkeypatch.setattr(urllib.request,"urlopen",lambda *a,**kw:io.BytesIO(b"not the requested photo"))
    with pytest.raises(ValueError,match="checksum"):
        download_demo_images(tmp_path / "wrong")
    assert not list((tmp_path / "wrong").glob("*.jpg"))
    def offline(*args,**kwargs): raise OSError("offline")
    monkeypatch.setattr(urllib.request,"urlopen",offline)
    with pytest.raises(RuntimeError,match="USE_DEMO_IMAGES=False"):
        download_demo_images(tmp_path / "offline")


@pytest.mark.parametrize("status",[429,503])
@pytest.mark.parametrize("retry_after",["7","http-date"])
def test_demo_photos_retry_429_and_respect_retry_after(tmp_path,monkeypatch,status,retry_after):
    import hashlib
    import io
    import cv2
    import urllib.request
    from datetime import datetime, timedelta, timezone
    from email.utils import format_datetime
    from urllib.error import HTTPError
    import roboqc.industrial.yoloe as module
    _,encoded = cv2.imencode(".jpg",np.zeros((20,20,3),np.uint8))
    photo = encoded.tobytes()
    monkeypatch.setattr(module,"DEMO_IMAGES",[dict(module.DEMO_IMAGES[0],sha1=hashlib.sha1(photo).hexdigest())])
    sleeps,calls = [],[]
    monkeypatch.setattr(module.time,"sleep",sleeps.append)
    header = format_datetime(datetime.now(timezone.utc)+timedelta(seconds=30),usegmt=True) if retry_after == "http-date" else retry_after
    def download(request,timeout):
        calls.append(request.full_url)
        if len(calls) == 1:
            raise HTTPError(request.full_url,status,"Transient server error",{"Retry-After":header},None)
        return io.BytesIO(photo)
    monkeypatch.setattr(urllib.request,"urlopen",download)
    out = module.download_demo_images(tmp_path / "photos")
    assert len(calls) == 2 and len(sleeps) == 1
    assert 25 <= sleeps[0] <= 30 if retry_after == "http-date" else sleeps == [7]
    report = json.loads((out / "download_report.json").read_text())
    assert report["status"] == "complete" and len(report["available"]) == 1


def test_demo_photos_keep_cached_images_and_stop_requests_when_rate_limited(tmp_path,monkeypatch):
    import hashlib
    import cv2
    import urllib.request
    from urllib.error import HTTPError
    import roboqc.industrial.yoloe as module
    _,encoded = cv2.imencode(".jpg",np.zeros((20,20,3),np.uint8))
    photo = encoded.tobytes()
    examples = [dict(i,sha1=hashlib.sha1(photo).hexdigest()) for i in module.DEMO_IMAGES[:3]]
    monkeypatch.setattr(module,"DEMO_IMAGES",examples)
    out = tmp_path / "photos"; out.mkdir()
    (out / examples[0]["filename"]).write_bytes(photo)
    calls,sleeps = [],[]
    monkeypatch.setattr(module.time,"sleep",sleeps.append)
    def limited(request,timeout):
        calls.append(request.full_url)
        raise HTTPError(request.full_url,429,"Too many requests",{},None)
    monkeypatch.setattr(urllib.request,"urlopen",limited)
    assert module.download_demo_images(out) == out
    assert calls == [examples[1]["url"]]*3 and sleeps == [5,10]
    report = json.loads((out / "download_report.json").read_text())
    assert report["status"] == "partial" and len(report["available"]) == 1 and len(report["skipped"]) == 2
    assert "429" in report["skipped"][0]["reason"]
    assert len(module.find_images(out)) == 1


def test_demo_photos_do_not_retry_before_long_retry_after(tmp_path,monkeypatch):
    import urllib.request
    from urllib.error import HTTPError
    import roboqc.industrial.yoloe as module
    calls,sleeps = [],[]
    monkeypatch.setattr(module.time,"sleep",sleeps.append)
    def limited(request,timeout):
        calls.append(request.full_url)
        raise HTTPError(request.full_url,429,"Too many requests",{"Retry-After":"3600"},None)
    monkeypatch.setattr(urllib.request,"urlopen",limited)
    out = tmp_path / "photos"
    with pytest.raises(RuntimeError,match="USE_DEMO_IMAGES=False"):
        module.download_demo_images(out)
    assert len(calls) == 1 and not sleeps
    report = json.loads((out / "download_report.json").read_text())
    assert report["status"] == "unavailable" and len(report["skipped"]) == len(module.DEMO_IMAGES)
