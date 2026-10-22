"""Comparison contracts on fixture arrays/fake APIs only. No weights or model runs."""

from contextlib import nullcontext
import json
import sys
import types

import numpy as np
import pytest

from roboqc.industrial.pretrained_compare import crop_image, restore_masks, select_boxes, mask_metrics, PretrainedMethod
from roboqc.industrial.sam_masks import save_candidates
from roboqc.industrial.pretrained_compare import _sam_model


@pytest.mark.parametrize("method",["grounding_dino_sam2","owlv2_sam2","florence2_sam2"])
@pytest.mark.parametrize("failure",[None,"once","always","other"])
def test_detector_safetensors_recovery_preserves_cache_and_is_bounded(tmp_path,monkeypatch,method,failure):
    from roboqc.industrial import pretrained_compare
    class SafetensorError(Exception): pass
    monkeypatch.setitem(sys.modules,"safetensors",types.SimpleNamespace(SafetensorError=SafetensorError))
    monkeypatch.setattr(pretrained_compare,"_sam_model",lambda config,device:(object(),{}))
    monkeypatch.setitem(sys.modules,"sam2",types.ModuleType("sam2"))
    monkeypatch.setitem(sys.modules,"sam2.sam2_image_predictor",types.SimpleNamespace(SAM2ImagePredictor=lambda model:object()))
    initial = tmp_path / "local_models"; initial.mkdir()
    original = initial / "existing.safetensors"; original.write_bytes(b"preserve this file")
    config = {"cache":str(initial),"revisions":{pretrained_compare.MODEL_IDS[method]:"pinned_revision"}}
    processors,models = [],[]
    class Processor:
        @classmethod
        def from_pretrained(cls,model_id,**kwargs):
            processors.append(dict(kwargs)); return object()
    class Model:
        config = types.SimpleNamespace(_commit_hash="resolved_revision")
        @classmethod
        def from_pretrained(cls,model_id,**kwargs):
            assert model_id == pretrained_compare.MODEL_IDS[method]
            assert kwargs["revision"] == "pinned_revision" and kwargs["trust_remote_code"] is False
            models.append(dict(kwargs))
            if failure == "other": raise ValueError("not a checkpoint format error")
            if failure == "always" or failure == "once" and len(models) == 1:
                raise SafetensorError("Error while deserializing header: header too large")
            return cls()
        def to(self,device):
            assert device == "fixture_device"; return self
        def eval(self): return self
    monkeypatch.setitem(sys.modules,"transformers",types.SimpleNamespace(AutoProcessor=Processor,
        AutoModelForZeroShotObjectDetection=Model,Florence2ForConditionalGeneration=Model,Sam3Model=Model))
    if failure == "always":
        with pytest.warns(RuntimeWarning),pytest.raises(RuntimeError,match="fresh local cache") as error:
            PretrainedMethod(method,config,"fixture_device")
        assert isinstance(error.value.__cause__,SafetensorError)
    elif failure == "other":
        with pytest.raises(ValueError,match="not a checkpoint format error"):
            PretrainedMethod(method,config,"fixture_device")
    elif failure == "once":
        with pytest.warns(RuntimeWarning,match="fresh local cache"):
            runner = PretrainedMethod(method,config,"fixture_device")
        assert runner.provenance["models"][-1]["cache_recovered"] is True
        assert runner.provenance["models"][-1]["cache_dir"] == models[-1]["cache_dir"]
    else:
        runner = PretrainedMethod(method,config,"fixture_device")
        assert runner.provenance["models"][-1]["cache_recovered"] is False
    assert len(models) == (2 if failure in ["once","always"] else 1)
    assert len(processors) == len(models)
    assert all(m["use_safetensors"] is True for m in models)
    assert original.read_bytes() == b"preserve this file"
    if len(models) == 2:
        assert models[0]["cache_dir"] != models[1]["cache_dir"]
        assert __import__("pathlib").Path(models[1]["cache_dir"]).parent == initial
        assert processors[1]["cache_dir"] == models[1]["cache_dir"]


@pytest.mark.parametrize("failure",[None,"index_error","invalid_state","always_bad"])
def test_sam_checkpoint_validation_and_fresh_cache_recovery(tmp_path,monkeypatch,failure):
    initial = tmp_path / "local_sam"
    config = {"sam2_id":"facebook/sam2.1-hiera-large","revisions":{},
              "cache":str(tmp_path / "drive"),"sam2_cache":str(initial)}
    downloads,loads,builds = [],[],[]
    weights = {"weight":"fixture only, not a tensor"}
    def download(model_id,filename,revision,cache_dir):
        assert model_id == config["sam2_id"] and revision == "main"
        folder = __import__("pathlib").Path(cache_dir)
        path = folder / "snapshots" / "resolved_revision" / filename
        path.parent.mkdir(parents=True,exist_ok=True)
        if not path.exists(): path.write_bytes(b"fixture checkpoint")
        downloads.append((folder,path))
        return str(path)
    def load(path,map_location,weights_only):
        assert map_location == "cpu" and weights_only is True
        loads.append(path)
        if failure == "always_bad" or failure == "index_error" and len(loads) == 1:
            raise IndexError("pop from empty list")
        if failure == "invalid_state" and len(loads) == 1: return {"model":{}}
        return {"model":weights}
    class Model:
        def load_state_dict(self,state,strict):
            assert state is weights and strict is True
        def to(self,device):
            assert device == "fixture_device"; return self
        def eval(self): return self
    def build(name,ckpt_path,device,apply_postprocessing):
        assert ckpt_path is None and device == "cpu" and apply_postprocessing is False
        builds.append(name); return Model()
    monkeypatch.setitem(sys.modules,"torch",types.SimpleNamespace(load=load))
    monkeypatch.setitem(sys.modules,"huggingface_hub",types.SimpleNamespace(hf_hub_download=download))
    monkeypatch.setitem(sys.modules,"sam2",types.ModuleType("sam2"))
    monkeypatch.setitem(sys.modules,"sam2.build_sam",types.SimpleNamespace(build_sam2=build,
        HF_MODEL_ID_TO_FILENAMES={config["sam2_id"]:("config.yaml","weights.pt")}))
    if failure == "always_bad":
        with pytest.warns(RuntimeWarning),pytest.raises(RuntimeError,match="after a fresh download") as error:
            _sam_model(config,"fixture_device")
        assert isinstance(error.value.__cause__,IndexError)
        assert len(downloads) == 2 and not builds
        return
    if failure:
        with pytest.warns(RuntimeWarning,match="fresh local cache"):
            _,source = _sam_model(config,"fixture_device")
    else:
        _,source = _sam_model(config,"fixture_device")
    assert source["cache_recovered"] == bool(failure)
    assert source["revision"] == "resolved_revision" and len(source["checkpoint_sha256"]) == 64
    assert len(downloads) == (2 if failure else 1) and builds == ["config.yaml"]
    assert downloads[0][1].read_bytes() == b"fixture checkpoint"  # Never replaced.
    if failure:
        assert downloads[1][0] != initial and downloads[1][0].parent == initial
        _sam_model(config,"fixture_device")
        assert downloads[-1][0] == downloads[1][0]  # Later methods reuse recovery.


def test_shared_roi_and_restoration_preserve_holes_and_full_frame_coordinates():
    image = np.zeros((100,200,3),np.uint8)
    cropped,roi = crop_image(image,[.25,.20,.75,.80])
    assert cropped.shape == (60,100,3) and roi == (50,20,150,80)
    mask = np.ones((60,100),bool); mask[10:15,10:15] = False
    restored = restore_masks([{"segmentation":mask,"predicted_iou":.9}],image.shape,roi)
    full = restored[0]["segmentation"]
    assert full.shape == (100,200) and not full[:20].any()
    assert not full[30:35,60:65].any() and full[20,50]
    with pytest.raises(ValueError): crop_image(image,[.8,.2,.1,.8])
    with pytest.raises(ValueError): restore_masks([{"segmentation":np.ones((2,2))}],image.shape,roi)


def test_box_nms_rejects_invalid_values_and_does_not_force_six_objects():
    boxes = [[0,0,10,10],[0,0,10,10],[20,0,30,10],[np.inf,0,40,10]]
    kept,scores,labels = select_boxes(boxes,[.3,.9,None,.99],["a","b","c","bad"],(100,100))
    assert labels == ["b","c"] and scores == [.9,None]
    separated = [[i*12,0,i*12+8,8] for i in range(7)]
    result,_,_ = select_boxes(separated,[.8]*7,["part"]*7,(100,100))
    assert len(result) == 7
    with pytest.raises(ValueError): select_boxes(separated,[],[],(100,100))


def test_ground_truth_matching_is_one_to_one_and_not_count_agreement():
    a = np.zeros((20,20),bool); a[2:6,2:6] = True
    b = np.zeros_like(a); b[12:16,12:16] = True
    exact = mask_metrics([b,a],[a,b])
    assert exact["tp"] == 2 and exact["f1"] == exact["matched_mean_iou"] == 1
    duplicates = mask_metrics([a,a],[a,b])
    assert duplicates["tp"] == duplicates["fp"] == duplicates["fn"] == 1
    assert duplicates["f1"] == .5  # Count is correct, masks are not.
    assert mask_metrics([], [a])["f1"] == 0
    assert mask_metrics([],[])["f1"] is None
    with pytest.raises(ValueError): mask_metrics([a],[np.ones((30,30))])


def test_direct_detector_scores_are_not_relabelled_as_mask_iou(tmp_path):
    image = np.zeros((20,20,3),np.uint8)
    annotations = [{"segmentation":np.ones((20,20),bool),"detector_score":.87,"prompt_label":"metal bracket"}]
    result,_ = save_candidates(image,annotations,tmp_path / "export",maximum_fraction=1)
    row = result["candidates"][0]
    assert row["model_predicted_iou"] is None and row["stability_score"] is None
    assert row["detector_score"] == .87 and row["prompt_label"] == "metal bracket"
    assert json.loads((tmp_path / "export/candidates.json").read_text())["accuracy"] == "not_measured"


class Array:
    def __init__(self,value): self.value = np.asarray(value)
    def cpu(self): return self
    def numpy(self): return self.value
    def tolist(self): return self.value.tolist()
    def float(self): return self
    def __getitem__(self,index): return Array(self.value[index])


@pytest.fixture
def fake_runtime(monkeypatch):
    # Only mock external API plumbing; never import a learned model or run its forward pass.
    monkeypatch.setitem(sys.modules,"torch",types.SimpleNamespace(inference_mode=nullcontext))
    monkeypatch.setitem(sys.modules,"PIL",types.SimpleNamespace(Image=types.SimpleNamespace(
        fromarray=lambda rgb:types.SimpleNamespace(size=(rgb.shape[1],rgb.shape[0])))))
    runner = PretrainedMethod.__new__(PretrainedMethod)
    runner.device = "cpu"
    runner.config = {"prompts":["metal bracket","sheet metal part"],"box_nms":.7,"max_detections":50,
        "dino_threshold":.25,"dino_text_threshold":.2,"owl_threshold":.1,"sam3_threshold":.5,
        "florence_max_tokens":1024,"imgsz":960,"yoloe_threshold":.25}
    return runner


@pytest.mark.parametrize("method,dtype",[("grounding_dino_sam2","float32"),("owlv2_sam2","float32"),
    ("florence2_sam2","float16"),("florence2_sam2","float32")])
def test_two_stage_methods_use_detector_boxes_for_individual_sam_prompts(fake_runtime,method,dtype):
    runner = fake_runtime; runner.name = method
    calls = []
    boxes = [[1,1,15,15],[20,1,30,15]]
    result = {"boxes":Array(boxes),"scores":Array([.9,.8]),"text_labels":["bracket","part"],"labels":Array([0,1])}
    class Inputs(dict):
        def to(self,device,dtype=None):
            calls.append(("input_to",device,dtype))
            return self
    class Processor:
        def __init__(self): self.image_processor = self
        def __call__(self,**kwargs):
            calls.append(("processor",kwargs)); return Inputs(input_ids=[1,2])
        def post_process_grounded_object_detection(self,outputs,ids,**kwargs):
            assert kwargs["threshold"] == .25 and kwargs["text_threshold"] == .2
            assert kwargs["target_sizes"] == [(32,40)]
            return [result]
        def post_process_object_detection(self,outputs,**kwargs):
            assert kwargs["threshold"] == .1 and kwargs["target_sizes"] == [(32,40)]
            return [result]
        def batch_decode(self,tokens,**kwargs): return ["fake generative answer"]
        def post_process_generation(self,text,task,image_size):
            assert task == "<OPEN_VOCABULARY_DETECTION>" and image_size == (40,32)
            return {task:{"bboxes":boxes}}
    runner.processor = Processor()
    class Model:
        def __call__(self,**kwargs): return "fake detector output"
        def generate(self,**kwargs):
            assert kwargs["do_sample"] is False
            assert calls[-1] == ("input_to",runner.device,self.dtype)
            assert kwargs["input_ids"] == [1,2]
            return "fake tokens"
    runner.model = Model()
    runner.model.dtype = dtype
    class Segmenter:
        def set_image(self,rgb): calls.append(("sam_image",rgb.shape))
        def predict(self,box,multimask_output):
            assert multimask_output is False
            calls.append(("sam_box",box.tolist()))
            mask = np.zeros((32,40),bool)
            x1,y1,x2,y2 = box.astype(int); mask[y1:y2,x1:x2] = True
            return mask[None],np.array([.96]),None
    runner.segmenter = Segmenter()
    annotations,details = runner.predict(np.zeros((32,40,3),np.uint8))
    assert len(annotations) == details["prompted_boxes"] == 2
    assert sum(c[0] == "sam_image" for c in calls) == 1
    assert [c[1] for c in calls if c[0] == "sam_box"] == boxes
    assert all(a["predicted_iou"] == .96 for a in annotations)
    if method == "florence2_sam2": assert all(a["detector_score"] is None for a in annotations)


def test_automatic_sam_and_optional_sam3_contracts(fake_runtime):
    runner = fake_runtime; runner.name = "sam2_auto"
    sample = np.zeros((32,40,3),np.uint8)
    runner.generator = types.SimpleNamespace(generate=lambda rgb:[{"segmentation":np.ones((32,40),bool),"predicted_iou":.9}])
    assert len(runner.predict(sample)[0]) == 1
    runner.name = "sam3_text"
    class Inputs(dict):
        def to(self,device): return self
    class Processor:
        def __call__(self,**kwargs):
            assert kwargs["text"] == "metal bracket"
            return Inputs(original_sizes=Array([[32,40]]))
        def post_process_instance_segmentation(self,outputs,**kwargs):
            assert kwargs["target_sizes"] == [[32,40]] and kwargs["mask_threshold"] == .5
            return [{"masks":[Array(np.ones((32,40),bool))],"scores":[.8]}]
    runner.processor = Processor(); runner.model = lambda **kw:"fake"
    annotations,_ = runner.predict(sample)
    assert annotations[0]["detector_score"] == .8 and "predicted_iou" not in annotations[0]


def test_yoloe_direct_masks_use_common_padding_and_do_not_invent_mask_quality(fake_runtime,monkeypatch):
    runner = fake_runtime; runner.name = "yoloe_text"
    monkeypatch.setitem(sys.modules,"ultralytics",types.ModuleType("ultralytics"))
    monkeypatch.setitem(sys.modules,"ultralytics.utils",types.ModuleType("ultralytics.utils"))
    monkeypatch.setitem(sys.modules,"ultralytics.utils.ops",types.SimpleNamespace(scale_masks=lambda tensor,size:tensor))
    result = types.SimpleNamespace(masks=types.SimpleNamespace(data=Array(np.ones((2,32,40)))),
        boxes=types.SimpleNamespace(conf=Array([.9,.8]),cls=Array([0,1])))
    def predict(image,**kwargs):
        assert kwargs["rect"] is False and kwargs["agnostic_nms"] is True
        assert kwargs["max_det"] == 50 and image[0,0].tolist() == [30,20,10]
        return [result]
    runner.model = types.SimpleNamespace(predict=predict)
    sample = np.full((32,40,3),[10,20,30],np.uint8)
    annotations,_ = runner.predict(sample)
    assert len(annotations) == 2 and annotations[0]["segmentation"].shape == (32,40)
    assert "predicted_iou" not in annotations[0] and annotations[0]["detector_score"] == .9
