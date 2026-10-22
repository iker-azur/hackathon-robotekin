"""CPU/CUDA SAM 3 API fixtures only: no downloads, models or real inference."""

from contextlib import nullcontext
from pathlib import Path
import sys
import types

import numpy as np
import pytest

from roboqc.industrial.sam3 import MODEL_ID, select_runtime, load_predictor, predict_image


@pytest.mark.parametrize("available,choice,expected",[(False,"auto","cpu"),(True,"auto","cuda"),
    (True,"cpu","cpu"),(False,"cpu","cpu"),(True,"cuda","cuda"),(False,"cuda",None),(False,"tpu",None)])
def test_sam3_runtime_policy_supports_cpu_and_t4_without_bf16(monkeypatch,available,choice,expected):
    monkeypatch.setitem(sys.modules,"torch",types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda:available),
        float16="float16",float32="float32"))
    if expected is None:
        with pytest.raises(ValueError): select_runtime(choice)
    else:
        assert select_runtime(choice) == (expected,"float16" if expected == "cuda" else "float32")


@pytest.mark.parametrize("device",["cpu","cuda"])
@pytest.mark.parametrize("failure",[None,"once","always","other"])
def test_sam3_safe_loading_device_dtype_and_bounded_recovery(tmp_path,monkeypatch,device,failure):
    calls = []
    class SafetensorError(Exception): pass
    monkeypatch.setitem(sys.modules,"safetensors",types.SimpleNamespace(SafetensorError=SafetensorError))
    monkeypatch.setitem(sys.modules,"torch",types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda:True),
        float16="float16",float32="float32"))
    original = tmp_path / "existing.weights"; original.write_bytes(b"preserve")
    class Processor:
        @classmethod
        def from_pretrained(cls,model_id,**kwargs):
            assert model_id == MODEL_ID == "facebook/sam3"; return cls()
    class Model:
        config = types.SimpleNamespace(_commit_hash="resolved_revision")
        @classmethod
        def from_pretrained(cls,model_id,**kwargs):
            assert model_id == MODEL_ID and kwargs["revision"] == "pinned_revision"
            assert kwargs["use_safetensors"] is True and kwargs["trust_remote_code"] is False
            assert kwargs["dtype"] == ("float16" if device == "cuda" else "float32")
            assert kwargs["attn_implementation"] == "sdpa"
            calls.append(kwargs)
            if failure == "other": raise ValueError("access or configuration error")
            if failure == "always" or failure == "once" and len(calls) == 1:
                raise SafetensorError("header too large")
            return cls()
        def to(self,selected):
            assert selected == device; return self
        def eval(self): return self
    monkeypatch.setitem(sys.modules,"transformers",types.SimpleNamespace(Sam3Processor=Processor,Sam3Model=Model))
    if failure == "always":
        with pytest.warns(RuntimeWarning),pytest.raises(RuntimeError,match="fresh-cache retry"):
            load_predictor(tmp_path,"pinned_revision",device)
    elif failure == "other":
        with pytest.raises(ValueError,match="access or configuration"):
            load_predictor(tmp_path,"pinned_revision",device)
    elif failure == "once":
        with pytest.warns(RuntimeWarning): runner,source = load_predictor(tmp_path,"pinned_revision",device)
        assert source["cache_recovered"] and calls[0]["cache_dir"] != calls[1]["cache_dir"]
    else:
        runner,source = load_predictor(tmp_path,"pinned_revision",device)
        assert source["model_id"] == "facebook/sam3" and not source["cache_recovered"]
        assert runner.device == source["device"] == device
    assert len(calls) == (2 if failure in ["once","always"] else 1)
    assert original.read_bytes() == b"preserve"


@pytest.mark.parametrize("device",["cpu","cuda"])
@pytest.mark.parametrize("case",["normal","empty","int64","uint8","float_binary",
    "bad_shape","bad_scores","not_binary","nonbinary_int","nan_mask"])
def test_sam3_image_processor_contract_and_mask_validation(monkeypatch,device,case):
    # No CUDA calls/autocast are provided: CPU must not try to use them.
    monkeypatch.setitem(sys.modules,"torch",types.SimpleNamespace(inference_mode=nullcontext))
    monkeypatch.setitem(sys.modules,"PIL",types.SimpleNamespace(Image=types.SimpleNamespace(
        fromarray=lambda rgb:types.SimpleNamespace(size=(rgb.shape[1],rgb.shape[0])))))
    dtype = "float16" if device == "cuda" else "float32"
    class Tensor:
        def __init__(self,value): self.value = np.asarray(value)
        def detach(self): return self
        def cpu(self): return self
        def numpy(self): return self.value
        def tolist(self): return self.value.tolist()
    class Inputs(dict):
        def to(self,selected,selected_dtype):
            assert selected == device and selected_dtype == dtype
            return self
    mask = np.ones((20,30),bool); mask[5:8,5:8] = False
    class Processor:
        def __call__(self,images,text,return_tensors):
            assert images.size == (30,20) and text == "metal bracket" and return_tensors == "pt"
            return Inputs(input_ids=[1,2],original_sizes=Tensor([[20,30]]))
        def post_process_instance_segmentation(self,output,threshold,mask_threshold,target_sizes):
            assert threshold == .4 and mask_threshold == .5 and target_sizes == [[20,30]]
            returned = mask
            if case == "bad_shape": returned = np.zeros((2,2),bool)
            elif case == "int64": returned = mask.astype(np.int64)  # Actual Transformers 5.2 output.
            elif case == "uint8": returned = mask.astype(np.uint8)
            elif case == "float_binary": returned = mask.astype(np.float32)
            elif case == "not_binary": returned = mask.astype(float)*.25
            elif case == "nonbinary_int": returned = mask.astype(np.int64)*2
            elif case == "nan_mask": returned = np.full(mask.shape,np.nan)
            masks = [] if case == "empty" else [Tensor(returned)]
            scores = [] if case == "empty" else [np.nan if case == "bad_scores" else .8]
            return [{"masks":masks,"scores":scores}]
    def model(**kwargs):
        assert kwargs["input_ids"] == [1,2]; return "fixture outputs"
    runner = types.SimpleNamespace(model=model,processor=Processor(),device=device,dtype=dtype)
    if case in ["bad_shape","bad_scores","not_binary","nonbinary_int","nan_mask"]:
        with pytest.raises(ValueError): predict_image(runner,np.zeros((20,30,3),np.uint8),"metal bracket")
    else:
        result = predict_image(runner,np.zeros((20,30,3),np.uint8),"metal bracket")
        assert len(result) == (0 if case == "empty" else 1)
        if result:
            assert result[0]["segmentation"].dtype == np.bool_
            assert not result[0]["segmentation"][5:8,5:8].any()
            assert result[0]["detector_score"] == .8 and "predicted_iou" not in result[0]
            mask[:] = False
            assert result[0]["segmentation"].any()
