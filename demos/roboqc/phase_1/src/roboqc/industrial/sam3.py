"""SAM 3 image segmentation through Transformers, on CPU or CUDA. No import-time inference."""

from pathlib import Path
import tempfile
from types import SimpleNamespace
import warnings

import numpy as np

MODEL_ID = "facebook/sam3"


def select_runtime(device="auto"):
    import torch
    if device not in ["auto","cpu","cuda"]: raise ValueError("DEVICE_CHOICE must be auto, cpu or cuda")
    selected = "cuda" if device == "auto" and torch.cuda.is_available() else "cpu" if device == "auto" else device
    if selected == "cuda" and not torch.cuda.is_available(): raise ValueError("CUDA requested but unavailable; use auto or cpu")
    return selected,torch.float16 if selected == "cuda" else torch.float32


def load_predictor(cache, revision="main", device="auto"):
    from safetensors import SafetensorError
    from transformers import Sam3Model, Sam3Processor
    device,dtype = select_runtime(device)
    cache = Path(cache); cache.mkdir(parents=True,exist_ok=True)
    kwargs = {"cache_dir":str(cache),"revision":revision,"trust_remote_code":False}
    for attempt in range(2):
        try:
            processor = Sam3Processor.from_pretrained(MODEL_ID,**kwargs)
            model = Sam3Model.from_pretrained(MODEL_ID,use_safetensors=True,dtype=dtype,
                attn_implementation="sdpa",**kwargs)
        except SafetensorError as exc:
            if attempt:
                raise RuntimeError(f"SAM 3 checkpoint unreadable after fresh-cache retry: {kwargs['cache_dir']}") from exc
            kwargs["cache_dir"] = tempfile.mkdtemp(prefix="recovery_",dir=cache)
            warnings.warn(f"SAM 3 checkpoint read failed: {exc}; retrying in fresh local cache {kwargs['cache_dir']}",RuntimeWarning)
            continue
        break
    model = model.to(device).eval()
    predictor = SimpleNamespace(model=model,processor=processor,device=device,dtype=dtype)
    return predictor,{"model_id":MODEL_ID,"resolved_revision":getattr(model.config,"_commit_hash",None),
        "device":device,"dtype":str(dtype),"cache_dir":kwargs["cache_dir"],"cache_recovered":bool(attempt)}


def predict_image(predictor, rgb, prompt, threshold=.4):
    import torch
    from PIL import Image
    rgb = np.asarray(rgb)
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
        raise ValueError("Expected a uint8 RGB image")
    if not isinstance(prompt,str) or not prompt.strip() or not 0 <= threshold <= 1:
        raise ValueError("Use a nonempty concept prompt and threshold inside [0,1]")
    # BatchFeature casts floating image tensors only, preserving integer token IDs.
    inputs = predictor.processor(images=Image.fromarray(rgb),text=prompt,return_tensors="pt").to(predictor.device,predictor.dtype)
    with torch.inference_mode():
        output = predictor.model(**inputs)
        result = predictor.processor.post_process_instance_segmentation(output,threshold=threshold,
            mask_threshold=.5,target_sizes=inputs["original_sizes"].tolist())[0]
    if len(result["masks"]) != len(result["scores"]): raise ValueError("Inconsistent SAM 3 masks and scores")
    annotations = []
    for mask,score in zip(result["masks"],result["scores"]):
        mask = mask.detach().cpu().numpy()
        score = float(score)
        if mask.shape != rgb.shape[:2]:
            raise ValueError(f"SAM 3 mask shape {mask.shape} does not match image {rgb.shape[:2]}")
        if not np.isfinite(score): raise ValueError(f"SAM 3 returned a nonfinite instance score: {score}")
        # Transformers 5.2 thresholds masks then casts them to torch.long (0/1),
        # not torch.bool. Validate values before converting; never threshold logits here.
        if mask.dtype.kind not in "biuf" or not ((mask == 0) | (mask == 1)).all():
            raise ValueError(f"SAM 3 returned non-binary mask values (dtype={mask.dtype}); expected only 0/1")
        annotations.append({"segmentation":mask.astype(bool,copy=True),"detector_score":score,"prompt_label":prompt})
    return annotations
