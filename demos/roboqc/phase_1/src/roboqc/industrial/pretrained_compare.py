"""Manual Colab comparison adapters. No imports load models or start inference."""

import hashlib
import os
from pathlib import Path
import tempfile
import warnings

import numpy as np

DEFAULT_METHODS = ["yoloe_text","sam2_auto","grounding_dino_sam2","owlv2_sam2","florence2_sam2"]
MODEL_IDS = {"grounding_dino_sam2":"IDEA-Research/grounding-dino-base",
             "owlv2_sam2":"google/owlv2-base-patch16-ensemble",
             "florence2_sam2":"florence-community/Florence-2-base",
             "sam3_text":"facebook/sam3"}


def crop_image(rgb, normalized_roi=None):
    """A shared optional workspace crop; no method gets a different region."""
    height,width = rgb.shape[:2]
    if normalized_roi is None:
        return rgb,(0,0,width,height)
    roi = np.asarray(normalized_roi,dtype=float)
    if roi.shape != (4,) or not np.isfinite(roi).all() or not (0 <= roi[0] < roi[2] <= 1 and 0 <= roi[1] < roi[3] <= 1):
        raise ValueError("ROI must be normalized [left,top,right,bottom] inside [0,1]")
    x1,y1,x2,y2 = int(roi[0]*width),int(roi[1]*height),int(roi[2]*width),int(roi[3]*height)
    if x2 <= x1 or y2 <= y1:
        raise ValueError("ROI is empty at this image resolution")
    return rgb[y1:y2,x1:x2],(x1,y1,x2,y2)


def restore_masks(annotations, image_shape, roi):
    x1,y1,x2,y2 = roi
    restored = []
    for annotation in annotations:
        mask = np.asarray(annotation["segmentation"],dtype=bool)
        if mask.shape != (y2-y1,x2-x1):
            raise ValueError("Mask dimensions do not match the processed ROI")
        full = np.zeros(image_shape[:2],bool)
        full[y1:y2,x1:x2] = mask
        restored.append(dict(annotation,segmentation=full))
    return restored


def select_boxes(boxes, scores, labels, size, iou_threshold=.7, maximum=50):
    """Class-agnostic NMS on detector boxes, not a forced object count."""
    boxes = np.asarray(boxes,dtype=float).reshape(-1,4)
    if len(scores) != len(boxes) or len(labels) != len(boxes):
        raise ValueError("Mismatched detector boxes, scores and labels")
    width,height = size
    finite = np.isfinite(boxes).all(axis=1)
    boxes[:,[0,2]] = boxes[:,[0,2]].clip(0,width)
    boxes[:,[1,3]] = boxes[:,[1,3]].clip(0,height)
    valid = [i for i,b in enumerate(boxes) if finite[i] and b[2] > b[0] and b[3] > b[1]
             and (scores[i] is None or np.isfinite(scores[i]))]
    order = sorted(valid,key=lambda i:float(scores[i]) if scores[i] is not None else 0,reverse=True)
    keep = []
    while order and len(keep) < maximum:
        i = order.pop(0); keep.append(i)
        survivors = []
        for j in order:
            intersection = np.maximum(0,np.minimum(boxes[i,2:],boxes[j,2:])-np.maximum(boxes[i,:2],boxes[j,:2])).prod()
            area_i = (boxes[i,2:]-boxes[i,:2]).prod(); area_j = (boxes[j,2:]-boxes[j,:2]).prod()
            if intersection/max(1,area_i+area_j-intersection) <= iou_threshold:
                survivors.append(j)
        order = survivors
    return boxes[keep], [scores[i] for i in keep], [labels[i] for i in keep]


def mask_metrics(predicted, reference, threshold=.5):
    """One-to-one mask matching, maximizing qualifying matches before IoU."""
    from scipy.optimize import linear_sum_assignment
    if any(np.asarray(m).ndim != 2 for m in [*predicted,*reference]):
        raise ValueError("Ground truth and predictions must be 2D binary masks")
    shapes = {np.asarray(m).shape for m in [*predicted,*reference]}
    if len(shapes) > 1: raise ValueError("Ground-truth masks must match original image dimensions")
    overlaps = np.zeros((len(reference),len(predicted)))
    for i,a in enumerate(reference):
        for j,b in enumerate(predicted):
            overlaps[i,j] = np.logical_and(a,b).sum()/max(1,np.logical_or(a,b).sum())
    matched = []
    if overlaps.size:
        # A plain maximum-IoU assignment can undercount threshold-qualified matches.
        bonus = min(overlaps.shape)+1
        rows,columns = linear_sum_assignment(-((overlaps >= threshold)*bonus+overlaps))
        matched = [float(overlaps[i,j]) for i,j in zip(rows,columns) if overlaps[i,j] >= threshold]
    tp = len(matched); fp = len(predicted)-tp; fn = len(reference)-tp
    return {"mask_iou_threshold":threshold,"tp":tp,"fp":fp,"fn":fn,
        "precision":tp/(tp+fp) if tp+fp else None,"recall":tp/(tp+fn) if tp+fn else None,
        "f1":2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None,
        "matched_mean_iou":float(np.mean(matched)) if matched else None}


def _sam_model(config, device):
    import torch
    from huggingface_hub import hf_hub_download
    from sam2.build_sam import build_sam2, HF_MODEL_ID_TO_FILENAMES
    name,filename = HF_MODEL_ID_TO_FILENAMES[config["sam2_id"]]
    # Large checkpoint reads stay on Colab's local disk, not the Drive mount.
    cache = Path(config.get("sam2_cache",Path(tempfile.gettempdir()) / "roboqc_sam2_cache"))
    cache.mkdir(parents=True,exist_ok=True)
    revision = config["revisions"].get(config["sam2_id"],"main")
    recovered = False
    for attempt in range(2):
        path = Path(hf_hub_download(config["sam2_id"],filename,revision=revision,cache_dir=str(cache)))
        try:
            state = torch.load(str(path),map_location="cpu",weights_only=True)
            if not isinstance(state,dict) or not isinstance(state.get("model"),dict) or not state["model"]:
                raise ValueError("SAM checkpoint must contain a nonempty model state dictionary")
        except Exception as exc:
            if attempt:
                raise RuntimeError(f"SAM checkpoint cannot be read after a fresh download: {path}. "
                                   "Existing files were preserved; inspect the chained error.") from exc
            # Never overwrite/delete a possibly damaged cache entry or relax weights_only.
            cache = Path(tempfile.mkdtemp(prefix="recovery_",dir=cache))
            warnings.warn(f"SAM checkpoint read failed ({type(exc).__name__}: {exc}); "
                          f"retrying once in a fresh local cache: {cache}",RuntimeWarning)
            recovered = True
            continue
        break
    config["sam2_cache"] = str(cache)  # Reuse a recovered download for later methods.
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda:handle.read(1024*1024),b""): digest.update(chunk)
    # Load the already validated weights once; do not reopen the checkpoint in SAM.
    model = build_sam2(name,ckpt_path=None,device="cpu",apply_postprocessing=False)
    model.load_state_dict(state["model"],strict=True)
    del state
    return model.to(device).eval(),{"model_id":config["sam2_id"],"revision":path.parent.name,
        "checkpoint_sha256":digest.hexdigest(),"checkpoint_path":str(path),"cache_recovered":recovered}


class PretrainedMethod:
    """Load only one method per instance; caller releases it before the next method."""
    def __init__(self, name, config, device):
        self.name,self.config,self.device = name,config,device
        self.provenance = {"method":name,"models":[]}
        if name == "sam2_auto" or name.endswith("_sam2"):
            self.sam,source = _sam_model(config,device)
            self.provenance["models"].append(source)
            if name == "sam2_auto":
                from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
                self.generator = SAM2AutomaticMaskGenerator(self.sam,points_per_side=config["points_per_side"],
                    points_per_batch=config["points_per_batch"],crop_n_layers=config["crop_layers"],
                    pred_iou_thresh=config["sam_iou"],stability_score_thresh=config["sam_stability"],
                    min_mask_region_area=0,use_m2m=True,output_mode="binary_mask")
            else:
                from sam2.sam2_image_predictor import SAM2ImagePredictor
                self.segmenter = SAM2ImagePredictor(self.sam)
        if name == "yoloe_text":
            from ultralytics import YOLOE
            directory = Path(config["cache"]) / "yoloe"; directory.mkdir(parents=True,exist_ok=True)
            previous = Path.cwd()
            try:
                os.chdir(directory)
                self.model = YOLOE(config["yoloe_checkpoint"],task="segment")
                self.model.set_classes(config["prompts"],self.model.get_text_pe(config["prompts"]))
            finally:
                os.chdir(previous)
            checkpoint = directory / Path(config["yoloe_checkpoint"]).name
            self.provenance["models"].append({"checkpoint":config["yoloe_checkpoint"],
                "sha256":hashlib.sha256(checkpoint.read_bytes()).hexdigest() if checkpoint.is_file() else None})
        elif name in MODEL_IDS:
            from safetensors import SafetensorError
            from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection, Florence2ForConditionalGeneration, Sam3Model
            model_id = MODEL_IDS[name]
            kwargs = {"cache_dir":config["cache"],"revision":config["revisions"].get(model_id,"main"),"trust_remote_code":False}
            model_class = Florence2ForConditionalGeneration if name == "florence2_sam2" else Sam3Model if name == "sam3_text" else AutoModelForZeroShotObjectDetection
            for attempt in range(2):
                try:
                    self.processor = AutoProcessor.from_pretrained(model_id,**kwargs)
                    self.model = model_class.from_pretrained(model_id,use_safetensors=True,**kwargs)
                except SafetensorError as exc:
                    if attempt:
                        raise RuntimeError(f"{model_id} weights remain unreadable after a fresh local cache "
                                           f"download: {kwargs['cache_dir']}. Previous files were preserved.") from exc
                    cache = Path(config["cache"]); cache.mkdir(parents=True,exist_ok=True)
                    kwargs["cache_dir"] = tempfile.mkdtemp(prefix=name + "_recovery_",dir=cache)
                    warnings.warn(f"{model_id}: {exc}; retrying once in a fresh local cache: "
                                  f"{kwargs['cache_dir']}",RuntimeWarning)
                    continue
                break
            self.model = self.model.to(device).eval()
            self.provenance["models"].append({"model_id":model_id,"revision":getattr(self.model.config,"_commit_hash",None),
                "cache_dir":str(kwargs["cache_dir"]),"cache_recovered":bool(attempt)})
        elif name != "sam2_auto":
            raise ValueError(f"Unknown comparison method: {name}")

    def predict(self, rgb):
        import torch
        from PIL import Image
        cfg = self.config; image = Image.fromarray(rgb); height,width = rgb.shape[:2]
        with torch.inference_mode():
            if self.name == "sam2_auto":
                return self.generator.generate(rgb),{"score_type":"SAM predicted IoU; not observed accuracy"}
            if self.name == "yoloe_text":
                from ultralytics.utils.ops import scale_masks
                result = self.model.predict(rgb[:,:,::-1].copy(),device=self.device,imgsz=cfg["imgsz"],
                    conf=cfg["yoloe_threshold"],rect=False,agnostic_nms=True,iou=cfg["box_nms"],
                    max_det=cfg["max_detections"],retina_masks=False,verbose=False)[0]
                if result.masks is None: return [],{"score_type":"YOLO detection confidence"}
                masks = scale_masks(result.masks.data[:,None].float().cpu(),(height,width))[:,0].numpy() > .5
                scores = result.boxes.conf.cpu().tolist(); classes = result.boxes.cls.cpu().tolist()
                annotations = [{"segmentation":m,"detector_score":float(s),"prompt_label":cfg["prompts"][int(c)]}
                               for m,s,c in zip(masks,scores,classes)]
                return annotations,{"score_type":"YOLO detection confidence; no predicted mask IoU"}
            if self.name == "sam3_text":
                # One concept prompt; use the first configured phrase for this adapter.
                inputs = self.processor(images=image,text=cfg["prompts"][0],return_tensors="pt").to(self.device)
                outputs = self.model(**inputs)
                result = self.processor.post_process_instance_segmentation(outputs,threshold=cfg["sam3_threshold"],
                    mask_threshold=.5,target_sizes=inputs["original_sizes"].tolist())[0]
                return [{"segmentation":m.cpu().numpy().astype(bool),"detector_score":float(s),"prompt_label":cfg["prompts"][0]}
                        for m,s in zip(result["masks"],result["scores"])],{"score_type":"SAM 3 instance score"}
            boxes,scores,labels = [],[],[]
            if self.name == "grounding_dino_sam2":
                inputs = self.processor(images=image,text=". ".join(cfg["prompts"])+".",return_tensors="pt").to(self.device)
                outputs = self.model(**inputs)
                result = self.processor.post_process_grounded_object_detection(outputs,inputs["input_ids"],
                    threshold=cfg["dino_threshold"],text_threshold=cfg["dino_text_threshold"],target_sizes=[(height,width)])[0]
                boxes,scores,labels = result["boxes"].cpu().numpy(),result["scores"].cpu().tolist(),result["text_labels"]
            elif self.name == "owlv2_sam2":
                inputs = self.processor(images=image,text=[cfg["prompts"]],return_tensors="pt").to(self.device)
                outputs = self.model(**inputs)
                result = self.processor.image_processor.post_process_object_detection(outputs,threshold=cfg["owl_threshold"],target_sizes=[(height,width)])[0]
                boxes,scores = result["boxes"].cpu().numpy(),result["scores"].cpu().tolist()
                labels = [cfg["prompts"][int(i)] for i in result["labels"].cpu().tolist()]
            elif self.name == "florence2_sam2":
                for phrase in cfg["prompts"]:
                    task = "<OPEN_VOCABULARY_DETECTION>"
                    # BatchFeature casts floating image tensors, keeping token IDs integer.
                    inputs = self.processor(images=image,text=task+phrase,return_tensors="pt").to(self.device,self.model.dtype)
                    tokens = self.model.generate(**inputs,max_new_tokens=cfg["florence_max_tokens"],num_beams=3,do_sample=False)
                    text = self.processor.batch_decode(tokens,skip_special_tokens=False)[0]
                    result = self.processor.post_process_generation(text,task=task,image_size=image.size)[task]
                    found = result.get("bboxes",[])
                    boxes.extend(found); scores.extend([None]*len(found)); labels.extend([phrase]*len(found))
            before_nms = len(boxes)
            boxes,scores,labels = select_boxes(boxes,scores,labels,image.size,cfg["box_nms"],cfg["max_detections"])
            if not len(boxes): return [],{"detector_boxes_before_nms":before_nms,"prompted_boxes":0}
            self.segmenter.set_image(rgb)
            annotations = []
            # Single-box calls avoid ambiguous multi-box/multimask output dimensions.
            for box,score,label in zip(boxes,scores,labels):
                masks,qualities,_ = self.segmenter.predict(box=box.astype(np.float32),multimask_output=False)
                annotations.append({"segmentation":masks[0].astype(bool),"predicted_iou":float(qualities[0]),
                                    "detector_score":score,"prompt_label":str(label)})
            return annotations,{"detector_boxes_before_nms":before_nms,"prompted_boxes":len(boxes),
                                "score_type":"Detector score and SAM predicted IoU are distinct; Florence detector scores unavailable"}
