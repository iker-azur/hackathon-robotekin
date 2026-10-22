"""Pretrained YOLOE preparation and unlabelled inspection. Nothing runs on import."""

import json
import shutil
import time
from pathlib import Path

import cv2
import numpy as np

from .vision_inference import CLASSES, Segmenter, draw_instances

DEFAULT_PROMPTS = {"bolt":"bolt", "nut":"metal nut", "washer":"metal washer", "bracket":"metal bracket"}


DEMO_IMAGES = [
    {"filename":"bolts_nuts_washers.jpg","title":"Bolts,Nuts & Washers","author":"Sbrennan83",
     "source":"https://commons.wikimedia.org/wiki/File:Bolts,Nuts_%26_Washers.jpg",
     "url":"https://upload.wikimedia.org/wikipedia/commons/e/e0/Bolts%2CNuts_%26_Washers.jpg",
     "license":"CC BY-SA 4.0","license_url":"https://creativecommons.org/licenses/by-sa/4.0/",
     "sha1":"b39b133808a4a39e49a04d64aa55c4104f2cb7fe"},
    {"filename":"angle_bracket.jpg","title":"Bronze angle bracket - 4 x 4 cm - A","author":"Fructibus",
     "source":"https://commons.wikimedia.org/wiki/File:Bronze_angle_bracket_-_4_x_4_cm_-_A.jpg",
     "url":"https://upload.wikimedia.org/wikipedia/commons/0/0a/Bronze_angle_bracket_-_4_x_4_cm_-_A.jpg",
     "license":"CC0 1.0","license_url":"https://creativecommons.org/publicdomain/zero/1.0/",
     "sha1":"54d9402f7f7c55b3964cfd92631edde51d406e23"},
    {"filename":"four_gray_brackets.jpg","title":"4 x Gray angle bracket - 12.5 cm - A","author":"Fructibus",
     "source":"https://commons.wikimedia.org/wiki/File:4_x_Gray_angle_bracket_-_12.5_cm_-_A.jpg",
     "url":"https://upload.wikimedia.org/wikipedia/commons/b/b4/4_x_Gray_angle_bracket_-_12.5_cm_-_A.jpg",
     "license":"CC0 1.0","license_url":"https://creativecommons.org/publicdomain/zero/1.0/",
     "sha1":"0fbc4e465a512ba6830cbd17fe744f49e6c32415",
     "notes":"Multiple formed angle brackets; related geometry, not the reference's exact stamped parts."},
    {"filename":"amplifier_bracket.jpg","title":"Bracket for amp. (25960984807)","author":"Robbie Sproule",
     "source":"https://commons.wikimedia.org/wiki/File:Bracket_for_amp._(25960984807).jpg",
     "url":"https://upload.wikimedia.org/wikipedia/commons/3/34/Bracket_for_amp._%2825960984807%29.jpg",
     "license":"CC BY 2.0","license_url":"https://creativecommons.org/licenses/by/2.0/",
     "sha1":"fac7bc1c96c891cc2b412334f2028d79a374b94c",
     "notes":"Stereo pair of the same formed bracket, not two independent scenes; not an object-count benchmark."},
    {"filename":"channel_connector.jpg","title":"Hilti Schienen Eckverbinder 90 degrees","author":"Best Tech Nick 25",
     "source":"https://commons.wikimedia.org/wiki/File:Hilti_Schienen_Eckverbinder_90%C2%B0.jpg",
     "url":"https://upload.wikimedia.org/wikipedia/commons/a/a2/Hilti_Schienen_Eckverbinder_90%C2%B0.jpg",
     "license":"CC0 1.0","license_url":"https://creativecommons.org/publicdomain/zero/1.0/",
     "sha1":"58260c1883db49524791dd84a2042e16b348071d",
     "notes":"Assembled bracket/channel connection: background-clutter test, not a loose bin-picking scene."},
]


def download_demo_images(output):
    """Download credited, unlabelled real photographs; never run inference."""
    import hashlib
    from datetime import datetime, timezone
    from email.utils import parsedate_to_datetime
    from urllib.error import HTTPError
    from urllib.request import Request, urlopen
    out = Path(output)
    out.mkdir(parents=True,exist_ok=True)
    available,skipped = [],[]
    blocked_reason = None
    requested = False
    for item in DEMO_IMAGES:
        path = out / item["filename"]
        if path.exists():
            data = path.read_bytes()
        elif blocked_reason:
            skipped.append({"filename":item["filename"],"reason":blocked_reason,"source":item["source"]})
            continue
        else:
            if requested:
                time.sleep(2)  # Pace sequential downloads; do not burst from shared Colab IPs.
            requested = True
            print("Downloading demo photo:",item["title"],flush=True)
            request = Request(item["url"],headers={"User-Agent":"RobotekinDemo/1.0 (educational robotics image preview)"})
            data,failure = None,""
            for attempt in range(3):
                try:
                    with urlopen(request,timeout=45) as response:
                        data = response.read(16*1024*1024+1)
                    break
                except HTTPError as exc:
                    failure = f"HTTP {exc.code}: {exc.reason}"
                    delay = 5 * 2**attempt
                    retry_after = exc.headers.get("Retry-After") if exc.headers else None
                    if retry_after:
                        try:
                            delay = max(delay,float(retry_after))
                        except ValueError:
                            try:
                                deadline = parsedate_to_datetime(retry_after)
                                delay = max(delay,(deadline-datetime.now(timezone.utc)).total_seconds())
                            except (TypeError,ValueError,OverflowError):
                                pass
                    if exc.code not in (429,500,502,503,504):
                        break
                    if attempt == 2 or delay > 60:
                        if exc.code == 429 or delay > 60:
                            blocked_reason = failure + "; further network requests deferred; retry later or upload photos"
                        break
                    print(f"{failure}; retry {attempt+1}/2 in {delay:g}s",flush=True)
                    time.sleep(delay)
                except OSError as exc:
                    failure = str(exc)
                    break
            if data is None:
                skipped.append({"filename":item["filename"],"reason":failure,"source":item["source"]})
                print("WARNING: photo unavailable:",item["filename"],"|",failure,flush=True)
                continue
        if len(data) > 16*1024*1024 or hashlib.sha1(data).hexdigest() != item["sha1"]:
            raise ValueError(f"Demo image checksum mismatch: {path}; existing files are not replaced")
        if cv2.imdecode(np.frombuffer(data,dtype=np.uint8),cv2.IMREAD_COLOR) is None:
            raise ValueError(f"Unreadable demo photo: {path}")
        if not path.exists():
            with path.open("xb") as handle:
                handle.write(data)
        print(item["filename"],"|",item["author"],"|",item["license"])
        if item.get("notes"):
            print("  Note:",item["notes"])
        available.append(item)
    (out / "sources.json").write_text(json.dumps(DEMO_IMAGES,indent=2),encoding="utf-8")
    report = {"status":"complete" if not skipped else "partial" if available else "unavailable",
              "requested":len(DEMO_IMAGES),"available":[i["filename"] for i in available],"skipped":skipped}
    (out / "download_report.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    credits = [f"{i['filename']}: {i['title']} by {i['author']}\n{i['source']}\n{i['license']}: {i['license_url']}\nChanges: none (original bytes).\n{i.get('notes','')}\n" for i in available]
    (out / "ATTRIBUTION.txt").write_text("\n".join(credits),encoding="utf-8")
    if not available:
        raise RuntimeError(f"No demo photos available. Details: {out / 'download_report.json'}. "
                           "Set USE_DEMO_IMAGES=False and supply IMAGE_PATH or upload photos.")
    if skipped:
        print(f"WARNING: continuing with {len(available)}/{len(DEMO_IMAGES)} demo photos. "
              f"Skipped images and reasons: {out / 'download_report.json'}",flush=True)
    print("Demo photos have no instance labels; accuracy/mAP cannot be measured.")
    return out


def find_images(path, limit=50):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Image path does not exist: {path}")
    candidates = [path] if path.is_file() else sorted(path.rglob("*"))
    images = [p for p in candidates if p.suffix.lower() in (".png",".jpg",".jpeg",".bmp",".webp")]
    if not images:
        raise ValueError("Provide ordinary part photos or a ZIP of photos; no annotations are needed")
    return images[:limit] if limit is not None else images


def compare_instances(original, exported, minimum_iou=.95):
    from scipy.optimize import linear_sum_assignment
    if len(original) != len(exported):
        raise AssertionError(f"ONNX export changed the instance count: PyTorch={len(original)}, ONNX={len(exported)}")
    if not original:
        return []
    similarities = np.array([[np.logical_and(a["mask"],b["mask"]).sum()/max(1,np.logical_or(a["mask"],b["mask"]).sum())
                              if a["type"] == b["type"] else 0.0 for b in exported] for a in original])
    rows, columns = linear_sum_assignment(-similarities)
    scores = [float(similarities[i,j]) for i,j in zip(rows,columns)]
    if min(scores) < minimum_iou:
        raise AssertionError("ONNX export changed classes or masks; retain the PyTorch bundle or investigate export")
    return scores


def prepare_bundle(output, images, checkpoint="yoloe-26n-seg.pt", prompts=None, imgsz=640,
                   confidence=.25, export_onnx=True, seed=42):
    prompts = DEFAULT_PROMPTS if prompts is None else prompts
    if set(prompts) != set(CLASSES) or any(not isinstance(v,str) or not v.strip() for v in prompts.values()):
        raise ValueError(f"Provide one nonempty text prompt for each canonical class: {CLASSES}")
    texts = [prompts[c].strip() for c in CLASSES]
    if len(set(texts)) != len(texts):
        raise ValueError("Prompts must distinguish all four classes")
    if not images:
        raise ValueError("Supply at least one ordinary photo for export verification")
    out = Path(output)
    out.mkdir(parents=True,exist_ok=False)
    from ultralytics import YOLOE
    model = YOLOE(str(checkpoint),task="segment")
    model.set_classes(texts,model.get_text_pe(texts))
    names = model.names
    actual = [names[i] for i in range(len(names))] if isinstance(names,dict) else list(names)
    if actual != texts:
        raise ValueError("YOLOE class order does not match the requested prompts")
    # Embeddings are saved in the checkpoint; local inference never calls set_classes.
    model.save(str(out / "segmentation.pt"))
    del model
    model = YOLOE(str(out / "segmentation.pt"),task="segment")
    manifest = {"schema_version":"industrial-vision-1.0","model_family":"yoloe",
        "source_checkpoint":str(checkpoint),"classes":list(CLASSES),"prompts":dict(zip(CLASSES,texts)),
        "imgsz":imgsz,"confidence":confidence,"seed":seed,"quality_enabled":False,
        "training_performed":False,"export_verified":False,
        "geometry":"planar estimates only; grasp candidates unvalidated"}
    manifest_path = out / "vision_manifest.json"
    manifest_path.write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    if export_onnx:
        exported = model.export(format="onnx",imgsz=imgsz,opset=17,dynamic=False,simplify=False,device="cpu")
        destination = out / "segmentation.onnx"
        if Path(exported).resolve() != destination.resolve():
            shutil.copy2(exported,destination)
    del model
    original = Segmenter(out,prefer_onnx=False,confidence=confidence)
    optimized = Segmenter(out,prefer_onnx=True,confidence=confidence) if export_onnx else None
    comparison = {"onnx_exported":export_onnx,"images":[],"mask_ious":[],"ground_truth_accuracy":"not_measured",
        "inference":{"imgsz":imgsz,"confidence":confidence,"rect":False,"agnostic_nms":True,
                     "iou":.7,"max_det":300,"retina_masks":False,"augment":False},"passed":False}
    comparison_path = out / "export_equivalence.json"
    for path in list(images)[:3]:
        frame = cv2.imread(str(path))
        if frame is None:
            raise ValueError(f"Unreadable image: {path}")
        instances = original.predict(frame)
        entry = {"path":str(path),"predicted_instances":len(instances),
                 "pytorch":[{k:obj[k] for k in ("type","confidence","bbox")} for obj in instances]}
        comparison["images"].append(entry)
        if optimized is not None:
            onnx_instances = optimized.predict(frame)
            entry["onnx_predicted_instances"] = len(onnx_instances)
            entry["onnx"] = [{k:obj[k] for k in ("type","confidence","bbox")} for obj in onnx_instances]
            try:
                comparison["mask_ious"].extend(compare_instances(instances,onnx_instances))
            except AssertionError as exc:
                comparison["error"] = str(exc)
                comparison_path.write_text(json.dumps(comparison,indent=2),encoding="utf-8")
                raise AssertionError(f"{exc}. Image: {path}. Diagnostics: {comparison_path}. "
                                     "ONNX remains unverified; set EXPORT_ONNX=False and rerun for a PT-only bundle.") from exc
    comparison["passed"] = True
    comparison_path.write_text(json.dumps(comparison,indent=2),encoding="utf-8")
    manifest["export_verified"] = True
    manifest_path.write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    return manifest


def inspect_images(bundle, images, output, confidence=.25, cpu_threads=2, max_visualizations=10):
    """Export overlays and masks; explicitly abstain from unmeasurable accuracy."""
    import torch
    torch.set_num_threads(cpu_threads)
    out = Path(output)
    out.mkdir(parents=True,exist_ok=False)
    if not images:
        raise ValueError("No input images")
    model = Segmenter(bundle,confidence=confidence)
    first = cv2.imread(str(images[0]))
    if first is None:
        raise ValueError(f"Unreadable image: {images[0]}")
    model.predict(first)  # warmup, excluded from latency
    rows,timings = [],[]
    for number,path in enumerate(images):
        frame = cv2.imread(str(path))
        if frame is None:
            raise ValueError(f"Unreadable image: {path}")
        start = time.perf_counter()
        objects = model.predict(frame)
        timings.append((time.perf_counter()-start)*1000)
        instances = []
        for index,obj in enumerate(objects):
            mask_file = f"{number:04d}_mask_{index:03d}.png"
            cv2.imwrite(str(out / mask_file),obj["mask"]*255)
            instances.append({k:v for k,v in obj.items() if k != "mask"} | {"mask_file":mask_file})
        overlay = None
        if number < max_visualizations:
            overlay = f"{number:04d}_overlay.png"
            cv2.imwrite(str(out / overlay),draw_instances(frame,objects))
        rows.append({"image":str(path),"objects":instances,"overlay":overlay,"latency_ms":timings[-1]})
    report = {"evaluation_mode":"unlabelled","images":len(rows),"device":"cpu",
        "confidence_threshold":confidence,"cpu_threads":cpu_threads,
        "latency_ms":{"p50":float(np.percentile(timings,50)),"p95":float(np.percentile(timings,95))},
        "accuracy":{"status":"not_measured","reason":"No ground-truth instance annotations supplied"},
        "quality":{"status":"unknown","reason":"Pretrained YOLOE is not a defect classifier"}}
    (out / "vision_report.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    (out / "predictions.json").write_text(json.dumps(rows,indent=2),encoding="utf-8")
    return report,rows
