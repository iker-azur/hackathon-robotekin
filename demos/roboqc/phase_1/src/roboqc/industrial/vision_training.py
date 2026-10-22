"""Manual Colab training/export helpers for real instance annotations."""

import json
import shutil
from pathlib import Path

import cv2
import numpy as np

from .vision_inference import CLASSES, QUALITY, quality_crop


def train_segmentation(data, output, epochs=80, seed=42, imgsz=640, resume=False):
    import torch
    import yaml
    from ultralytics import YOLO
    data, out = Path(data), Path(output)
    config = yaml.safe_load((data / "data.yaml").read_text())
    config["path"] = str(data.resolve())
    local_yaml = out / "data_resolved.yaml"
    local_yaml.write_text(yaml.safe_dump(config), encoding="utf-8")
    checkpoint = out / "segmentation_train/weights/last.pt"
    model = YOLO(str(checkpoint) if resume and checkpoint.exists() else "yolo11n-seg.pt")
    model.train(data=str(local_yaml), project=str(out), name="segmentation_train", exist_ok=resume,
        epochs=epochs, imgsz=imgsz, batch=8 if torch.cuda.is_available() else 2,
        device=0 if torch.cuda.is_available() else "cpu", workers=2, seed=seed,
        patience=15, resume=bool(resume and checkpoint.exists()),
        degrees=180, translate=.10, scale=.25, fliplr=.5, flipud=.5,
        hsv_h=0, hsv_s=.15, hsv_v=.25, mosaic=.3, close_mosaic=10,
        mixup=0, overlap_mask=False)
    best = YOLO(str(out / "segmentation_train/weights/best.pt"))
    metrics = best.val(data=str(local_yaml), split="val", device="cpu")
    shutil.copy2(out / "segmentation_train/weights/best.pt", out / "segmentation.pt")
    exported = best.export(format="onnx", imgsz=imgsz, dynamic=False, simplify=False, opset=17, device="cpu")
    shutil.copy2(exported, out / "segmentation.onnx")
    summary = {k: float(v) for k,v in metrics.results_dict.items()}
    (out / "segmentation_validation.json").write_text(json.dumps(summary, indent=2))
    manifest = {"schema_version": "industrial-vision-1.0", "classes": CLASSES, "imgsz": imgsz,
                "quality_enabled": False, "quality_threshold": .8, "seed": seed,
                "export_verified": False,
                "geometry": "planar estimates only; grasp candidates unvalidated"}
    (out / "vision_manifest.json").write_text(json.dumps(manifest, indent=2))
    # Compare actual postprocessed CPU outputs, not just whether export returned a path.
    from .vision_inference import Segmenter
    original, optimized = Segmenter(out, False), Segmenter(out, True)
    differences = []
    for path in sorted((data / "images/val").glob("*.png"))[:10]:
        frame = cv2.imread(str(path))
        a,b = original.predict(frame), optimized.predict(frame)
        if len(a) != len(b):
            raise AssertionError(f"Export changes instance count: {path}")
        # Match by mask IoU to avoid relying on backend output order.
        from scipy.optimize import linear_sum_assignment
        if not a:
            continue
        overlaps = np.array([[np.logical_and(x["mask"],y["mask"]).sum()/max(1,np.logical_or(x["mask"],y["mask"]).sum()) for y in b] for x in a])
        rows, cols = linear_sum_assignment(-overlaps)
        for i,j in zip(rows,cols):
            if a[i]["type"] != b[j]["type"] or overlaps[i,j] < .95:
                raise AssertionError(f"Export changes class/mask: {path}")
            differences.append(float(overlaps[i,j]))
    (out / "export_equivalence.json").write_text(json.dumps({"mask_ious": differences, "passed": True}, indent=2))
    manifest["export_verified"] = True
    (out / "vision_manifest.json").write_text(json.dumps(manifest, indent=2))
    return summary


def train_quality(data, output, epochs=20, patience=4, seed=42):
    import torch
    from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
    data, out = Path(data), Path(output)
    definitions = json.loads((data / "quality_definitions.json").read_text())
    records = json.loads((data / "instances.json").read_text())
    examples = {split: [(r,i) for r in records if r["split"] == split for i in r["instances"] if i["quality"] in QUALITY] for split in ("train","val","test")}
    if not all(definitions.get(k) for k in QUALITY) or any(set(i["quality"] for r,i in rows) != set(QUALITY) for rows in examples.values()):
        (out / "quality_pending.json").write_text(json.dumps({"status": "pending", "reason": "Define all three observable classes and annotate every split"}))
        return {"status": "pending"}
    torch.manual_seed(seed)
    def tensors(rows):
        x, y = [], []
        for record, instance in rows:
            frame = cv2.imread(str(data / record["image"]))
            mask = cv2.imread(str(data / instance["mask_path"]), 0) > 0
            rgb = quality_crop(frame,mask).astype(np.float32)/255
            x.append(((rgb-[.485,.456,.406])/[.229,.224,.225]).transpose(2,0,1))
            y.append(QUALITY.index(instance["quality"]))
        return torch.tensor(np.array(x), dtype=torch.float32), torch.tensor(y)
    tx,ty = tensors(examples["train"])
    vx,vy = tensors(examples["val"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
    model.classifier[-1] = torch.nn.Linear(model.classifier[-1].in_features,3)
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(),lr=1e-4)
    first,best,stale = 0,float("inf"),0
    checkpoint = out / "quality_checkpoint.pt"
    if checkpoint.exists():
        state = torch.load(checkpoint,map_location=device,weights_only=True)
        model.load_state_dict(state["model"]); opt.load_state_dict(state["optimizer"])
        first,best,stale = state["epoch"],state["best"],state["stale"]
    for epoch in range(first,epochs):
        model.train()
        for batch in torch.randperm(len(tx)).split(32):
            opt.zero_grad()
            loss = torch.nn.functional.cross_entropy(model(tx[batch].to(device)),ty[batch].to(device))
            loss.backward(); opt.step()
        model.eval()
        with torch.inference_mode():
            score = sum(float(torch.nn.functional.cross_entropy(model(x.to(device)),y.to(device),reduction="sum")) for x,y in zip(vx.split(32),vy.split(32))) / len(vy)
        if score < best:
            best,stale = score,0
            torch.save(model.state_dict(),out / "quality.pt")
        else:
            stale += 1
        torch.save({"model": model.state_dict(),"optimizer":opt.state_dict(),"epoch":epoch+1,"best":best,"stale":stale},checkpoint)
        if stale >= patience:
            break
    model.load_state_dict(torch.load(out / "quality.pt",map_location=device,weights_only=True))
    model = model.cpu().eval()
    sample = vx[:1]
    torch.onnx.export(model,sample,str(out / "quality.onnx"),input_names=["images"],output_names=["logits"],opset_version=17,dynamo=False)
    import onnxruntime as ort
    session = ort.InferenceSession(str(out / "quality.onnx"),providers=["CPUExecutionProvider"])
    with torch.inference_mode():
        expected = model(sample).numpy()
        p = torch.cat([model(x).softmax(-1) for x in vx.split(32)]).numpy()
    np.testing.assert_allclose(expected,session.run(None,{"images":sample.numpy()})[0],atol=1e-4,rtol=1e-3)
    threshold = 1.01
    for candidate in [.5,.7,.8,.9,.95,.99]:
        accepted = p.max(-1) >= candidate
        if accepted.any() and float(np.mean(p.argmax(-1)[accepted] != vy.numpy()[accepted])) <= .05:
            threshold = candidate; break
    manifest = json.loads((out / "vision_manifest.json").read_text())
    manifest.update(quality_enabled=True,quality_threshold=threshold,quality_definitions=definitions)
    (out / "vision_manifest.json").write_text(json.dumps(manifest,indent=2))
    return {"status": "trained", "threshold": threshold, "validation_loss": best}
