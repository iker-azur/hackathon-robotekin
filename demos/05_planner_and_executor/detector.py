"""An optional open-vocabulary detector, for the things colour segmentation cannot name.

Colour segmentation finds "a red thing" and nothing else, which is enough for the
cubes and balls of the scene and useless for "el plátano". This module adds an
open-vocabulary detector so that a person can name an object the robot has never been
told about.

It is off by default and that is a decision, not an oversight:

* **It will not work on the simulation's own objects.** The extra cubes and balls are
  flat-shaded primitives, and an open-vocabulary detector looking for "banana" is being
  asked to find a banana. Score it on a webcam pointed at real objects, or on real USD
  assets, and report it there.
* **It is the slowest thing in the demo** by an order of magnitude, and it needs a
  one-off download of roughly 600 MB.
* **The project's own guide says start with colour segmentation** (level 2 of vision,
  "un detector identifica objetos", with the rule that level 2 is easier to debug than
  level 3). Keeping it behind a flag means the demo can be shown even if the download
  fails on the day.

Implementation note: OWLv2 ships inside ``transformers``, which the Isaac environment
already has at 4.57.6 through gliner2, so this adds no dependency. YOLO-World would
mean a new ``ultralytics`` install into an environment whose versions are pinned
deliberately, which is the kind of change that costs a day.
"""

from __future__ import annotations

import re

import numpy as np

from scene import Thing

MODEL_ID = "google/owlv2-base-patch16-ensemble"

# Below this the box is a suggestion, not a detection. OWLv2 scores text prompts
# independently per box, so it will happily report "banana" at 0.31 for a shadow.
MIN_SCORE = 0.20

# Boxes are normalised to 0..1 by OWLv2 and cover only part of an object, so they are
# squared off to the centre of the box before being recorded. A box that already
# covers most of the frame is left alone.
PAD = 0.5


def parse_queries(text: str) -> list[str]:
    """Split a query list, keeping only the words a detector can be given."""
    return [q.strip() for q in re.split(r"[,;]", text or "") if q.strip()]


def normalize_box(box: list[float], width: int, height: int, name: str, score: float) -> Thing:
    """One detector box as a Thing, squared off to the box centre.

    ``kind`` is ``other`` because an open-vocabulary detector does not answer "cube or
    ball"; it answers the name that was asked for. Grounding already treats ``other`` as
    "anything that is not a ball or a cube", which is the right reading.
    """
    x0, y0, x1, y1 = box
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    half_w, half_h = (x1 - x0) * PAD / 2.0, (y1 - y0) * PAD / 2.0
    x0, x1 = max(0.0, cx - half_w), min(1.0, cx + half_w)
    y0, y1 = max(0.0, cy - half_h), min(1.0, cy + half_h)
    return Thing(
        id=-1,
        kind="other",
        color="unknown",
        centroid_px=[cx * width, cy * height],
        bbox_px=[int(x0 * width), int(y0 * height), max(1, int((x1 - x0) * width)), max(1, int((y1 - y0) * height))],
        area_px=int(max(1, (x1 - x0) * width) * max(1, (y1 - y0) * height)),
        mean_hsv=[0, 0, 0],
        confidence=round(float(score), 3),
        source=f"detector:{name}",
    )


class Owlv2:
    """A thin wrapper that returns Things, loaded only when a detector is asked for."""

    def __init__(self, queries: list[str], model_id: str = MODEL_ID, device: str | None = None):
        import torch
        from transformers import Owlv2ForObjectDetection, Owlv2Processor

        if not queries:
            raise ValueError("an open-vocabulary detector needs at least one query")
        import truststore

        truststore.inject_into_ssl()  # Norton re-signs HTTPS on this machine
        self.torch = torch
        self.queries = queries
        self.processor = Owlv2Processor.from_pretrained(model_id)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = Owlv2ForObjectDetection.from_pretrained(model_id).to(self.device).eval()
        self.threshold = MIN_SCORE

    def __call__(self, image_bgr: np.ndarray) -> list[Thing]:
        import cv2

        height, width = image_bgr.shape[:2]
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        inputs = self.processor(text=[self.queries], images=rgb, return_tensors="pt").to(self.device)
        with self.torch.inference_mode():
            outputs = self.model(**inputs)
        results = self.processor.post_process_grounded_object_detection(
            outputs, threshold=self.threshold, target_sizes=[(height, width)]
        )[0]
        found: list[Thing] = []
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            box = [
                float(box[0]) / width,
                float(box[1]) / height,
                float(box[2]) / width,
                float(box[3]) / height,
            ]
            found.append(normalize_box(box, width, height, str(label), float(score)))
        return found
