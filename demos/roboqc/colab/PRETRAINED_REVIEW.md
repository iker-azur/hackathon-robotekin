# Pretrained alternatives for the actual stamped brackets

Review date: 2026-10-20. This is a broad, primary-source review of practical
available model families, not an exhaustive review of every model on the internet.
No weights, model inference, benchmarks or notebook cells were run by the coding
agent. Nothing below establishes accuracy on the user's parts.

## What the comparison needs to distinguish

`data/image_test_1.jpg` and `image_test_2.jpg` show six separated physical pieces;
`image_test_3.jpg` is an overlapping pile. Whole-part segmentation must include
visible bent flanges and attached pins while excluding the background seen through
holes. Gloves/cables/table edges are distractors. A correct count alone is not a
correct set of masks. Hidden surfaces and stack height remain unknown.

The central hypothesis is that automatic whole-part localization followed by a
box-prompted mask decoder may be more useful than unrestricted region proposals.
This must be tested, not assumed. A detector miss cannot be repaired by SAM.

## Methods implemented in 04c

| Method | Released pretrained model / API | Why include it | Main tradeoff |
|---|---|---|---|
| YOLOE text segmentation | `yoloe-26l-seg.pt` | Larger direct-mask baseline; no separate detector-to-mask stage | General text semantics may still miss unfamiliar stamped parts; Ultralytics licensing |
| SAM 2.1 automatic masks | `facebook/sam2.1-hiera-large` | Existing class-agnostic baseline under identical inputs/cropping | Proposes surfaces/subparts/background; no target semantics |
| Grounding DINO + SAM 2.1 | `IDEA-Research/grounding-dino-base` | Ground text phrases into boxes, then request one mask per box | Two model stages; detector misses and merged boxes propagate |
| OWLv2 + SAM 2.1 | `google/owlv2-base-patch16-ensemble` | Contrastive open-vocabulary detection as a different proposal mechanism | Prompt sensitivity, overlapping duplicate boxes; resource usage |
| Florence-2 + SAM 2.1 | `florence-community/Florence-2-base` | Generative open-vocabulary box proposals rather than a conventional detection head | Generative mistakes/truncation; no calibrated per-box confidence returned by this task |
| Optional SAM 3 | `facebook/sam3`, native Transformers API | Text-prompted instance masks from a different segmentation family | Checkpoint access approval/token and model-license review; not needed for the five defaults |

The three detector-plus-SAM methods deliberately share the same SAM checkpoint.
This compares localization strategies while holding mask refinement constant;
they are different pipelines, not five wholly independent mask decoders.
Their latency includes both stages. SAM 3 currently uses the first configured
phrase; the other text detectors receive the configured phrase list (Florence
runs the phrases separately). This difference is recorded/documented, not hidden.

Primary sources:

- [Ultralytics YOLOE](https://docs.ultralytics.com/models/yoloe/): available text/visual models and export semantics. Larger generic benchmarks are not a stamped-part guarantee.
- [Meta SAM 2](https://github.com/facebookresearch/sam2): automatic and box-prompted masks; source pinned to commit `2b90b9f5ceec907a1c18123530e92e794ad901a4`.
- [Grounding DINO model card](https://huggingface.co/IDEA-Research/grounding-dino-base) and [native processor source, Transformers 5.2.0](https://github.com/huggingface/transformers/blob/v5.2.0/src/transformers/models/grounding_dino/processing_grounding_dino.py).
- [OWLv2 documentation](https://huggingface.co/docs/transformers/v5.2.0/model_doc/owlv2) and [checkpoint](https://huggingface.co/google/owlv2-base-patch16-ensemble).
- [Florence-2 documentation](https://huggingface.co/docs/transformers/v5.2.0/en/model_doc/florence2) and [native Transformers checkpoint](https://huggingface.co/florence-community/Florence-2-base): use native classes, not arbitrary remote model code.
- [Grounded SAM 2 reference project](https://github.com/IDEA-Research/Grounded-SAM-2): established detector/grounding plus SAM pipeline implementations.
- [SAM 3 checkpoint/access terms](https://huggingface.co/facebook/sam3) and [Transformers API](https://huggingface.co/docs/transformers/v5.2.0/en/model_doc/sam3).

Check each source's current code and weight licenses before distribution. The
notebook intentionally installs Transformers 5.2.0 for the verified API signatures,
does not enable `trust_remote_code`, and does not mix in Optimum ONNX.

## Other families considered

- [RF-DETR segmentation](https://inference-models.roboflow.com/models/rfdetr-instance-segmentation/): released generic weights target COCO categories. Good custom-training candidate, but not a fair out-of-box stamped-bracket detector without matching labels.
- Closed-set COCO YOLO/Mask R-CNN: same category mismatch; changing the output class name does not teach the model a new part.
- [HQ-SAM](https://github.com/SysCV/sam-hq): a useful boundary-quality alternative, but a prompted mask model does not by itself solve finding the intended complete pieces.
- Smaller SAM variants primarily address compute cost; faster region generation is not evidence of better object grouping.
- Community industrial models require matching object types, downloadable trusted weights and usable licenses. No verified universal pretrained model for these exact brackets was found in this review.

## How to run and interpret 04c

1. Start a fresh Colab T4 GPU runtime; upload `04c_compare_pretrained_segmentation.ipynb`.
2. Set `IMAGE_PATH` to a Drive image/folder, or leave it blank to upload your own
   photos. The public notebooks contain no embedded dataset/reference images.
   Dependencies/weights still need internet and several GB of cache.
3. Run all cells. Methods load sequentially; CPU is detected but can be very slow.
   `load_seconds_including_downloads` is separate from synchronized inference time.
   Optional warmup is excluded. Results are saved after each completed image.
4. Inspect side-by-side overlays and individual PNG masks. Failures have traceback
   files and explicit statuses; a failed method is not counted as a zero-detection
   success. `run_status.json` reports whether five actually completed all inputs.
5. Read detector confidence and predicted mask IoU as different, uncalibrated
   scores. Florence's missing detector score stays null, not a fabricated 1.0.
6. To measure accuracy, set `GT_MANIFEST_PATH` to a JSON mapping image filenames
   to lists of reviewed full-frame binary instance-mask PNGs, relative to that JSON.
   Missing key = unlabelled; [] = explicitly verified empty scene. Mask matching
   computes one-to-one precision/recall/F1 at IoU 0.5 and matched mean IoU, not mAP.
   Count references never influence inference, thresholds or mask selection.
7. To enable SAM 3, obtain approved Hugging Face access, set a Colab Secret named
   `HF_TOKEN`, then `ENABLE_SAM3=True`. Failures remain visible; no model is silently substituted.

Full-frame inference is the default. `ROI_NORM` optionally applies the same
normalized workspace rectangle to every method. Masks are restored to original
coordinates; area filtering is relative to the processed region. Crop settings
and thresholds are recorded. Cropping can exclude clutter but cannot prove object
identity or recover invisible boundaries. Do not crop out reference objects and
then interpret missed-object recall as whole-scene performance.

The comparison ZIP contains settings, per-model provenance, reports, overlays,
input copies, raw mask arrays, selected PNG masks and runtime source. Large
All weights use configurable local `MODEL_CACHE` / `SAM2_CACHE`, avoiding large
checkpoint reads from the Drive mount. They disappear when the Colab runtime is
reset; result folders still use the configured Drive location. Each SAM load validates
the checkpoint with `weights_only=True` before strict state loading. An unreadable
checkpoint triggers one fresh-cache download, preserving the previous file and
reusing the SAM recovery cache for subsequent methods. Detector safetensors header
errors trigger one retry in a separate fresh cache, preserving the previous files.
Processor and model use the same retry cache and requested revision, and provenance
records the successful cache. Other errors are not retried as weight corruption.
Full errors are printed and saved; safetensors loading is explicit and no unsafe
pickle fallback is used. This is not a production robot model bundle.
The supplied photos are not established as an openly licensed dataset.

## Recommendation before results exist

Try Grounding DINO + SAM and OWLv2 + SAM first as the new localization strategies;
include Florence as a different generative proposal mechanism. Compare all five
before choosing a winner. If they still fail to separate whole brackets, use
reviewed masks to fine-tune an instance model instead of treating count-matching
or threshold changes as proof of success. This remains a development comparison;
independent real captures are needed for deployment validation.
