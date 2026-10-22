# RoboQC · Robotekin 2026

Start with the [Colab notebooks and integration guide](colab/README.md).
The workflow trains voice understanding first. For vision without an annotated
dataset, start with [04a: pretrained YOLOE](colab/04a_try_yoloe_pretrained.ipynb),
then notebook 06 in unlabelled mode; skip 04 and 05 until custom training is needed.
For class-agnostic segmentation of the four supplied stamped-part photos, use
[04b: SAM 2.1](colab/04b_try_sam2_stamped_parts.ipynb) independently; its image inputs
are supplied by upload/Drive; its candidate masks are not yet a classified robot detector.
Use [04c: pretrained comparison](colab/04c_compare_pretrained_segmentation.ipynb)
to compare five methods on the actual `data` photos, with optional SAM 3 as a sixth.
With approved `facebook/sam3` access, use
[04d: SAM 3 only](colab/04d_try_sam3_pretrained.ipynb) on a fresh CPU or GPU runtime;
upload your photos or use a Drive path; gated weights stay on local Colab disk.
Run notebooks manually in Google Colab; notebook execution is not part of local tests.

After 04e export, use the [PySide6 review and training workstation](review_app/README.md)
to select/edit instance masks, export approved datasets, and explicitly start
YOLOE/YOLO training and held-out validation. Source proposal archives stay read-only.

Existing demos remain in [phase_1](phase_1/README.md) and
[phase_2](phase_2/README.md). Industrial artifacts use new folders and versioned
manifests, leaving existing datasets and trained weights intact.
