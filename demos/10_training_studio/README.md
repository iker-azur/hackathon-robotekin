# Robotekin Bin-Picking Training Studio

Desktop studio for training and evaluating a LeRobot ACT imitation-learning
policy for the SO-101 bin-picking task. The visual identity uses Rubinnova and
Robotekin Hackathon branding.

## Sections

- **Overview:** dataset and checkpoint counts plus the ACT workflow.
- **Datasets:** discovers local datasets, lists repositories from the `MikelRubi` Hugging Face account, downloads a selected LeRobot dataset, and lets you select task groups and individual episodes for training.
- **Preparation:** links demonstration collection, dataset validation and ACT training.
- **Training:** selects RGB plus target mask, RGB-only or mask-only observations, optional joint-state input, ACT VAE, fine-tuning, episode split, vision backbone, transformer/VAE architecture, optimizer regularization, mixed precision, learning rate, batch size, action chunk and checkpoint/log intervals.
- **Training Dashboard:** shows live ACT metrics and a collapsible training-loss plot.
- **Results:** lists ACT checkpoints and full sorting evaluation reports.
- **Bin Picking Metrics:** reads segmentation, offline action and end-to-end reports; the selected report chart is collapsible.

The studio launches `demos/09_voice_bin_picking/tools/train_act.py` for training,
`offline_eval_policy.py` for action-error evaluation on held-out episodes, and
`eval_sorting.py` for complete sorting evaluation in Isaac Sim. Automatic and
manual episode splits are saved in `studio_run_config.json` after a successful
training run. ACT training logs do not include in-loop validation, so held-out
action and task metrics are produced by the evaluation commands after training.

## Install and launch on Windows

From the workspace folder containing `hackathon-robotekin`:

```powershell
uv pip install --link-mode=copy --python .\hackathon-robotekin\.venv-so101\Scripts\python.exe -r .\hackathon-robotekin\demos\10_training_studio\requirements.txt
& .\hackathon-robotekin\.venv-so101\Scripts\python.exe .\hackathon-robotekin\demos\10_training_studio\app.py
```

In **Training Python**, select the bin-picking environment with LeRobot 0.4.4.
Full sorting evaluation also needs Isaac Sim and `recoverykit` dependencies.
In **Datasets**, choose **Refresh Hub list**, check the repositories you want,
and choose **Download checked**. Downloads are stored under
`demos/09_voice_bin_picking/data` and checked for LeRobot episode metadata,
actions and camera images before the studio makes them available to ACT.
Use the **Pick-cube** and **Up-down** quick selectors to check repositories with
matching names; individual repository checkboxes remain available afterward.
When task metadata is present, the dataset details list its task groups and
episode counts. Checking or unchecking a group updates the episode checklist;
you can then choose individual episodes. All selected episodes feed the
train/validation/test split. Choose RGB, target mask or both in **ACT
observation inputs**. Episodes keep their synchronized images, actions and
states together.
Public datasets can be downloaded directly. For private datasets, sign in on
this computer first with `hf auth login`; the studio uses the Hugging Face
credentials already stored locally and never asks for a token in the app.

## Workflow

1. Collect demonstrations with `demos/09_voice_bin_picking/tools/collect_demos.py`.
2. Validate them with `tools/validate_dataset.py`.
3. Choose a dataset, observation mode, held-out split and new output name.
4. Choose **Pick-up** or **Up-down** in **Experiment output**, then train ACT. Checkpoints are saved under `demos/09_voice_bin_picking/outputs/<task>/<run name>`.
5. Run held-out action evaluation and full sorting evaluation from **Training Runs**.

The first-run baseline is 10,000 steps, batch 8, chunk size 50, 10 actions per
prediction, learning rate `1e-5`, RGB plus target-mask input, joint state enabled
and VAE enabled. It uses an 80/20 episode split, pretrained ResNet18, transformer size 512,
8 attention heads, 4 encoder layers, 1 decoder layer, and dropout 0.1. Adjust
batch size and backbone size for the available GPU memory. Temporal ensembling
is optional and requires one action per prediction.

Other initial values are weight decay `1e-4`, backbone learning rate `1e-5`,
dropout `0.1`, KL weight `10`, feed-forward size `3200`, VAE latent size `32`,
4 VAE encoder layers, 4 data-loader workers, checkpoint every 2,500 steps and
logs every 200 steps. Mixed precision, pre-norm, dilation and temporal
ensembling start disabled. Setup parameter controls are keyboard-only to avoid
accidental mouse-wheel, spinner or dropdown changes; dataset and output-folder
selection remain mouse-accessible.

The advanced ACT controls are passed into LeRobot's `ACTConfig` and recorded in
the run metadata. The trainer checks whether the selected LeRobot version
supports each non-default architecture setting and stops with a clear message
instead of silently ignoring unsupported values. The former Tiling Preview and
Augmentation UI code is preserved as a recovery reference in
`archive/tiling_augmentation_sections.py.txt`; the active ACT pipeline does not
apply those preview or image-transform controls.
