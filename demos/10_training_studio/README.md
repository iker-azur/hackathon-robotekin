# Robotekin Bin-Picking Training Studio

Desktop studio for training and evaluating a LeRobot ACT imitation-learning
policy for the SO-101 bin-picking task. The visual identity uses Rubinnova and
Robotekin Hackathon branding.

## Sections

- **Overview:** dataset and checkpoint counts plus the ACT workflow.
- **Datasets:** discovers local datasets, lists repositories from the `MikelRubi` Hugging Face account, downloads a selected LeRobot dataset, and lets you select task groups and individual episodes for training.
- **Preparation:** links demonstration collection, dataset validation and ACT training.
- **Training:** selects RGB plus target mask, RGB-only or mask-only observations, optional joint-state input, ACT VAE, fine-tuning, episode split, vision backbone, transformer/VAE architecture, optimizer regularization, mixed precision, learning rate, batch size, action chunk and checkpoint/log intervals.
- **Training Dashboard:** runs Validate, Prepare and Train separately or together; shows stage status, step progress, train/validation ACT loss, action L1, VAE KL, gradient norm, learning rate, throughput and GPU memory.
- **Results:** lists ACT checkpoints and full sorting evaluation reports.
- **Bin Picking Metrics:** reads segmentation, offline action and end-to-end reports; the selected report chart is collapsible.
- **Inspección de pernos:** independent supervised front/wrist classifier with local download destination, labelled snapshot download from `MikelRubi`, session selection, pair previews, grouped validation, live training metrics and two-image inference. See [the pin inspection guide](../10_pin_inspection/README.md) for its separate training dependencies and outputs.

The studio launches `demos/09_voice_bin_picking/tools/train_act.py` for training,
`offline_eval_policy.py` for action-error evaluation on held-out episodes, and
`eval_sorting.py` for complete sorting evaluation in Isaac Sim. Automatic and
manual episode splits are saved in `studio_run_config.json` after a successful
training run. During ACT training, the dashboard reports validation loss and
action L1 on a fixed sample of up to 64 frames from held-out episodes at the
logging interval. At least one validation episode must be held out; post-training
offline and full sorting evaluations remain available for broader checks.

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
Use the **Pick-cube**, **Up-down Good** and **Up-down Bad** quick selectors to check
repositories with matching names; individual repository checkboxes remain available
afterward. Up-down Bad matches names containing `bad` (`up-down-bad_<date>`); Up-down
Good matches the remaining `up-down_<date>` recordings, so each branch can be
downloaded and trained independently.
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
3. Choose a dataset, observation mode, held-out split and new output name. Keep at least one episode in the validation split.
4. Choose **Pick-up** or **Up-down** in **Experiment output**, then run the stages from **Training Dashboard**. Pick-up checkpoints are saved under `runs/Pick-up/<run name>`; Up-down checkpoints are grouped under `runs/Up-down/Good/<run name>` or `runs/Up-down/Bad/<run name>`. The Up-down branch is inferred from a Good/Bad token in the selected dataset name and can be changed in Experiment output.
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
