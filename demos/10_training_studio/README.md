# Robotekin Bin-Picking Training Studio

Desktop studio for training and evaluating a LeRobot ACT imitation-learning
policy for the SO-101 bin-picking task. The visual identity uses Rubinnova and
Robotekin Hackathon branding.

## Sections

- **Overview:** dataset and checkpoint counts plus the ACT workflow.
- **Datasets:** discovers local datasets, lists repositories from the `MikelRubi` Hugging Face account, and downloads a selected LeRobot dataset for training.
- **Preparation:** links demonstration collection, dataset validation and ACT training.
- **Tiling Preview:** previews camera crops only; ACT receives full observations.
- **Augmentation:** previews image changes; online image augmentation is disabled for this trainer.
- **Training:** selects RGB plus target mask, RGB-only or mask-only observations, optional joint-state input, ACT VAE, fine-tuning, episode split, learning rate, batch size, action chunk and checkpoint/log intervals.
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
In **Datasets**, choose **Refresh Hub list**, select a repository and choose
**Download selected**. Downloads are stored under
`demos/09_voice_bin_picking/data` and checked for LeRobot episode metadata,
actions and camera images before the studio makes them available to ACT.
Public datasets can be downloaded directly. For private datasets, sign in on
this computer first with `hf auth login`; the studio uses the Hugging Face
credentials already stored locally and never asks for a token in the app.

## Workflow

1. Collect demonstrations with `demos/09_voice_bin_picking/tools/collect_demos.py`.
2. Validate them with `tools/validate_dataset.py`.
3. Choose a dataset, observation mode, held-out split and new output name.
4. Train ACT. Checkpoints are saved under `demos/09_voice_bin_picking/outputs/<run name>`.
5. Run held-out action evaluation and full sorting evaluation from **Training Runs**.

The ACT defaults are 50,000 steps, batch 16, chunk size 50, 25 actions per
prediction, learning rate `1e-5`, RGB plus target-mask input, joint state enabled
and VAE enabled. Adjust batch size for the available GPU memory.
