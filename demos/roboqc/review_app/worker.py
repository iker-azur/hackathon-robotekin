"""Explicit user-started training/validation subprocess. Not run on app startup."""

import argparse
import json
from pathlib import Path
import sys


def event(**values):
    print('ROBOQC_EVENT ' + json.dumps(values, default=str), flush=True)


# Augmentation keys forwarded to Ultralytics, with the values used when a job file omits them.
# They equal the former hard-coded values (hue/saturation off, any rotation, flips, light mosaic)
# plus Ultralytics' own defaults for the rest, so older job files train exactly as before.
AUGMENTATION_DEFAULTS = dict(
    fliplr=.5, flipud=.5, degrees=180, translate=.1, scale=.5, shear=0, perspective=0,
    hsv_h=0, hsv_s=0, hsv_v=.2, mosaic=.2, mixup=0, cutmix=0, copy_paste=0, copy_paste_mode='flip', close_mosaic=10,
)


def augmentation_kwargs(config):
    """Augmentation arguments for model.train(); accepts the old 'brightness' key for hsv_v."""
    values = {key: config.get(key, default) for key, default in AUGMENTATION_DEFAULTS.items()}
    if 'hsv_v' not in config and 'brightness' in config:
        values['hsv_v'] = config['brightness']
    values['close_mosaic'] = min(int(values['close_mosaic']), int(config.get('epochs', values['close_mosaic'])))
    return values


def run(config):
    import torch
    import yaml
    from ultralytics import YOLO, YOLOE
    dataset = Path(config['dataset']).resolve()
    output = Path(config['output']).resolve()
    output.mkdir(parents=True, exist_ok=False)
    data = yaml.safe_load((dataset / 'data.yaml').read_text(encoding='utf-8'))
    data['path'] = str(dataset)
    resolved = output / 'data_resolved.yaml'
    resolved.write_text(yaml.safe_dump(data), encoding='utf-8')
    (output / 'job.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    device = config['device']
    if device == 'auto':
        device = 0 if torch.cuda.is_available() else 'cpu'
    event(status='loading', device=device, output=output)
    model_class = YOLOE if config['family'] == 'YOLOE' else YOLO
    model = model_class(config['weights'])
    if config['action'] == 'train':
        kwargs = {}
        if config['family'] == 'YOLOE':
            from ultralytics.models.yolo.yoloe import YOLOEPESegTrainer
            kwargs['trainer'] = YOLOEPESegTrainer
        def epoch_finished(trainer):
            metrics = {str(k): float(v) for k, v in trainer.metrics.items()}
            event(status='epoch', epoch=trainer.epoch + 1, epochs=trainer.epochs, metrics=metrics)
        model.add_callback('on_fit_epoch_end', epoch_finished)
        model.train(data=str(resolved), project=str(output), name='train', exist_ok=False,
                    epochs=config['epochs'], imgsz=config['imgsz'], batch=config['batch'],
                    patience=config['patience'], device=device, seed=config['seed'], workers=0,
                    overlap_mask=False, **augmentation_kwargs(config), **kwargs)
        event(status='finished', weights=output / 'train/weights/best.pt')
    else:
        split = config['split']
        if split not in ('val', 'test') or not list((dataset / 'images' / split).glob('*.png')):
            raise ValueError('Selected held-out split is empty')
        metrics = model.val(data=str(resolved), split=split, project=str(output), name='validation',
                            device=device, imgsz=config['imgsz'], batch=config['batch'],
                            workers=0, plots=True, save_json=True)
        report = {'split': split, 'metrics': {k: float(v) for k, v in metrics.results_dict.items()},
                  'speed_ms_per_image': metrics.speed, 'weights': config['weights']}
        (output / 'validation.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
        event(status='finished', report=report)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config')
    arguments = parser.parse_args()
    try:
        run(json.loads(Path(arguments.config).read_text(encoding='utf-8')))
    except Exception:
        import traceback
        traceback.print_exc()
        event(status='failed')
        sys.exit(1)
