"""Augmentation page: parameters, live 3x3 preview on real reviewed images, offline export."""

from pathlib import Path
import uuid

import numpy as np
from PySide6.QtCore import QTimer, Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QGridLayout, QHBoxLayout, QLabel, QSizePolicy,
    QVBoxLayout, QWidget,
)

from . import augment
from .core import zip_dataset
from .page_jobs import path_row, spin
from .theme import GAP, hex_to_rgb as rgb_of
from .widgets import ActionButton, Card, ControlColumn, FormGrid, Page

PREVIEW_SIDE = 720
CUSTOM = 'Custom'


class PreviewCell(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        self.image = QLabel()
        self.image.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image.setMinimumSize(96, 64)
        self.image.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        self.caption = QLabel('')
        self.caption.setProperty('muted', True)
        self.caption.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.caption.setFixedHeight(18)
        layout.addWidget(self.image, 1)
        layout.addWidget(self.caption)
        self.source = None

    def set_sample(self, rgb, caption):
        rgb = np.ascontiguousarray(rgb)
        h, w = rgb.shape[:2]
        self.source = QImage(rgb.data, w, h, w * 3, QImage.Format.Format_RGB888).copy()
        self.caption.setText(caption)
        self.rescale()

    def rescale(self):
        if self.source is not None:
            self.image.setPixmap(QPixmap.fromImage(self.source).scaled(
                self.image.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.rescale()


class AugmentPage(Page):
    def __init__(self, win):
        super().__init__('Augmentation', 'Tune how training images are varied, preview it on your own reviewed masks, '
                                         'and optionally export an augmented copy of the dataset.')
        self.win = win
        self.seed = 1
        self.source = None
        self.widgets = {}
        self._loading = False
        body = QHBoxLayout()
        body.setSpacing(16)
        self.column = ControlColumn()
        body.addWidget(self.column)
        body.addLayout(self._build_preview(), 1)
        self.root.addLayout(body, 1)
        self._build_controls()
        self._build_export()
        self.timer = QTimer(self)
        self.timer.setSingleShot(True)
        self.timer.timeout.connect(self.render)
        self.select_preset('Metal parts (default)')

    # -- construction -------------------------------------------------------------------

    def _build_controls(self):
        presets = Card('Preset')
        row = QHBoxLayout()
        row.setSpacing(GAP)
        self.presets = QComboBox()
        self.presets.addItems(list(augment.PRESETS) + [CUSTOM])
        self.presets.activated.connect(lambda index: self.select_preset(self.presets.itemText(index)))
        row.addWidget(self.presets, 1)
        reset = ActionButton('Reset', None)
        reset.clicked.connect(lambda: self.select_preset('Metal parts (default)'))
        row.addWidget(reset)
        presets.add_layout(row)
        note = QLabel('Defaults keep hue and saturation fixed (the colour of a part carries information) and allow any '
                      'rotation (parts are photographed from above).')
        note.setProperty('muted', True)
        note.setWordWrap(True)
        presets.add(note)
        self.column.add(presets)
        for group in augment.GROUPS:
            card = Card(group)
            form = FormGrid()
            for field in augment.FIELDS:
                if field['group'] != group:
                    continue
                widget = self._make_widget(field)
                label = field['label'] + ('' if field['scope'] == 'train' else '  (preview / offline)')
                form.add_row(label, widget, field['tip'])
                self.widgets[field['key']] = widget
            card.add_layout(form)
            if group == 'Composition':
                note = QLabel('Mosaic, mixup and CutMix run inside Ultralytics while training, so the preview does not '
                              'show them. Copy-paste uses your reviewed masks.')
                note.setProperty('muted', True)
                note.setWordWrap(True)
                card.add(note)
            self.column.add(card)

    def _make_widget(self, field):
        if field['kind'] == 'choice':
            widget = QComboBox()
            widget.addItems(list(field['min']))
            widget.currentTextChanged.connect(self.on_changed)
        elif field['kind'] == 'int':
            widget = spin(field['default'], int(field['min']), int(field['max']))
            widget.valueChanged.connect(self.on_changed)
        else:
            widget = QDoubleSpinBox()
            widget.setDecimals(4 if field['step'] < .01 else 2)
            widget.setRange(field['min'], field['max'])
            widget.setSingleStep(field['step'])
            widget.setValue(field['default'])
            widget.valueChanged.connect(self.on_changed)
        return widget

    def _build_export(self):
        card = Card('Offline export', 'Writes a copy of a reviewed dataset with extra augmented training images. '
                                      'Validation and test files are copied untouched.')
        form = FormGrid()
        self.export_dataset = path_row(form, 'Reviewed dataset', folder=True)
        self.copies = spin(3, 1, 50)
        form.add_row('Copies per training image', self.copies)
        self.max_side = spin(1280, 256, 4096)
        self.max_side.setSuffix(' px')
        form.add_row('Longest side of copies', self.max_side, 'Large photos are shrunk to keep the dataset light.')
        self.export_seed = spin(42, 0, 2147483647)
        form.add_row('Seed', self.export_seed)
        card.add_layout(form)
        self.zip_check = QCheckBox('Also create a ZIP next to the folder')
        self.zip_check.setChecked(True)
        card.add(self.zip_check)
        button = ActionButton('Export augmented dataset', None, role='primary')
        button.clicked.connect(self.export)
        card.add(button)
        self.column.add(card)

    def _build_preview(self):
        column = QVBoxLayout()
        column.setSpacing(16)
        card = Card('Preview')
        row = QHBoxLayout()
        row.setSpacing(GAP)
        row.addWidget(QLabel('Source'))
        self.source_combo = QComboBox()
        self.source_combo.setMinimumWidth(320)
        self.source_combo.currentIndexChanged.connect(self.load_source)
        row.addWidget(self.source_combo, 1)
        self.reroll_button = ActionButton('Re-roll', 'reroll_preview')
        self.reroll_button.clicked.connect(self.reroll)
        row.addWidget(self.reroll_button)
        card.add_layout(row)
        self.status = QLabel('Open a bundle, accept an image in Review, then pick it here.')
        self.status.setProperty('muted', True)
        self.status.setWordWrap(True)
        card.add(self.status)
        grid = QGridLayout()
        grid.setSpacing(12)
        self.cells = []
        for index in range(9):
            cell = PreviewCell()
            self.cells.append(cell)
            grid.addWidget(cell, index // 3, index % 3)
            grid.setRowStretch(index // 3, 1)
            grid.setColumnStretch(index % 3, 1)
        card.add_layout(grid, 1)
        column.addWidget(card, 1)
        return column

    # -- configuration -----------------------------------------------------------------

    def config(self):
        values = {}
        for key, widget in self.widgets.items():
            values[key] = widget.currentText() if isinstance(widget, QComboBox) else widget.value()
        return augment.clean(values)

    def set_config(self, config):
        config = augment.clean(config)
        self._loading = True
        for key, widget in self.widgets.items():
            if isinstance(widget, QComboBox):
                widget.setCurrentText(config[key])
            else:
                widget.setValue(config[key])
        self._loading = False
        self.sync_preset_name()
        self.schedule()

    def select_preset(self, name):
        if name != CUSTOM:
            self.set_config(augment.preset(name))
            self.save()

    def sync_preset_name(self):
        config = self.config()
        name = next((n for n in augment.PRESETS if augment.preset(n) == config), CUSTOM)
        self.presets.setCurrentText(name)
        training = getattr(self.win, 'training', None)
        if training is not None:
            training.refresh_summary()

    def on_changed(self, *args):
        if self._loading:
            return
        self.sync_preset_name()
        self.save()
        self.schedule()

    def save(self):
        if self.win.project is not None:
            self.win.project.set_pref('augmentation', self.config())

    def load_project(self):
        stored = self.win.project.pref('augmentation')
        if stored:
            self.set_config(stored)
        self.refresh_sources()

    def summary(self):
        name = self.presets.currentText()
        config = self.config()
        used = f"flip {config['fliplr']:.2f}/{config['flipud']:.2f}, rotate ±{config['degrees']:.0f}°, scale ±{config['scale']:.2f}, " \
               f"brightness {config['hsv_v']:.2f}, mosaic {config['mosaic']:.2f}, copy-paste {config['copy_paste']:.2f}"
        return f'<b>{name}</b><br>{used}.<br>Edit them in the Augmentation tab; they are passed to Ultralytics when training starts.'

    # -- preview -----------------------------------------------------------------------

    def refresh_sources(self):
        self.source_combo.blockSignals(True)
        self.source_combo.clear()
        self.source_combo.addItem('Current image in Review', 'current')
        project, bundle = self.win.project, self.win.bundle
        if project is not None:
            for image_id, review in project.state['reviews'].items():
                if review['status'] == 'accepted' and review['instances']:
                    self.source_combo.addItem(bundle.by_image[image_id]['source_relative_path'], image_id)
        self.source_combo.blockSignals(False)
        self.load_source()

    def load_source(self, *args):
        data = self.source_combo.currentData()
        if data is None:
            return
        if data == 'current':
            review = self.win.review
            if review.rgb is None:
                self.source = None
                self.status.setText('Open an image in Review, or accept one and choose it here.')
                self.clear_cells()
                return
            masks = [i['mask'] for i in review.doc.instances if i.get('keep', True)]
            labels = [i['label'] for i in review.doc.instances if i.get('keep', True)]
            self.set_source(review.rgb, masks, labels)
            return
        bundle, project = self.win.bundle, self.win.project
        def load(progress):
            return bundle.rgb(data), project.load_masks(data)
        def loaded(result):
            rgb, instances = result
            self.set_source(rgb, [i['mask'] for i in instances], [i['label'] for i in instances])
        self.win.background(load, loaded)

    def set_source(self, rgb, masks, labels):
        rgb, masks = augment.fit_max_side(np.ascontiguousarray(rgb), [np.asarray(m, bool) for m in masks], PREVIEW_SIDE)
        self.source = (rgb, masks, list(labels))
        self.status.setText(f'{len(masks)} masks from the source image. Each cell is a random sample; '
                            'dropped = masks that left the frame or became tiny.')
        self.render()

    def clear_cells(self):
        for cell in self.cells:
            cell.image.clear()
            cell.source = None
            cell.caption.setText('')

    def schedule(self):
        self.timer.start(150)

    def reroll(self):
        self.seed += 1
        self.render()

    def render(self):
        if self.source is None:
            return
        rgb, masks, labels = self.source
        config = self.config()
        project = self.win.project
        for index, cell in enumerate(self.cells):
            rng = np.random.default_rng((self.seed, index))
            sample = augment.augment_sample(rgb, masks, labels, config, rng)
            colors = [rgb_of(project.color_of(label)) if project else (37, 99, 235) for label in sample['labels']]
            picture = augment.draw_preview(sample['image'], sample['masks'], colors or [(37, 99, 235)])
            cell.set_sample(picture, f"{len(sample['masks'])} masks" + (f" · {sample['dropped']} dropped" if sample['dropped'] else ''))

    # -- offline export ----------------------------------------------------------------

    def export(self):
        dataset = self.export_dataset.text().strip()
        if not dataset:
            self.win.error('Choose the reviewed dataset folder to augment.')
            return
        start = (self.win.project.pref('augment_dir') if self.win.project else '') or ''
        parent = QFileDialog.getExistingDirectory(self.win, 'Output parent (a new subfolder is created)', start)
        if not parent:
            return
        if self.win.project:
            self.win.project.set_pref('augment_dir', parent)
        destination = Path(parent) / ('augmented_dataset_' + uuid.uuid4().hex[:12])
        config, copies, seed = self.config(), self.copies.value(), self.export_seed.value()
        max_side, make_zip = self.max_side.value(), self.zip_check.isChecked()
        def run(progress):
            result = augment.export_augmented(dataset, destination, copies, config, seed, max_side, progress)
            if make_zip:
                zip_dataset(destination, destination.with_suffix('.zip'))
            return result
        def done(result):
            self.win.training.dataset.setText(str(destination))
            self.win.validation.dataset.setText(str(destination))
            self.win.info('Augmented dataset exported',
                          f"{destination}\n\n{result['added']} augmented images added to {result['train_images']} "
                          f"training images ({result['skipped']} skipped because every mask left the frame)."
                          + ('\nA ZIP was created next to the folder.' if make_zip else ''))
        self.win.background(run, done)
