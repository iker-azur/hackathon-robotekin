"""Four-page Qt workstation. Bundle code and model runtime stay isolated."""

from collections import OrderedDict
import csv
import json
from pathlib import Path
import sys
import uuid

import cv2
import numpy as np
from PySide6.QtCore import QObject, QPointF, QProcess, Qt, QThread, QTimer, Signal
from PySide6.QtGui import QColor, QDesktopServices, QImage, QPainter, QPen, QPixmap
from PySide6.QtCore import QUrl
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QFormLayout,
    QGraphicsScene, QGraphicsView, QHBoxLayout, QHeaderView, QLabel, QLineEdit,
    QListWidget, QListWidgetItem, QMainWindow, QMessageBox, QPlainTextEdit,
    QProgressBar, QPushButton, QSpinBox, QSplitter, QTabWidget, QTableWidget,
    QTableWidgetItem, QVBoxLayout, QWidget,
)

from .core import Bundle, Project, export_dataset, write_json, zip_dataset


def button(text, callback, layout):
    widget = QPushButton(text)
    widget.clicked.connect(callback)
    layout.addWidget(widget)
    return widget


def spin(value, low, high, decimal=False):
    widget = QDoubleSpinBox() if decimal else QSpinBox()
    widget.setRange(low, high)
    widget.setValue(value)
    return widget


def pixmap(rgb):
    rgb = np.ascontiguousarray(rgb, np.uint8)
    return QPixmap.fromImage(QImage(rgb.data, rgb.shape[1], rgb.shape[0], rgb.strides[0],
                                   QImage.Format.Format_RGB888).copy())


class Task(QObject):
    done = Signal(object)
    failed = Signal(str)
    progress = Signal(int, int)

    def __init__(self, function):
        super().__init__()
        self.function = function

    def run(self):
        try:
            self.done.emit(self.function(self.progress.emit))
        except Exception as exc:
            self.failed.emit(f'{type(exc).__name__}: {exc}')


class Canvas(QGraphicsView):
    changed = Signal()

    def __init__(self):
        super().__init__()
        self.setScene(QGraphicsScene(self))
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.rgb = None
        self.instances = []
        self.active = -1
        self.tool = 'Pan'
        self.radius = 12
        self.previous = None
        self.setMinimumSize(480, 360)

    def display(self, rgb, instances):
        self.rgb, self.instances = rgb, instances
        self.render_masks()
        self.fitInView(self.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)

    def render_masks(self):
        if self.rgb is None:
            return
        result = self.rgb.copy()
        for index, row in enumerate(self.instances):
            if not row.get('keep', True):
                continue
            mask = row['mask']
            color = np.array([(index * 83 + 70) % 210 + 30, (index * 47 + 90) % 210 + 30,
                              (index * 127 + 40) % 210 + 30])
            result[mask] = (.60 * result[mask] + .40 * color).astype(np.uint8)
            contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(result, contours, -1, (255, 255, 0) if index == self.active else tuple(map(int, color)),
                             3 if index == self.active else 1)
            ys, xs = np.nonzero(mask)
            if len(xs):
                cv2.putText(result, str(index + 1), (int(xs.mean()), int(ys.mean())),
                            cv2.FONT_HERSHEY_SIMPLEX, .6, (255, 20, 20), 2)
        self.scene().clear()
        self.scene().addPixmap(pixmap(result))
        self.scene().setSceneRect(0, 0, result.shape[1], result.shape[0])

    def wheelEvent(self, event):
        self.scale(1.2 if event.angleDelta().y() > 0 else 1 / 1.2,
                   1.2 if event.angleDelta().y() > 0 else 1 / 1.2)

    def paint_at(self, event):
        if not 0 <= self.active < len(self.instances):
            return
        point = self.mapToScene(event.position().toPoint())
        x, y = int(point.x()), int(point.y())
        mask = self.instances[self.active]['mask']
        if not (0 <= x < mask.shape[1] and 0 <= y < mask.shape[0]):
            self.previous = None
            return
        value = 0 if self.tool == 'Erase' else 1
        raster = mask.astype(np.uint8)
        if self.previous is not None:
            cv2.line(raster, self.previous, (x, y), value, self.radius * 2)
        cv2.circle(raster, (x, y), self.radius, value, -1)
        mask[:] = raster > 0
        self.previous = (x, y)
        self.changed.emit()
        self.render_masks()

    def mousePressEvent(self, event):
        if self.tool != 'Pan' and event.button() == Qt.MouseButton.LeftButton:
            self.previous = None
            self.paint_at(event)
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self.tool != 'Pan' and event.buttons() & Qt.MouseButton.LeftButton:
            self.paint_at(event)
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        self.previous = None
        super().mouseReleaseEvent(event)


class MetricPlot(QWidget):
    def __init__(self):
        super().__init__()
        self.rows = []
        self.setMinimumHeight(180)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor('#16202c'))
        painter.setPen(QColor('#dde5ef'))
        painter.drawText(12, 20, 'Mask mAP50-95 (green), box mAP50-95 (blue) — live results.csv')
        for key, color in [('metrics/mAP50-95(M)', '#48d3a2'), ('metrics/mAP50-95(B)', '#5ea8ff')]:
            values = []
            for row in self.rows:
                try:
                    values.append(float(row[key]))
                except (KeyError, ValueError):
                    pass
            painter.setPen(QPen(QColor(color), 2))
            for i in range(1, len(values)):
                painter.drawLine(QPointF(20 + (i - 1) * (self.width()-40) / max(1, len(values)-1),
                                        self.height()-20-values[i-1]*(self.height()-55)),
                                 QPointF(20 + i*(self.width()-40)/max(1, len(values)-1),
                                         self.height()-20-values[i]*(self.height()-55)))


class Window(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Robotekin | Segmentation Review & Training')
        self.resize(1440, 920)
        self.bundle = self.project = None
        self.current_id = self.current_proposal = None
        self.displayed_proposal_index = 0
        self.instances = []
        self.dirty = False
        self.cache = OrderedDict()
        self.process = None
        self.task_thread = None
        self.job_output = None
        self.job_buffer = ''
        self.load_initial_proposal = False
        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)
        self.build_explorer()
        self.build_viewer()
        self.build_training()
        self.build_validation()
        self.busy = QProgressBar()
        self.busy.setMaximumWidth(350)
        self.statusBar().addPermanentWidget(self.busy)
        self.busy.hide()
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.poll_metrics)
        self.timer.start(2000)
        self.setStyleSheet('QWidget {font-family: "Segoe UI"; font-size: 12px;} '
                           'QPushButton {padding: 7px;} QTabBar::tab {padding: 12px 25px;}')

    def page(self, name):
        page = QWidget()
        layout = QVBoxLayout(page)
        self.tabs.addTab(page, name)
        return layout

    def error(self, message):
        QMessageBox.warning(self, 'Review workstation', str(message))

    def background(self, function, finished, failed=None):
        if self.task_thread is not None:
            self.error('Wait for the current file operation to finish')
            return
        thread = QThread(self)
        task = Task(function)
        task.moveToThread(thread)
        thread.started.connect(task.run)
        task.progress.connect(self.progress)
        task.done.connect(finished)
        task.failed.connect(failed or self.error)
        task.done.connect(thread.quit)
        task.failed.connect(thread.quit)
        task.done.connect(task.deleteLater)
        task.failed.connect(task.deleteLater)
        thread.finished.connect(self.task_finished)
        thread.finished.connect(thread.deleteLater)
        self.task_thread, self.task = thread, task
        self.tabs.setEnabled(False)
        self.busy.setRange(0, 0)
        self.busy.show()
        thread.start()

    def task_finished(self):
        self.task_thread = self.task = None
        self.tabs.setEnabled(True)
        self.busy.hide()
        if self.load_initial_proposal:
            self.load_initial_proposal = False
            QTimer.singleShot(0, self.change_proposal)

    def progress(self, done, total):
        self.busy.setRange(0, total)
        self.busy.setValue(done)

    def build_explorer(self):
        layout = self.page('1  Explorer')
        row = QHBoxLayout()
        button('Open 04e ZIP', lambda: self.open_bundle(False), row)
        button('Open extracted bundle', lambda: self.open_bundle(True), row)
        button('Export approved dataset', self.export, row)
        self.zip_check = QCheckBox('Also create ZIP')
        self.zip_check.setChecked(True)
        row.addWidget(self.zip_check)
        self.holes_check = QCheckBox('Allow hole filling in YOLO labels (lossy)')
        row.addWidget(self.holes_check)
        layout.addLayout(row)
        self.source_label = QLabel('Open the results ZIP from Drive → Robotekin2026/exports. Not the inputs ZIP.')
        self.source_label.setWordWrap(True)
        layout.addWidget(self.source_label)
        self.search = QLineEdit()
        self.search.setPlaceholderText('Filter nested source path or review status…')
        self.search.textChanged.connect(self.filter_images)
        layout.addWidget(self.search)
        self.table = QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels(['Source path', 'Size', 'Alternatives OK/total', 'Candidate counts', 'Review', 'Selected instances'])
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.table.cellDoubleClicked.connect(lambda row, col: self.open_image(row))
        layout.addWidget(self.table)
        self.summary = QLabel('Double-click an image to review. No source files are deleted or modified.')
        layout.addWidget(self.summary)

    def discard_changes(self):
        return not self.dirty or QMessageBox.question(self, 'Unsaved mask edits',
            'Discard unsaved edits? Accept this image first to save them.') == QMessageBox.StandardButton.Yes

    def open_bundle(self, folder):
        if not self.discard_changes():
            return
        if folder:
            source = QFileDialog.getExistingDirectory(self, '04e bundle folder')
        else:
            source, _ = QFileDialog.getOpenFileName(self, '04e results ZIP', '', 'ZIP (*.zip)')
        if not source:
            return
        project_folder = QFileDialog.getExistingDirectory(self, 'Choose separate review project folder (empty or existing project)')
        if not project_folder:
            return
        def load(progress):
            bundle = Bundle(source)
            try:
                project = Project(project_folder, bundle)
            except Exception:
                bundle.close()
                raise
            return bundle, project
        self.background(load, self.bundle_loaded)

    def bundle_loaded(self, result):
        if self.bundle:
            self.bundle.close()
        self.bundle, self.project = result
        self.cache.clear()
        self.current_id = self.current_proposal = None
        self.instances = []
        self.dirty = False
        self.proposals.blockSignals(True)
        self.proposals.clear()
        self.proposals.blockSignals(False)
        self.mask_list.clear()
        self.canvas.scene().clear()
        self.canvas.rgb = None
        self.source_label.setText(f'Source (read-only): {self.bundle.source}\nReview project: {self.project.folder}')
        self.refresh_table()

    def refresh_table(self):
        self.table.setRowCount(len(self.bundle.images))
        for index, image in enumerate(self.bundle.images):
            alternatives = self.bundle.alternatives[image['image_id']]
            good = [r for r in alternatives if r['status'] == 'ok']
            decision = self.project.decision(image['image_id']) or {}
            values = [image['source_relative_path'], f"{image.get('width', '?')} × {image.get('height', '?')}",
                      f'{len(good)}/{len(alternatives)}', ', '.join(str(r.get('kept_masks', '?')) for r in good),
                      decision.get('status', 'unreviewed'), str(len(decision.get('instances', [])))]
            for column, value in enumerate(values):
                item = QTableWidgetItem(value)
                item.setData(Qt.ItemDataRole.UserRole, image['image_id'])
                self.table.setItem(index, column, item)
        reviewed = len(self.project.state['reviews'])
        self.summary.setText(f'{len(self.bundle.images)} images | {reviewed} reviewed | '
                             f'{len(self.bundle.images)-reviewed} remaining. Double-click to review.')
        self.filter_images()

    def filter_images(self):
        text = self.search.text().casefold()
        for row in range(self.table.rowCount()):
            self.table.setRowHidden(row, text not in ' '.join(self.table.item(row, col).text()
                                                            for col in (0, 4)).casefold())

    def build_viewer(self):
        layout = self.page('2  Review')
        top = QHBoxLayout()
        button('Previous', lambda: self.navigate(-1), top)
        button('Next', lambda: self.navigate(1), top)
        self.proposals = QComboBox()
        self.proposals.setMinimumWidth(320)
        self.proposals.currentIndexChanged.connect(self.change_proposal)
        top.addWidget(self.proposals, 1)
        self.tool = QComboBox()
        self.tool.addItems(['Pan', 'Paint', 'Erase'])
        self.tool.currentTextChanged.connect(lambda value: setattr(self.canvas, 'tool', value))
        top.addWidget(self.tool)
        self.brush = spin(12, 1, 100)
        self.brush.setPrefix('Brush px: ')
        self.brush.valueChanged.connect(lambda value: setattr(self.canvas, 'radius', value))
        top.addWidget(self.brush)
        button('Fit', lambda: self.canvas.fitInView(self.canvas.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio), top)
        layout.addLayout(top)
        split = QSplitter()
        self.canvas = Canvas()
        self.canvas.changed.connect(self.mark_dirty)
        split.addWidget(self.canvas)
        panel = QWidget()
        side = QVBoxLayout(panel)
        self.image_label = QLabel('Choose an image in Explorer')
        self.image_label.setWordWrap(True)
        side.addWidget(self.image_label)
        self.mask_list = QListWidget()
        self.mask_list.itemChanged.connect(self.mask_changed)
        self.mask_list.currentRowChanged.connect(self.active_mask)
        side.addWidget(self.mask_list)
        self.class_name = QLineEdit('part')
        self.class_name.setPlaceholderText('Class name for selected instance')
        side.addWidget(self.class_name)
        button('Assign class to active instance', self.assign_class, side)
        button('Add empty instance (paint it)', self.add_instance, side)
        self.group = QLineEdit()
        self.group.setPlaceholderText('Capture/session group — same scene stays in one split')
        self.group.textEdited.connect(self.mark_dirty)
        side.addWidget(self.group)
        self.negative = QCheckBox('Confirmed empty scene / no target parts')
        self.negative.toggled.connect(self.mark_dirty)
        side.addWidget(self.negative)
        button('Accept selected masks and save revision', self.accept, side)
        button('Exclude image from training (keep files)', self.exclude, side)
        side.addWidget(QLabel('Mouse wheel: zoom. Pan: drag.\nPaint/erase affects active instance only.\nUnchecked instances are not exported.\nChanging proposals discards unsaved edits.'))
        split.addWidget(panel)
        split.setSizes([1000, 360])
        layout.addWidget(split)

    def mark_dirty(self, *args):
        self.dirty = True

    def open_image(self, row):
        if not self.discard_changes():
            return
        image_id = self.table.item(row, 0).data(Qt.ItemDataRole.UserRole)
        if self.bundle.by_image[image_id]['status'] != 'ok':
            self.error('Invalid image in source manifest')
            return
        def load(progress):
            if image_id in self.cache:
                rgb = self.cache.pop(image_id)
            else:
                rgb = self.bundle.rgb(image_id)
            self.cache[image_id] = rgb
            while sum(value.nbytes for value in self.cache.values()) > 128 * 1024**2 and len(self.cache) > 1:
                self.cache.popitem(last=False)
            return image_id, rgb, self.project.load_masks(image_id)
        self.background(load, self.image_loaded)

    def image_loaded(self, result):
        image_id, self.rgb, saved = result
        self.current_id = image_id
        image = self.bundle.by_image[image_id]
        self.image_label.setText(image['source_relative_path'])
        decision = self.project.decision(image_id) or {}
        self.group.setText(decision.get('group', str(Path(image['source_relative_path']).parent)))
        self.negative.setChecked(decision.get('negative', False))
        self.proposals.blockSignals(True)
        self.proposals.clear()
        self.proposals.addItem('Saved reviewed masks' if saved or decision.get('negative') else 'Manual / empty', None)
        selected = 0
        for row in self.bundle.alternatives[image_id]:
            text = f"{row['method']} | {row.get('prompt') or 'combined'} | {row['status']} | {row.get('kept_masks', '?')} masks"
            self.proposals.addItem(text, row['proposal_id'])
            if not decision and selected == 0 and row['status'] == 'ok':
                selected = self.proposals.count() - 1
        self.proposals.setCurrentIndex(selected)
        self.proposals.blockSignals(False)
        self.instances = saved
        self.current_proposal = decision.get('proposal_id')
        self.dirty = False
        self.displayed_proposal_index = 0
        self.tabs.setCurrentIndex(1)
        self.show_instances()
        if selected:
            self.load_initial_proposal = True

    def change_proposal(self, *args):
        if self.current_id is None:
            return
        proposal = self.proposals.currentData()
        if not self.discard_changes():
            self.proposals.blockSignals(True)
            self.proposals.setCurrentIndex(self.displayed_proposal_index)
            self.proposals.blockSignals(False)
            return
        image_id = self.current_id
        requested_index = self.proposals.currentIndex()
        def load(progress):
            return self.bundle.candidates(proposal) if proposal else self.project.load_masks(image_id)
        def loaded(instances):
            self.instances = instances
            self.current_proposal = proposal if proposal else (self.project.decision(image_id) or {}).get('proposal_id')
            self.dirty = False
            self.displayed_proposal_index = requested_index
            self.show_instances()
        def failed(message):
            self.proposals.blockSignals(True)
            self.proposals.setCurrentIndex(self.displayed_proposal_index)
            self.proposals.blockSignals(False)
            self.error(message)
        self.background(load, loaded, failed)

    def show_instances(self):
        self.mask_list.blockSignals(True)
        self.mask_list.clear()
        for index, row in enumerate(self.instances):
            item = QListWidgetItem(f"{index+1}: {row['label']} | {int(row['mask'].sum())} px")
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Checked if row.get('keep', True) else Qt.CheckState.Unchecked)
            self.mask_list.addItem(item)
        self.mask_list.blockSignals(False)
        self.canvas.display(self.rgb, self.instances)
        self.mask_list.setCurrentRow(0 if self.instances else -1)

    def mask_changed(self, item):
        index = self.mask_list.row(item)
        self.instances[index]['keep'] = item.checkState() == Qt.CheckState.Checked
        self.dirty = True
        self.canvas.render_masks()

    def active_mask(self, index):
        self.canvas.active = index
        if 0 <= index < len(self.instances):
            self.class_name.setText(self.instances[index]['label'])
        self.canvas.render_masks()

    def assign_class(self):
        index = self.mask_list.currentRow()
        if index >= 0 and self.class_name.text().strip():
            self.instances[index]['label'] = self.class_name.text().strip()
            self.dirty = True
            self.show_instances()
            self.mask_list.setCurrentRow(index)

    def add_instance(self):
        if self.current_id:
            self.instances.append({'id': None, 'label': self.class_name.text().strip() or 'part',
                                   'mask': np.zeros(self.rgb.shape[:2], bool), 'keep': True})
            self.dirty = True
            self.show_instances()
            self.mask_list.setCurrentRow(len(self.instances)-1)
            self.tool.setCurrentText('Paint')

    def accept(self):
        self.save_review('accepted')

    def exclude(self):
        self.save_review('excluded')

    def save_review(self, status):
        if not self.current_id:
            return
        image_id, proposal, instances = self.current_id, self.current_proposal, self.instances
        group, negative = self.group.text(), self.negative.isChecked()
        def save(progress):
            return self.project.save(image_id, proposal, instances, status, group, negative)
        def saved(result):
            self.dirty = False
            self.refresh_table()
            self.statusBar().showMessage(f"Saved {status} revision {result['revision']}", 10000)
        self.background(save, saved)

    def navigate(self, step):
        if self.current_id:
            index = next(i for i, row in enumerate(self.bundle.images) if row['image_id'] == self.current_id)
            index += step
            while 0 <= index < self.table.rowCount():
                if not self.table.isRowHidden(index):
                    self.open_image(index)
                    break
                index += step

    def export(self):
        if not self.project:
            self.error('Open a bundle first')
            return
        if self.dirty:
            self.error('Save or discard current mask edits before exporting')
            return
        parent = QFileDialog.getExistingDirectory(self, 'Export parent folder (fresh dataset subfolder created)')
        if not parent:
            return
        output = Path(parent) / ('reviewed_dataset_' + uuid.uuid4().hex[:12])
        allow_holes, make_zip = self.holes_check.isChecked(), self.zip_check.isChecked()
        def save(progress):
            export_dataset(self.project, output, allow_holes=allow_holes, progress=progress)
            if make_zip:
                zip_dataset(output, output.with_suffix('.zip'), progress)
            return output
        def done(path):
            self.dataset.setText(str(path))
            self.val_dataset.setText(str(path))
            QMessageBox.information(self, 'Dataset exported', f'{path}\n' +
                                    ('ZIP alongside folder.' if make_zip else 'Ready for training.'))
        self.background(save, done)

    def path_row(self, form, name, default='', folder=False):
        row = QHBoxLayout()
        entry = QLineEdit(default)
        row.addWidget(entry)
        def browse():
            path = QFileDialog.getExistingDirectory(self, name) if folder else QFileDialog.getOpenFileName(self, name, '', 'Weights (*.pt);;All files (*)')[0]
            if path:
                entry.setText(path)
        button('Browse', browse, row)
        form.addRow(name, row)
        return entry

    def build_training(self):
        layout = self.page('3  Training')
        form = QFormLayout()
        self.dataset = self.path_row(form, 'Reviewed dataset folder', folder=True)
        self.weights = self.path_row(form, 'Pretrained weights / official checkpoint name', 'yoloe-26s-seg.pt')
        self.family = QComboBox()
        self.family.addItems(['YOLOE', 'YOLO'])
        form.addRow('Model family (YOLOE uses YOLOEPESegTrainer)', self.family)
        self.device = QComboBox()
        self.device.addItems(['auto', 'cpu', '0'])
        form.addRow('Device (0 = CUDA GPU)', self.device)
        self.options = {}
        for key, title, value, low, high, decimal in [
            ('epochs', 'Epochs', 80, 1, 10000, False), ('imgsz', 'Image size', 640, 64, 4096, False),
            ('batch', 'Batch size (lower for limited VRAM)', 4, 1, 128, False),
            ('patience', 'Early stopping patience', 15, 0, 1000, False),
            ('seed', 'Seed', 42, 0, 2147483647, False), ('degrees', 'Rotation ± degrees', 180, 0, 180, True),
            ('fliplr', 'Horizontal flip probability', .5, 0, 1, True), ('flipud', 'Vertical flip probability', .5, 0, 1, True),
            ('brightness', 'Brightness augmentation', .2, 0, 1, True), ('mosaic', 'Mosaic probability', .2, 0, 1, True)]:
            widget = spin(value, low, high, decimal)
            form.addRow(title, widget)
            self.options[key] = widget
        layout.addLayout(form)
        row = QHBoxLayout()
        button('Start training…', lambda: self.start_job('train'), row)
        button('Stop job…', self.stop_job, row)
        button('Open run folder', self.open_run, row)
        layout.addLayout(row)
        self.train_progress = QProgressBar()
        layout.addWidget(self.train_progress)
        self.plot = MetricPlot()
        layout.addWidget(self.plot)
        self.train_log = QPlainTextEdit()
        self.train_log.setReadOnly(True)
        self.train_log.setMaximumBlockCount(2500)
        layout.addWidget(self.train_log)

    def build_validation(self):
        layout = self.page('4  Validation')
        form = QFormLayout()
        self.val_dataset = self.path_row(form, 'Reviewed dataset folder', folder=True)
        self.val_weights = self.path_row(form, 'Trained best.pt')
        self.val_split = QComboBox()
        self.val_split.addItems(['test', 'val'])
        form.addRow('Held-out split (test once after model choice)', self.val_split)
        layout.addLayout(form)
        row = QHBoxLayout()
        button('Run validation…', lambda: self.start_job('val'), row)
        button('Stop job…', self.stop_job, row)
        button('Open run folder', self.open_run, row)
        layout.addLayout(row)
        self.val_report = QPlainTextEdit()
        self.val_report.setReadOnly(True)
        layout.addWidget(self.val_report)
        self.artifacts = QComboBox()
        self.artifacts.currentIndexChanged.connect(self.show_artifact)
        layout.addWidget(self.artifacts)
        self.artifact_view = QLabel('Validation plots and confusion matrices appear here')
        self.artifact_view.setMinimumHeight(280)
        self.artifact_view.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.artifact_view)

    def start_job(self, action):
        if self.process is not None and self.process.state() != QProcess.ProcessState.NotRunning:
            self.error('A job is already running')
            return
        dataset = Path((self.dataset if action == 'train' else self.val_dataset).text())
        if not (dataset / 'manifest.json').is_file() or not (dataset / 'data.yaml').is_file():
            self.error('Select a reviewed dataset exported by this app')
            return
        weights = (self.weights if action == 'train' else self.val_weights).text().strip()
        if not weights:
            self.error('Select weights')
            return
        if QMessageBox.question(self, 'Start model job',
            'This starts model execution on this computer and may download official weights/text encoders. '
            'Only load trusted .pt files. Continue?') != QMessageBox.StandardButton.Yes:
            return
        parent = QFileDialog.getExistingDirectory(self, 'Output parent (new run folder created)')
        if not parent:
            return
        output = Path(parent) / (action + '_' + uuid.uuid4().hex[:12])
        config = {key: widget.value() for key, widget in self.options.items()}
        config.update(action=action, dataset=str(dataset.resolve()), output=str(output), weights=weights,
                      family=self.family.currentText(), device=self.device.currentText(), split=self.val_split.currentText())
        config_path = Path(parent) / (output.name + '.job.json')
        write_json(config_path, config)
        self.job_output = output
        self.job_action = action
        self.job_buffer = ''
        self.process = QProcess(self)
        self.process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        self.process.readyReadStandardOutput.connect(self.read_job_output)
        self.process.errorOccurred.connect(lambda error: self.error(self.process.errorString()))
        self.process.finished.connect(self.job_finished)
        self.train_progress.setValue(0)
        self.train_progress.setMaximum(config['epochs'])
        self.train_log.clear()
        self.process.setProgram(sys.executable)
        self.process.setArguments(['-u', str(Path(__file__).with_name('worker.py')), str(config_path)])
        self.process.start()

    def read_job_output(self):
        text = bytes(self.process.readAllStandardOutput()).decode('utf-8', errors='replace')
        self.train_log.moveCursor(self.train_log.textCursor().MoveOperation.End)
        self.train_log.insertPlainText(text)
        if self.job_output and self.job_output.exists():
            with (self.job_output / 'console.log').open('a', encoding='utf-8') as handle:
                handle.write(text)
        self.job_buffer += text
        while '\n' in self.job_buffer:
            line, self.job_buffer = self.job_buffer.split('\n', 1)
            if line.startswith('ROBOQC_EVENT '):
                try:
                    value = json.loads(line[len('ROBOQC_EVENT '):])
                    if 'epoch' in value:
                        self.train_progress.setValue(value['epoch'])
                    if 'weights' in value:
                        self.val_weights.setText(value['weights'])
                    if 'report' in value:
                        self.val_report.setPlainText(json.dumps(value['report'], indent=2))
                    self.statusBar().showMessage(str(value))
                except (ValueError, TypeError):
                    pass
        if len(self.job_buffer) > 100000:
            self.job_buffer = self.job_buffer[-100000:]

    def poll_metrics(self):
        if not self.job_output or not self.job_output.exists():
            return
        csv_path = self.job_output / 'train/results.csv'
        if csv_path.exists():
            try:
                with csv_path.open(encoding='utf-8') as handle:
                    self.plot.rows = [{k.strip(): v.strip() for k, v in row.items() if k and v is not None}
                                      for row in csv.DictReader(handle)]
                self.plot.update()
            except OSError:
                pass

    def job_finished(self, code, status):
        self.read_job_output()
        self.poll_metrics()
        self.statusBar().showMessage(f'Job exited with code {code}. Outputs: {self.job_output}')
        if code != 0:
            self.error(f'Job failed or stopped (code {code}). See Training logs and run folder.')
        self.artifacts.clear()
        if self.job_output.exists():
            for path in sorted(self.job_output.rglob('*.png')) + sorted(self.job_output.rglob('*.jpg')):
                self.artifacts.addItem(str(path.relative_to(self.job_output)), str(path))

    def show_artifact(self, index):
        path = self.artifacts.currentData()
        if path:
            image = QPixmap(path)
            self.artifact_view.setPixmap(image.scaled(self.artifact_view.size(), Qt.AspectRatioMode.KeepAspectRatio,
                                                      Qt.TransformationMode.SmoothTransformation))

    def stop_job(self):
        if self.process and self.process.state() != QProcess.ProcessState.NotRunning:
            if QMessageBox.question(self, 'Stop job', 'Stop the worker? The last fully saved epoch checkpoint remains; '
                                   'the current epoch may be lost.') == QMessageBox.StandardButton.Yes:
                process = self.process
                process.terminate()
                QTimer.singleShot(5000, lambda: process.kill() if process.state() != QProcess.ProcessState.NotRunning else None)

    def open_run(self):
        if self.job_output:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.job_output)))

    def closeEvent(self, event):
        if self.task_thread or (self.process and self.process.state() != QProcess.ProcessState.NotRunning):
            self.error('Wait for the file operation to finish or stop the model job before closing')
            event.ignore()
            return
        if not self.discard_changes():
            event.ignore()
            return
        if self.bundle:
            self.bundle.close()
        event.accept()


def main():
    application = QApplication(sys.argv)
    window = Window()
    window.show()
    sys.exit(application.exec())
