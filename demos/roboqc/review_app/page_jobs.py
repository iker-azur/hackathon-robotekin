"""Training and validation pages. Models only run when the user starts a job."""

import csv
import json
from pathlib import Path
import sys
import uuid

from PySide6.QtCore import QObject, QPointF, QProcess, QRectF, Qt, QTimer, QUrl, Signal
from PySide6.QtGui import QColor, QDesktopServices, QFont, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QComboBox, QFileDialog, QHBoxLayout, QLabel, QLineEdit, QMessageBox, QPlainTextEdit, QProgressBar, QPushButton,
    QSizePolicy, QSpinBox, QVBoxLayout, QWidget,
)

from . import augment
from .core import write_json
from .theme import BUTTON_W, GAP, PALETTE
from .widgets import ActionButton, Card, ControlColumn, FormGrid, Page


def path_row(form, label, default='', folder=False, filter_text='Weights (*.pt);;All files (*)'):
    row = QHBoxLayout()
    row.setSpacing(GAP)
    entry = QLineEdit(default)
    row.addWidget(entry, 1)
    browse = QPushButton('Browse…')
    browse.setMinimumWidth(BUTTON_W - 16)
    def pick():
        path = (QFileDialog.getExistingDirectory(entry, label) if folder
                else QFileDialog.getOpenFileName(entry, label, '', filter_text)[0])
        if path:
            entry.setText(path)
    browse.clicked.connect(pick)
    row.addWidget(browse)
    form.add_wide(label, row)
    return entry


def spin(value, low, high):
    widget = QSpinBox()
    widget.setRange(low, high)
    widget.setValue(value)
    return widget


class MetricPlot(QWidget):
    """Live mask / box mAP50-95 curves read from Ultralytics' results.csv."""

    SERIES = [('metrics/mAP50-95(M)', 'Mask mAP50-95', '#15803D'), ('metrics/mAP50-95(B)', 'Box mAP50-95', '#2563EB')]

    def __init__(self):
        super().__init__()
        self.rows = []
        self.setMinimumHeight(220)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor(PALETTE['surface']))
        left, top, right, bottom = 44, 36, 16, 28
        area = QRectF(left, top, self.width() - left - right, self.height() - top - bottom)
        painter.setFont(QFont('Segoe UI', 8))
        for step in range(5):
            y = area.bottom() - area.height() * step / 4
            painter.setPen(QPen(QColor(PALETTE['border']), 1))
            painter.drawLine(QPointF(area.left(), y), QPointF(area.right(), y))
            painter.setPen(QColor(PALETTE['muted']))
            painter.drawText(QRectF(0, y - 8, left - 6, 16), Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
                             f'{step / 4:.2f}')
        x_legend = left
        for key, name, color in self.SERIES:
            painter.setPen(QPen(QColor(color), 3))
            painter.drawLine(QPointF(x_legend, 16), QPointF(x_legend + 18, 16))
            painter.setPen(QColor(PALETTE['text']))
            painter.drawText(QPointF(x_legend + 24, 20), name)
            x_legend += 150
            values = []
            for row in self.rows:
                try:
                    values.append(float(row[key]))
                except (KeyError, ValueError):
                    pass
            if len(values) < 2:
                continue
            painter.setPen(QPen(QColor(color), 2))
            span = len(values) - 1
            points = [QPointF(area.left() + area.width() * i / span,
                              area.bottom() - area.height() * max(0., min(1., v))) for i, v in enumerate(values)]
            for a, b in zip(points, points[1:]):
                painter.drawLine(a, b)
        painter.setPen(QColor(PALETTE['muted']))
        painter.drawText(QRectF(area.left(), area.bottom() + 6, area.width(), 16), Qt.AlignmentFlag.AlignCenter,
                         f'epoch (1 – {len(self.rows)})' if self.rows else 'epoch')
        painter.setPen(QPen(QColor(PALETTE['border']), 1))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRect(QRectF(.5, .5, self.width() - 1, self.height() - 1))


class JobRunner(QObject):
    """One training / validation subprocess at a time."""

    output = Signal(str)
    event = Signal(dict)
    finished = Signal(int, str)

    def __init__(self, win):
        super().__init__(win)
        self.win = win
        self.process = None
        self.output_dir = None
        self.action = None
        self.buffer = ''

    @property
    def running(self):
        return self.process is not None and self.process.state() != QProcess.ProcessState.NotRunning

    def start(self, action, config, parent):
        if self.running:
            self.win.error('A job is already running')
            return False
        output = Path(parent) / (action + '_' + uuid.uuid4().hex[:12])
        config = {**config, 'action': action, 'output': str(output)}
        config_path = Path(parent) / (output.name + '.job.json')
        write_json(config_path, config)
        self.output_dir, self.action, self.buffer = output, action, ''
        self.process = QProcess(self)
        self.process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        self.process.readyReadStandardOutput.connect(self.read)
        self.process.errorOccurred.connect(lambda error: self.win.error(self.process.errorString()))
        self.process.finished.connect(self.done)
        self.process.setProgram(sys.executable)
        self.process.setArguments(['-u', str(Path(__file__).with_name('worker.py')), str(config_path)])
        self.process.start()
        return True

    def read(self):
        text = bytes(self.process.readAllStandardOutput()).decode('utf-8', errors='replace')
        self.output.emit(text)
        if self.output_dir and self.output_dir.exists():
            with (self.output_dir / 'console.log').open('a', encoding='utf-8') as handle:
                handle.write(text)
        self.buffer += text
        while '\n' in self.buffer:
            line, self.buffer = self.buffer.split('\n', 1)
            if line.startswith('ROBOQC_EVENT '):
                try:
                    self.event.emit(json.loads(line[len('ROBOQC_EVENT '):]))
                except (ValueError, TypeError):
                    pass
        if len(self.buffer) > 100000:
            self.buffer = self.buffer[-100000:]

    def done(self, code, status):
        self.read()
        self.finished.emit(code, self.action)

    def stop(self):
        if self.running:
            process = self.process
            process.terminate()
            QTimer.singleShot(5000, lambda: process.kill() if process.state() != QProcess.ProcessState.NotRunning else None)

    def open_folder(self):
        if self.output_dir:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.output_dir)))


class _JobPage(Page):
    """Control column on the left, work area on the right."""

    def __init__(self, win, title, subtitle):
        super().__init__(title, subtitle)
        self.win = win
        self.runner = win.job
        body = QHBoxLayout()
        body.setSpacing(16)
        self.column = ControlColumn()
        body.addWidget(self.column)
        self.work = QVBoxLayout()
        self.work.setSpacing(16)
        body.addLayout(self.work, 1)
        self.root.addLayout(body, 1)

    def confirm_and_start(self, action, config):
        dataset = Path(config['dataset'])
        if not (dataset / 'manifest.json').is_file() or not (dataset / 'data.yaml').is_file():
            self.win.error('Select a reviewed dataset exported by this app')
            return False
        if not config['weights'].strip():
            self.win.error('Select weights')
            return False
        if QMessageBox.question(self.win, 'Start model job',
                                'This starts model execution on this computer and may download official weights/text '
                                'encoders. Only load trusted .pt files. Continue?') != QMessageBox.StandardButton.Yes:
            return False
        start = (self.win.project.pref('runs_dir') if self.win.project else '') or ''
        parent = QFileDialog.getExistingDirectory(self.win, 'Output parent (a new run folder is created)', start)
        if not parent:
            return False
        if self.win.project:
            self.win.project.set_pref('runs_dir', parent)
        return self.runner.start(action, config, parent)

    def job_buttons(self, start_text, start_callback):
        card = Card('Run')
        self.start_button = ActionButton(start_text, None, role='primary')
        self.start_button.clicked.connect(start_callback)
        card.add(self.start_button)
        row = QHBoxLayout()
        row.setSpacing(GAP)
        self.stop_button = ActionButton('Stop', None, role='danger')
        self.stop_button.clicked.connect(self.stop)
        self.folder_button = ActionButton('Open run folder', None)
        self.folder_button.clicked.connect(self.runner.open_folder)
        row.addWidget(self.stop_button)
        row.addWidget(self.folder_button)
        card.add_layout(row)
        self.column.add(card)

    def stop(self):
        if self.runner.running and QMessageBox.question(
                self.win, 'Stop job', 'Stop the worker? The last fully saved epoch checkpoint remains; '
                'the current epoch may be lost.') == QMessageBox.StandardButton.Yes:
            self.runner.stop()


class TrainingPage(_JobPage):
    def __init__(self, win):
        super().__init__(win, 'Training', 'Fine-tune YOLOE or YOLO segmentation on the dataset you exported.')
        data = Card('Dataset and model')
        form = FormGrid()
        self.dataset = path_row(form, 'Reviewed dataset', folder=True)
        self.weights = path_row(form, 'Weights / checkpoint', 'yoloe-26s-seg.pt')
        self.family = QComboBox()
        self.family.addItems(['YOLOE', 'YOLO'])
        form.add_row('Model family', self.family, 'YOLOE uses YOLOEPESegTrainer; YOLO uses the standard trainer.')
        self.device = QComboBox()
        self.device.addItems(['auto', 'cpu', '0'])
        form.add_row('Device', self.device, '0 = first CUDA GPU.')
        data.add_layout(form)
        self.column.add(data)

        params = Card('Hyperparameters')
        form = FormGrid()
        self.options = {}
        for key, title, value, low, high, tip in [
                ('epochs', 'Epochs', 80, 1, 10000, 'Early stopping can end sooner.'),
                ('imgsz', 'Image size', 640, 64, 4096, 'Training resolution in pixels.'),
                ('batch', 'Batch size', 4, 1, 128, 'Lower it if you run out of VRAM.'),
                ('patience', 'Early-stopping patience', 15, 0, 1000, 'Epochs without improvement before stopping.'),
                ('seed', 'Seed', 42, 0, 2147483647, 'Makes runs repeatable.')]:
            widget = spin(value, low, high)
            form.add_row(title, widget, tip)
            self.options[key] = widget
        params.add_layout(form)
        self.column.add(params)

        aug = Card('Augmentation')
        self.aug_summary = QLabel()
        self.aug_summary.setWordWrap(True)
        aug.add(self.aug_summary)
        edit = ActionButton('Edit augmentation', 'page_3')
        edit.clicked.connect(lambda: win.go(2))
        aug.add(edit)
        self.column.add(aug)
        self.job_buttons('Start training', self.start)

        progress = Card('Progress')
        self.progress = QProgressBar()
        self.progress.setFormat('epoch %v of %m')
        progress.add(self.progress)
        self.plot = MetricPlot()
        progress.add(self.plot, 1)
        self.work.addWidget(progress, 3)
        log = Card('Log')
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(2500)
        self.log.setFont(QFont('Consolas', 9))
        log.add(self.log, 1)
        self.work.addWidget(log, 2)

        self.runner.output.connect(self.on_output)
        self.runner.event.connect(self.on_event)
        self.runner.finished.connect(self.on_finished)
        self.metrics_timer = QTimer(self)
        self.metrics_timer.timeout.connect(self.poll_metrics)
        self.metrics_timer.start(2000)
        self.refresh_summary()

    def refresh_summary(self):
        self.aug_summary.setText(self.win.augment.summary())

    def config(self):
        config = {key: widget.value() for key, widget in self.options.items()}
        config.update(augment.train_kwargs(self.win.augment.config()))
        config.update(dataset=str(Path(self.dataset.text()).resolve()) if self.dataset.text() else '',
                      weights=self.weights.text().strip(), family=self.family.currentText(),
                      device=self.device.currentText(), split='val')
        return config

    def start(self):
        config = self.config()
        if self.confirm_and_start('train', config):
            self.progress.setValue(0)
            self.progress.setMaximum(config['epochs'])
            self.log.clear()
            self.plot.rows = []

    def on_output(self, text):
        if self.runner.action != 'train':
            return
        self.log.moveCursor(self.log.textCursor().MoveOperation.End)
        self.log.insertPlainText(text)

    def on_event(self, value):
        if self.runner.action != 'train':
            return
        if 'epoch' in value:
            self.progress.setValue(value['epoch'])
        if 'weights' in value:
            self.win.validation.weights.setText(value['weights'])
        self.win.status(str(value), 8000)

    def on_finished(self, code, action):
        if action != 'train':
            return
        self.poll_metrics()
        self.win.status(f'Training exited with code {code}. Outputs: {self.runner.output_dir}', 15000)
        if code != 0:
            self.win.error(f'Job failed or stopped (code {code}). See the log and the run folder.')

    def poll_metrics(self):
        output = self.runner.output_dir
        if self.runner.action != 'train' or not output or not output.exists():
            return
        csv_path = output / 'train/results.csv'
        if csv_path.exists():
            try:
                with csv_path.open(encoding='utf-8') as handle:
                    self.plot.rows = [{k.strip(): v.strip() for k, v in row.items() if k and v is not None}
                                      for row in csv.DictReader(handle)]
                self.plot.update()
            except OSError:
                pass


class ValidationPage(_JobPage):
    def __init__(self, win):
        super().__init__(win, 'Validation', 'Measure a trained model on the held-out split of an exported dataset.')
        data = Card('Dataset and weights')
        form = FormGrid()
        self.dataset = path_row(form, 'Reviewed dataset', folder=True)
        self.weights = path_row(form, 'Trained best.pt')
        self.split = QComboBox()
        self.split.addItems(['test', 'val'])
        form.add_row('Held-out split', self.split, 'Use val while developing; run test once, after choosing the model.')
        data.add_layout(form)
        self.column.add(data)
        self.job_buttons('Run validation', self.start)

        report = Card('Metrics')
        self.report = QPlainTextEdit()
        self.report.setReadOnly(True)
        self.report.setFont(QFont('Consolas', 9))
        report.add(self.report, 1)
        self.work.addWidget(report, 2)
        plots = Card('Plots and confusion matrices')
        self.artifacts = QComboBox()
        self.artifacts.currentIndexChanged.connect(self.show_artifact)
        plots.add(self.artifacts)
        self.view = QLabel('Plots appear here after a validation run.')
        self.view.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.view.setMinimumHeight(260)
        self.view.setProperty('muted', True)
        plots.add(self.view, 1)
        self.work.addWidget(plots, 3)
        self.runner.event.connect(self.on_event)
        self.runner.finished.connect(self.on_finished)

    def config(self):
        return dict(dataset=str(Path(self.dataset.text()).resolve()) if self.dataset.text() else '',
                    weights=self.weights.text().strip(), family=self.win.training.family.currentText(),
                    device=self.win.training.device.currentText(), split=self.split.currentText(),
                    imgsz=self.win.training.options['imgsz'].value(), batch=self.win.training.options['batch'].value())

    def start(self):
        self.report.clear()
        self.confirm_and_start('val', self.config())

    def on_event(self, value):
        if self.runner.action == 'val' and 'report' in value:
            self.report.setPlainText(json.dumps(value['report'], indent=2))

    def on_finished(self, code, action):
        if action != 'val':
            return
        self.win.status(f'Validation exited with code {code}. Outputs: {self.runner.output_dir}', 15000)
        if code != 0:
            self.win.error(f'Job failed or stopped (code {code}). See the run folder.')
        self.artifacts.clear()
        output = self.runner.output_dir
        if output and output.exists():
            for path in sorted(output.rglob('*.png')) + sorted(output.rglob('*.jpg')):
                self.artifacts.addItem(str(path.relative_to(output)), str(path))

    def show_artifact(self, index):
        path = self.artifacts.currentData()
        if path:
            image = QPixmap(path)
            self.view.setPixmap(image.scaled(self.view.size(), Qt.AspectRatioMode.KeepAspectRatio,
                                             Qt.TransformationMode.SmoothTransformation))
