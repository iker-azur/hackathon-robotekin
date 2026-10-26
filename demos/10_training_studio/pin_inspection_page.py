"""Independent paired-image classification page for Training Studio."""

import codecs
import json
import math
import re
import sys
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from PySide6.QtCore import QProcess, QProcessEnvironment, Qt, QTimer
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QAbstractSpinBox,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

PIN_DIR = Path(__file__).resolve().parents[1] / "10_pin_inspection"
OUTPUTS = PIN_DIR / "outputs"
LABELS = {"sin_perno": "Pieza sin perno", "con_perno": "Pieza con perno"}


class PinInspectionPage(QWidget):
    def __init__(self, studio):
        super().__init__()
        self.studio = studio
        self.pairs = []
        self._buttons = []
        self._action = ""
        self._buffer = ""
        self._decoder = codecs.getincrementaldecoder("utf-8")("replace")
        self._settings = {}
        self._error = False
        self._stop_requested = False
        self._run_output = None
        self._stop_path = None
        self.process = QProcess(self)
        self.process.setProcessChannelMode(QProcess.MergedChannels)
        self.process.started.connect(self._send_settings)
        self.process.readyReadStandardOutput.connect(self._read_output)
        self.process.finished.connect(self._finished)
        self.process.errorOccurred.connect(self._process_error)
        page, layout = studio._new_page(
            "Inspección de pernos", "Vistas frontal y de muñeca de la misma pieza, juntas o por separado. "
            "Una clasificación con perno/sin perno, independiente del entrenamiento ACT.")
        container = QVBoxLayout(self)
        container.setContentsMargins(0, 0, 0, 0)
        container.addWidget(page)
        self.tabs = QTabWidget()
        self.tabs.addTab(self._data_tab(), "Datos")
        self.tabs.addTab(self._training_tab(), "Entrenamiento")
        self.tabs.addTab(self._results_tab(), "Resultados")
        layout.addWidget(self.tabs)
        console, console_layout = studio._card("Estado y registro")
        self.status = QLabel("Selecciona la carpeta del descargador o descarga las fotos de MikelRubi.")
        self.status.setWordWrap(True)
        console_layout.addWidget(self.status)
        self.progress = QProgressBar()
        self.progress.hide()
        console_layout.addWidget(self.progress)
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(4000)
        self.log.setFixedHeight(130)
        console_layout.addWidget(self.log)
        self.stop_button = QPushButton("Detener proceso")
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(self.stop)
        console_layout.addWidget(self.stop_button)
        layout.addWidget(console)
        self.data_root.textChanged.connect(self._root_changed)
        self.refresh_runs()

    @property
    def busy(self):
        return self.process.state() != QProcess.NotRunning

    @property
    def is_training(self):
        return self.busy and self._action == "train"

    def _button(self, text, callback, layout):
        button = QPushButton(text)
        self._buttons.append(button)
        button.clicked.connect(callback)
        layout.addWidget(button)
        return button

    def _data_tab(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        form = QFormLayout()
        self.owner = QLineEdit("MikelRubi")
        self.data_root = QLineEdit(str(PIN_DIR / "data"))
        form.addRow("Cuenta de Hugging Face", self.owner)
        form.addRow("Carpeta local", self.data_root)
        layout.addLayout(form)
        buttons = QHBoxLayout()
        self._button("Elegir carpeta…", self._choose_root, buttons)
        self.download_button = self._button("Descargar fotos etiquetadas", self.download, buttons)
        self.scan_button = self._button("Analizar carpeta", self.scan, buttons)
        layout.addLayout(buttons)
        self.download_status = QLabel("Reutiliza pin_snapshots y las etiquetas originales. No necesita vídeos ni ACT.")
        self.download_status.setWordWrap(True)
        layout.addWidget(self.download_status)
        self.sessions = QListWidget()
        self.sessions.setMinimumHeight(100)
        self.sessions.setMaximumHeight(160)
        self.sessions.itemChanged.connect(self._selection_changed)
        layout.addWidget(QLabel("Sesiones para entrenamiento y validación"))
        layout.addWidget(self.sessions)
        self.counts = QLabel("Sin parejas analizadas")
        self.counts.setWordWrap(True)
        layout.addWidget(self.counts)
        self.prepare_button = self._button("Comprobar partición de sesiones 80/20", self.prepare, layout)
        self.pair_list = QListWidget()
        self.pair_list.setMaximumHeight(130)
        self.pair_list.currentItemChanged.connect(self._preview_pair)
        layout.addWidget(self.pair_list)
        images = QHBoxLayout()
        self.preview_labels = {}
        for camera in ("front", "wrist"):
            column = QVBoxLayout()
            column.addWidget(QLabel(camera))
            image = QLabel("Sin imagen")
            image.setAlignment(Qt.AlignCenter)
            image.setMinimumSize(180, 140)
            image.setMaximumHeight(180)
            column.addWidget(image)
            images.addLayout(column)
            self.preview_labels[camera] = image
        layout.addLayout(images)
        self.exclusions = QPlainTextEdit()
        self.exclusions.setReadOnly(True)
        self.exclusions.setMaximumHeight(85)
        self.exclusions.setPlaceholderText("Aquí se mostrarán las parejas excluidas y sus motivos.")
        layout.addWidget(self.exclusions)
        return page

    def _training_tab(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        description = QLabel("ResNet18 preentrenada compartida por las cámaras elegidas; fusión 512 por cámara → 128 → 2. "
                             "Se ajustan extractor y clasificador. Cada vista conserva el encuadre completo a 224×224.")
        description.setWordWrap(True)
        layout.addWidget(description)
        form = QFormLayout()
        self.python_path = QLineEdit(sys.executable)
        python_row = QHBoxLayout()
        python_row.addWidget(self.python_path)
        self._button("Elegir Python…", self._choose_python, python_row)
        form.addRow("Intérprete del clasificador", python_row)
        self.epochs = QSpinBox()
        self.batch_size = QSpinBox()
        self.seed = QSpinBox()
        for spin, maximum, value, minimum in (
            (self.epochs, 10000, 30, 1), (self.batch_size, 512, 16, 1),
            (self.seed, 2147483647, 42, 0),
        ):
            self._style_spin(spin, minimum, maximum, value)
        self.learning_rate = QLineEdit("0.0001")
        self.backbone_lr = QLineEdit("0.00001")
        self.device = QComboBox()
        self.device.addItems(["auto", "cuda", "cpu"])
        self.cameras = QComboBox()
        for text, cameras in (("front + wrist", ["front", "wrist"]), ("Solo front", ["front"]),
                              ("Solo wrist", ["wrist"])):
            self.cameras.addItem(text, cameras)
        self.run_name = QLineEdit("pin_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
        for title, widget in (("Cámaras", self.cameras), ("Épocas", self.epochs), ("Parejas por lote", self.batch_size),
                              ("Aprendizaje del clasificador", self.learning_rate),
                              ("Aprendizaje de ResNet18", self.backbone_lr), ("Semilla", self.seed),
                              ("Dispositivo", self.device), ("Nombre de ejecución", self.run_name)):
            form.addRow(title, widget)
        layout.addLayout(form)
        layout.addWidget(self._augmentation_box())
        path = QLabel(f"Modelos y métricas: {OUTPUTS}\nNecesita requirements-training.txt del módulo de pernos; "
                      "el primer uso puede descargar los pesos ImageNet de ResNet18.")
        path.setWordWrap(True)
        layout.addWidget(path)
        self.train_button = self._button("Entrenar clasificador", self.start_training, layout)
        self.train_button.setObjectName("primaryButton")
        self.metrics = QLabel("Las métricas de validación aparecerán por época.")
        self.metrics.setWordWrap(True)
        layout.addWidget(self.metrics)
        self.chart = type(self.studio.live_loss_chart)()
        layout.addWidget(self.chart)
        layout.addStretch(1)
        return page

    @staticmethod
    def _style_spin(spin, minimum, maximum, value, suffix=""):
        spin.setRange(minimum, maximum)
        spin.setValue(value)
        spin.setSuffix(suffix)
        spin.setKeyboardTracking(False)
        spin.setButtonSymbols(QAbstractSpinBox.UpDownArrows)
        # Define the native subcontrols explicitly: Qt 6.12/Windows can otherwise
        # give the embedded editor a one-pixel width with the Studio's stylesheet.
        spin.setStyleSheet("""
            QSpinBox { padding: 6px 28px 6px 8px; min-height: 24px; }
            QSpinBox::up-button {
                subcontrol-origin: border; subcontrol-position: top right;
                width: 24px; border-left: 1px solid #d9dce2;
            }
            QSpinBox::down-button {
                subcontrol-origin: border; subcontrol-position: bottom right;
                width: 24px; border-left: 1px solid #d9dce2;
            }
            QSpinBox::up-arrow { image: url("PIN_UP_ARROW"); width: 8px; height: 5px; }
            QSpinBox::down-arrow { image: url("PIN_DOWN_ARROW"); width: 8px; height: 5px; }
            QSpinBox QLineEdit {
                padding: 0px; margin: 0px; border: 0px;
                color: #252932; background: white;
            }
        """.replace("PIN_UP_ARROW", (Path(__file__).parent / "icons/spin-up.svg").as_posix())
           .replace("PIN_DOWN_ARROW", (Path(__file__).parent / "icons/spin-down.svg").as_posix()))

    def _augmentation_box(self):
        self.augmentation_box = QGroupBox("Aumentación de datos (solo entrenamiento; se sortea de nuevo en cada época)")
        self.augmentation_box.setCheckable(True)
        self.augmentation_box.setChecked(False)
        grid = QGridLayout(self.augmentation_box)
        for column, title in enumerate(("Transformación", "Probabilidad", "Intensidad máxima")):
            grid.addWidget(QLabel(title), 0, column)
        # key, label, default probability %, (config key, divisor to config units, min, max, default, suffix)
        self.augmentations = {}
        for row, (key, text, probability, strength) in enumerate((
            ("horizontal_flip", "Volteo horizontal", 50, None),
            ("vertical_flip", "Volteo vertical", 50, None),
            ("rotation", "Rotación", 50, ("rotation_degrees", 1, 1, 180, 15, " °")),
            ("translation", "Traslación", 50, ("translation_fraction", 100, 1, 50, 10, " %")),
            ("perspective", "Cambio de perspectiva", 30, ("perspective_scale", 100, 1, 100, 30, " %")),
        ), start=1):
            check = QCheckBox(text)
            chance = QSpinBox()
            self._style_spin(chance, 1, 100, probability, " %")
            grid.addWidget(check, row, 0)
            grid.addWidget(chance, row, 1)
            amount = None
            if strength:
                amount = QSpinBox()
                self._style_spin(amount, strength[2], strength[3], strength[4], strength[5])
                grid.addWidget(amount, row, 2)
            else:
                grid.addWidget(QLabel("—"), row, 2)
            self.augmentations[key] = (check, chance, strength, amount)
        return self.augmentation_box

    def augmentation_settings(self):
        """Probabilities as 0..1; unselected transforms are sent as 0 so the run config records them."""
        enabled = self.augmentation_box.isChecked()
        settings = {}
        for key, (check, chance, strength, amount) in self.augmentations.items():
            settings[key] = chance.value() / 100 if enabled and check.isChecked() else 0.0
            if strength:
                settings[strength[0]] = amount.value() / strength[1]
        return settings

    def _results_tab(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        self._button("Actualizar ejecuciones", self.refresh_runs, layout)
        self.runs = QListWidget()
        self.runs.setMaximumHeight(150)
        self.runs.currentItemChanged.connect(self._run_selected)
        layout.addWidget(self.runs)
        self.result_metrics = QPlainTextEdit()
        self.result_metrics.setReadOnly(True)
        self.result_metrics.setMaximumHeight(230)
        layout.addWidget(self.result_metrics)
        self.inference_paths = {}
        for camera in ("front", "wrist"):
            row = QHBoxLayout()
            row.addWidget(QLabel(camera))
            field = QLineEdit()
            self.inference_paths[camera] = field
            row.addWidget(field)
            self._button("Elegir imagen…", lambda _=False, name=camera: self._choose_image(name), row)
            layout.addLayout(row)
        self.predict_button = self._button("Clasificar con best.pt", self.predict, layout)
        self.prediction = QLabel("Elige las vistas de la misma pieza que usa la ejecución.")
        self.prediction.setWordWrap(True)
        layout.addWidget(self.prediction)
        layout.addStretch(1)
        return page

    def _choose_root(self):
        folder = QFileDialog.getExistingDirectory(self, "Carpeta del descargador de pernos", self.data_root.text())
        if folder:
            self.data_root.setText(folder)

    def _choose_python(self):
        path, _ = QFileDialog.getOpenFileName(self, "Intérprete Python del clasificador")
        if path:
            self.python_path.setText(path)

    def _choose_image(self, camera):
        path, _ = QFileDialog.getOpenFileName(self, f"Imagen {camera}", self.data_root.text(),
                                             "Imágenes (*.png *.jpg *.jpeg *.bmp *.webp *.tif *.tiff)")
        if path:
            self.inference_paths[camera].setText(path)

    def _root_changed(self):
        self.pairs = []
        self.sessions.clear()
        self.pair_list.clear()
        self.counts.setText("Carpeta cambiada; pulsa Analizar carpeta.")
        self.exclusions.clear()
        for label in self.preview_labels.values():
            label.setText("Sin imagen")

    def selected_repos(self):
        return [self.sessions.item(i).data(Qt.UserRole) for i in range(self.sessions.count())
                if self.sessions.item(i).checkState() == Qt.Checked]

    def _selection_changed(self):
        selected = set(self.selected_repos())
        self.pair_list.clear()
        counts = {key: 0 for key in LABELS}
        for pair in self.pairs:
            if not any(source["repo_id"] in selected for source in pair["sources"]):
                continue
            counts[pair["label"]] += 1
            item = QListWidgetItem(f"{LABELS[pair['label']]} · {pair['repo_id']} · {pair['pair_id']} "
                                   f"· ubicación {pair.get('location')}")
            item.setData(Qt.UserRole, pair)
            self.pair_list.addItem(item)
        self.counts.setText(f"{len(selected)} sesiones seleccionadas · "
                            f"{sum(counts.values())} parejas únicas · "
                            f"Con perno: {counts['con_perno']} · Sin perno: {counts['sin_perno']}")
        self.pair_list.setCurrentRow(0)

    def _preview_pair(self, item, _previous=None):
        if item is None:
            for label in self.preview_labels.values():
                label.setText("Sin imagen")
            return
        pair = item.data(Qt.UserRole)
        for camera, label in self.preview_labels.items():
            pixmap = QPixmap(pair[camera])
            label.setPixmap(pixmap.scaled(340, 180, Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def download(self):
        if not self.owner.text().strip() or not self.data_root.text().strip():
            self.status.setText("Indica la cuenta y el directorio de destino.")
            return
        self._launch("download", {"owner": self.owner.text().strip(), "output": self.data_root.text().strip(),
                                  "with_pin_labels": [], "without_pin_labels": []})

    def scan(self):
        self._launch("scan", {"data_root": self.data_root.text().strip()})

    def prepare(self):
        self._launch("prepare", {"data_root": self.data_root.text().strip(),
                                 "selected_repos": self.selected_repos(), "seed": self.seed.value()})

    def start_training(self):
        if self.busy:
            self.status.setText("Ya hay un proceso de inspección activo.")
            return
        if self.studio.process.state() != QProcess.NotRunning:
            self.status.setText("Espera a que termine el proceso ACT antes de entrenar el clasificador.")
            return
        name = self.run_name.text().strip()
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name):
            self.status.setText("El nombre de ejecución admite letras, números, guiones y puntos.")
            return
        output = OUTPUTS / name
        if output.exists():
            self.status.setText("La ejecución ya existe; elige un nombre nuevo para conservarla.")
            return
        try:
            head_lr, backbone_lr = float(self.learning_rate.text()), float(self.backbone_lr.text())
        except ValueError:
            self.status.setText("Las tasas de aprendizaje deben ser números positivos.")
            return
        if any(not math.isfinite(value) or value <= 0 for value in (head_lr, backbone_lr)):
            self.status.setText("Las tasas de aprendizaje deben ser finitas y positivas.")
            return
        if not self.selected_repos():
            self.status.setText("Analiza la carpeta y selecciona sesiones primero.")
            return
        OUTPUTS.mkdir(parents=True, exist_ok=True)
        self._stop_path = OUTPUTS / f".{name}.{uuid4().hex}.stop"
        self._run_output = output
        self.chart.clear()
        self._launch("train", {"data_root": self.data_root.text().strip(), "output": str(output),
                               "selected_repos": self.selected_repos(), "epochs": self.epochs.value(),
                               "batch_size": self.batch_size.value(), "seed": self.seed.value(),
                               "device": self.device.currentText(), "cameras": self.cameras.currentData(),
                               "augmentation": self.augmentation_settings(),
                               "learning_rate": head_lr,
                               "backbone_lr": backbone_lr, "stop_file": str(self._stop_path)})

    def predict(self):
        item = self.runs.currentItem()
        if item is None or not (Path(item.data(Qt.UserRole)) / "best.pt").is_file():
            self.prediction.setText("Selecciona una ejecución con best.pt.")
            return
        cameras = self._run_cameras(Path(item.data(Qt.UserRole)))
        paths = {camera: self.inference_paths[camera].text().strip() for camera in cameras}
        if not all(path and Path(path).is_file() for path in paths.values()):
            self.prediction.setText(f"Selecciona la imagen {' y '.join(cameras)} de la misma pieza.")
            return
        self._launch("predict", {"checkpoint": str(Path(item.data(Qt.UserRole)) / "best.pt"), **paths})

    def _launch(self, action, settings, clear=True):
        if self.busy:
            self.status.setText("Ya hay un proceso de inspección activo.")
            return
        python = Path(self.python_path.text().strip().strip('"'))
        if not python.is_file():
            self.status.setText("Selecciona un intérprete Python existente en Entrenamiento.")
            return
        self._action, self._settings, self._error, self._stop_requested = action, settings, False, False
        self._buffer = ""
        self._decoder.reset()
        if clear:
            self.log.clear()
        environment = QProcessEnvironment.systemEnvironment()
        environment.insert("PYTHONIOENCODING", "utf-8")
        environment.insert("PYTHONUTF8", "1")
        self.process.setProcessEnvironment(environment)
        self.process.setWorkingDirectory(str(PIN_DIR))
        script = PIN_DIR / ("download_worker.py" if action == "download" else "classifier_worker.py")
        arguments = ["-X", "utf8", "-u", str(script)] + ([] if action == "download" else [action])
        self._set_busy(True)
        self.status.setText(f"Proceso {action} en curso…")
        self.progress.setRange(0, 0)
        self.progress.show()
        self.process.start(str(python), arguments)

    def _send_settings(self):
        self.process.write(json.dumps(self._settings, allow_nan=False).encode("utf-8"))
        self.process.closeWriteChannel()

    def _set_busy(self, busy):
        for widget in (self.owner, self.data_root, self.sessions, self.python_path, self.epochs,
                       self.batch_size, self.seed, self.learning_rate, self.backbone_lr, self.device, self.cameras, self.augmentation_box,
                       self.run_name, *self.inference_paths.values(), *self._buttons):
            widget.setEnabled(not busy)
        self.stop_button.setEnabled(busy)

    def _read_output(self):
        self._buffer += self._decoder.decode(bytes(self.process.readAllStandardOutput()))
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            self._consume_line(line)

    def _consume_line(self, line):
        if not line.startswith("@@PIN@@"):
            self.log.appendPlainText(line)
            return
        try:
            payload = json.loads(line[len("@@PIN@@"):])
            self._handle_event(payload)
        except (ValueError, KeyError, TypeError) as error:
            self._error = True
            self.status.setText(f"Respuesta del proceso inválida: {error}")

    def _handle_event(self, payload):
        event = payload["event"]
        if event == "data":
            if self._action == "scan":
                self.pairs = payload["pairs"]
                self.sessions.blockSignals(True)
                self.sessions.clear()
                for repo, counts in sorted(payload["sessions"].items()):
                    item = QListWidgetItem(f"{repo} · con: {counts.get('con_perno', 0)} "
                                           f"· sin: {counts.get('sin_perno', 0)}")
                    item.setData(Qt.UserRole, repo)
                    item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
                    item.setCheckState(Qt.Checked)
                    self.sessions.addItem(item)
                self.sessions.blockSignals(False)
                self._selection_changed()
            self.exclusions.setPlainText(json.dumps({"duplicates_removed": payload["duplicates_removed"],
                                                    "rejected": payload["rejected"]}, ensure_ascii=False, indent=2))
            self.status.setText(f"Parejas leídas: {len(payload['pairs'])}; "
                                f"exclusiones: {len(payload['rejected'])}.")
            if "split_counts" in payload:
                self.status.setText("Partición válida por sesiones: " + json.dumps(payload["split_counts"], ensure_ascii=False))
        elif event == "status":
            self.status.setText(payload["text"])
        elif event == "epoch":
            epoch, train, val = payload["epoch"], payload["train"], payload["validation"]
            self.progress.setRange(0, payload["epochs"])
            self.progress.setValue(epoch)
            self.chart.add_value("Training loss", epoch, train["loss"])
            self.chart.add_value("Validation loss", epoch, val["loss"])
            text = (f"Época {epoch}/{payload['epochs']} · Train loss {train['loss']:.4f}, "
                    f"accuracy {train['accuracy']:.1%} · Validación loss {val['loss']:.4f}, "
                    f"accuracy {val['accuracy']:.1%}, F1 macro {val['macro_f1']:.3f}")
            self.metrics.setText(text)
            self.log.appendPlainText(text)
        elif event == "result":
            self.status.setText(f"Entrenamiento {payload['status']}: {payload['output']}")
        elif event == "prediction":
            self.prediction.setText(LABELS[payload["label"]] + " · " + " · ".join(
                f"{LABELS[label]}: {value:.1%}" for label, value in payload["probabilities"].items()))
        elif event == "error":
            self._error = True
            self.status.setText(payload["message"])
            self.log.appendPlainText(payload["message"])

    def _finished(self, code, _exit_status):
        self._read_output()
        self._buffer += self._decoder.decode(b"", final=True)
        if self._buffer:
            self._consume_line(self._buffer)
            self._buffer = ""
        action = self._action
        self._set_busy(False)
        self.progress.hide()
        if self._stop_path and self._stop_path.exists():
            self._stop_path.unlink()
        if action == "train" and code != 0:
            self._mark_run_stopped()
        if code != 0 and not self._error:
            self.status.setText("Proceso detenido." if self._stop_requested else
                                f"El proceso terminó con código {code}; revisa el registro.")
        if action == "download":
            self.download_status.setText("Descarga completada." if code == 0 else
                                         "Descarga incompleta; revisa el registro y download_report.json.")
            if (Path(self.data_root.text()) / "inventory.jsonl").is_file() and not self._stop_requested:
                self._launch("scan", {"data_root": self.data_root.text().strip()}, clear=False)
        if action == "train":
            self.refresh_runs()

    def _process_error(self, error):
        if error == QProcess.FailedToStart:
            self._error = True
            self._set_busy(False)
            self.progress.hide()
            self.status.setText("No se pudo iniciar Python: " + self.process.errorString())

    def stop(self):
        if not self.busy:
            return
        self._stop_requested = True
        if self._action == "train" and self._stop_path:
            self._stop_path.write_text("stop", encoding="utf-8")
        else:
            self.process.terminate()
        self.status.setText("Deteniendo; se conservarán los checkpoints completos…")
        # Capture this process invocation, so the timer cannot kill a subsequent job.
        pid = self.process.processId()
        QTimer.singleShot(5000, lambda: self.process.kill() if self.busy and self.process.processId() == pid else None)

    def _mark_run_stopped(self):
        if not self._run_output:
            return
        path = self._run_output / "status.json"
        if path.is_file():
            state = json.loads(path.read_text(encoding="utf-8"))
            if state.get("status") == "running":
                state["status"] = "interrupted" if self._stop_requested else "failed"
                temporary = path.with_suffix(".json.tmp")
                temporary.write_text(json.dumps(state, indent=2), encoding="utf-8")
                temporary.replace(path)

    def shutdown(self):
        if self.busy:
            self.stop()
            if not self.process.waitForFinished(1500):
                self.process.kill()
                self.process.waitForFinished(1000)
            self._mark_run_stopped()

    def refresh_runs(self):
        if not hasattr(self, "runs"):
            return
        self.runs.clear()
        for folder in sorted(OUTPUTS.glob("*/run_config.json"), reverse=True):
            status_file = folder.parent / "status.json"
            try:
                state = json.loads(status_file.read_text(encoding="utf-8")) if status_file.exists() else {}
                item = QListWidgetItem(f"{folder.parent.name} · {state.get('status', 'unknown')}")
                item.setData(Qt.UserRole, str(folder.parent))
                self.runs.addItem(item)
            except (OSError, ValueError):
                continue
        self.runs.setCurrentRow(0)

    def _run_selected(self, item, _previous=None):
        if item is None:
            self.result_metrics.clear()
            return
        folder = Path(item.data(Qt.UserRole))
        try:
            path = folder / "metrics.json"
            data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"note": "Sin épocas completas"}
            cameras = self._run_cameras(folder)
            self.result_metrics.setPlainText(f"Cámaras: {' + '.join(cameras)}\n"
                                            "Validación del mejor checkpoint (filas reales, columnas predichas):\n" +
                                            json.dumps(data, ensure_ascii=False, indent=2))
            self.prediction.setText(f"Esta ejecución clasifica con: {' + '.join(cameras)}.")
        except (OSError, ValueError) as error:
            self.result_metrics.setPlainText(str(error))

    @staticmethod
    def _run_cameras(folder):
        # Runs created before camera selection always used both views.
        config = json.loads((folder / "run_config.json").read_text(encoding="utf-8"))
        return config.get("cameras", ["front", "wrist"])
