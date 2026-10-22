"""Phase 2 inspection workstation: live camera, visual inventory, commands and results.

Camera capture runs in CameraWorker, speech and classification in the Phase 1
workers. The window only renders state and never blocks on processing.
"""

from __future__ import annotations

import os
import sys
from typing import Any, Dict, Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QKeySequence, QPixmap, QShortcut
from PySide6.QtWidgets import (
    QAbstractItemView, QFrame, QHBoxLayout, QLabel, QMainWindow, QMessageBox, QPlainTextEdit,
    QPushButton, QScrollArea, QSizePolicy, QStackedWidget, QTableWidget, QTableWidgetItem,
    QComboBox, QVBoxLayout, QWidget,
)

from roboqc.commands.schema import StructuredCommand
from roboqc.speech.engines import DEFAULT_ENGINE, ENGINES, get_engine
from roboqc.ui.camera_worker import CameraWorker
from roboqc.ui.main_window import RECORD_ACTIVE_STYLE, STYLESHEET, _badge, _card, _label, _segmented, _wait_for
from roboqc.ui.workers import ClassificationWorker, ModelLoadWorker, RecordingWorker, TranscriptionWorker
from roboqc.vision.calibration import WorkspaceCalibration
from roboqc.vision.inspection import QUALITY_LABEL_ES
from roboqc.vision.quality import DEFECTIVE, GOOD, UNCERTAIN
from roboqc.vision.resolver import CLARIFY, INFO, NOT_FOUND, OK, UNSUPPORTED, VisionResult, resolve_command
from roboqc.vision.world_state import WorldSnapshot

IDLE, RECORDING, PROCESSING, CLASSIFYING = "idle", "recording", "processing", "classifying"

RESULT_STYLE = {
    OK: ("Correcto", "#15803d", "#dcfce7"),
    INFO: ("Informacion", "#1d4ed8", "#dbeafe"),
    CLARIFY: ("Aclaracion", "#a16207", "#fef9c3"),
    NOT_FOUND: ("No encontrado", "#b91c1c", "#fee2e2"),
    UNSUPPORTED: ("No ejecutable", "#475569", "#e2e8f0"),
}
QUALITY_STYLE = {
    GOOD: ("#15803d", "#dcfce7"),
    DEFECTIVE: ("#b91c1c", "#fee2e2"),
    UNCERTAIN: ("#a16207", "#fef9c3"),
}
MODEL_LABELS = {"baseline": "TF-IDF", "minilm": "MiniLM", "jepa": "JEPA", "industrial": "Industrial (Colab)"}
INVENTORY_COLUMNS = ["ID", "Tipo", "Color", "Forma", "Calidad", "Vista", "Movimiento", "Confianza"]


def _interpretation_text(cmd: StructuredCommand) -> str:
    lines = [f"Interpretacion: {cmd.status}"]
    for i, action in enumerate(cmd.actions, 1):
        params = [f"{k}={v}" for k, v in action.parameters.items() if v not in (None, "", "UNSPECIFIED")]
        line = f"{i}. {action.intent}"
        if params:
            line += " (" + ", ".join(params) + ")"
        if action.negated:
            line += " [negada]"
        lines.append(line)
    return "\n".join(lines)


def _set_badge(label: QLabel, text: str, fg: str, bg: str) -> None:
    label.setText(text)
    label.setStyleSheet(
        f"QLabel {{ color: {fg}; background: {bg}; border-radius: 10px; padding: 3px 10px; font-weight: 600; }}"
    )


class WorkstationWindow(QMainWindow):
    def __init__(
        self,
        source: Any,
        cfg: Dict[str, Any],
        calibration: Optional[WorkspaceCalibration] = None,
        model_dir: str = "models",
        preferred_model: Optional[str] = None,
        commands_enabled: bool = True,
    ):
        super().__init__()
        self.setWindowTitle("RoboQC - Estacion de inspeccion")
        self.resize(1280, 820)
        self.setMinimumSize(960, 640)
        self.setStyleSheet(STYLESHEET)

        self.cfg = cfg
        self.commands_enabled = commands_enabled  # False when commands are typed in another window
        self.model_dir = model_dir
        self.preferred_model = preferred_model
        self.engine = None
        self.rec_worker = None
        self.trans_worker = None
        self.cls_worker = None
        self.model_worker = None
        self._state = IDLE
        self._has_mic = False
        self._has_engine = False

        self.camera = CameraWorker(source, cfg, calibration)
        self.world = self.camera.pipeline.world
        from roboqc.vision.sorting import SortingCoordinator
        self.sorting = SortingCoordinator(cfg.get("industrial"), calibration)

        self._build_ui()
        self._populate_mics()
        self._populate_engines()
        self.camera.updated.connect(self._on_camera_update)
        self.camera.error.connect(self._on_camera_error)
        self.camera.start()
        if commands_enabled:
            self._load_model()
            self.statusBar().showMessage("Cargando modelo de voz e intenciones...")
        else:
            self.input_card.hide()
        self._refresh_controls()

    # ---- layout -------------------------------------------------------------
    def _build_ui(self) -> None:
        page = QWidget()
        page.setObjectName("page")
        self.setCentralWidget(page)
        root = QVBoxLayout(page)
        root.setContentsMargins(20, 16, 20, 12)
        root.setSpacing(14)

        header = QHBoxLayout()
        header.addWidget(_label("RoboQC", "title", wrap=False))
        header.addWidget(_label("Estacion de inspeccion", "muted", wrap=False))
        header.addStretch(1)
        self.fps_label = _label("", "muted", wrap=False)
        header.addWidget(self.fps_label)
        root.addLayout(header)

        body = QHBoxLayout()
        body.setSpacing(14)
        body.addLayout(self._build_left(), 3)
        body.addLayout(self._build_right(), 2)
        root.addLayout(body, 1)

        if self.commands_enabled:
            shortcut = QShortcut(QKeySequence("Ctrl+Return"), self)
            shortcut.activated.connect(self.run_input)

    def _build_left(self) -> QVBoxLayout:
        col = QVBoxLayout()
        col.setSpacing(14)

        cam_card, cam_layout = _card()
        cam_layout.addWidget(_label("Camara en vivo", "section", wrap=False))
        self.video = QLabel("Iniciando camara...")
        self.video.setAlignment(Qt.AlignCenter)
        self.video.setMinimumSize(480, 320)
        self.video.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.video.setStyleSheet("background: #0f172a; color: #e2e8f0; border-radius: 10px;")
        cam_layout.addWidget(self.video, 1)
        col.addWidget(cam_card, 3)

        inv_card, inv_layout = _card()
        inv_layout.addWidget(_label("Inventario visual (clic para resaltar)", "section", wrap=False))
        self.table = QTableWidget(0, len(INVENTORY_COLUMNS))
        self.table.setHorizontalHeaderLabels(INVENTORY_COLUMNS)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.itemSelectionChanged.connect(self._on_table_selection)
        inv_layout.addWidget(self.table, 1)
        col.addWidget(inv_card, 2)
        return col

    def _build_right(self) -> QVBoxLayout:
        col = QVBoxLayout()
        col.setSpacing(14)
        self.input_card = self._build_input_card()
        col.addWidget(self.input_card)
        col.addWidget(self._build_summary_card())
        col.addWidget(self._build_result_card(), 1)
        return col

    def _build_input_card(self) -> QFrame:
        card, layout = _card()
        head = QHBoxLayout()
        head.addWidget(_label("Instruccion", "section", wrap=False))
        head.addStretch(1)
        modes, self.mode_group = _segmented(("Voz", "Texto"))
        head.addLayout(modes)
        layout.addLayout(head)

        self.pages = QStackedWidget()
        self.pages.addWidget(self._build_voice_tab())
        self.pages.addWidget(self._build_text_tab())
        self.mode_group.idClicked.connect(self.pages.setCurrentIndex)
        layout.addWidget(self.pages, 1)

        self.btn_run = QPushButton("Ejecutar  (Ctrl+Enter)")
        self.btn_run.clicked.connect(self.run_input)
        layout.addWidget(self.btn_run)
        return card

    def _build_voice_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        row = QHBoxLayout()
        self.engine_combo = QComboBox()
        self.engine_combo.setToolTip("Motor de reconocimiento de voz.")
        row.addWidget(self.engine_combo)
        self.mic_combo = QComboBox()
        row.addWidget(self.mic_combo, 1)
        self.btn_record = QPushButton("Grabar")
        self.btn_record.setMinimumWidth(110)
        self.btn_record.clicked.connect(self.toggle_recording)
        row.addWidget(self.btn_record)
        layout.addLayout(row)
        self.trans_edit = QPlainTextEdit()
        self.trans_edit.setPlaceholderText("La transcripcion aparecera aqui. Puedes corregirla antes de ejecutar.")
        layout.addWidget(self.trans_edit, 1)
        return tab

    def _build_text_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        self.manual_edit = QPlainTextEdit()
        self.manual_edit.setPlaceholderText("Ejemplo: ¿Cuántas piezas rojas hay?")
        layout.addWidget(self.manual_edit, 1)
        return tab

    def _build_summary_card(self) -> QFrame:
        card, layout = _card()
        layout.addWidget(_label("Resumen de inspeccion", "section", wrap=False))
        row = QHBoxLayout()
        row.setSpacing(10)
        self.stat_labels: Dict[str, QLabel] = {}
        for key, title in (("visible", "Visibles"), ("GOOD", "Buenas"),
                           ("DEFECTIVE", "Defectuosas"), ("UNCERTAIN", "Inciertas")):
            box, box_layout = _card("subcard")
            number = _label("0", "value", wrap=False)
            number.setStyleSheet("font-size: 22px; font-weight: 700;")
            box_layout.addWidget(_label(title, "muted", wrap=False))
            box_layout.addWidget(number)
            row.addWidget(box, 1)
            self.stat_labels[key] = number
        layout.addLayout(row)
        return card

    def _build_result_card(self) -> QFrame:
        card, layout = _card()
        head = QHBoxLayout()
        head.addWidget(_label("Resultado", "section", wrap=False))
        head.addStretch(1)
        self.result_badge = _badge("Sin comando", "#475569", "#e2e8f0")
        head.addWidget(self.result_badge)
        layout.addLayout(head)
        self.interpretation = _label("", "muted")
        layout.addWidget(self.interpretation)
        self.result_text = _label("Graba o escribe una instruccion y pulsa Ejecutar.")
        self.result_text.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setWidget(self.result_text)
        # Explicit colors: the dark system palette otherwise leaves dark text on a dark area.
        self.result_text.setStyleSheet("QLabel { background: #ffffff; color: #0f172a; padding: 6px; }")
        scroll.viewport().setStyleSheet("background: #ffffff;")
        layout.addWidget(scroll, 1)
        self.plan_json = QPlainTextEdit()
        self.plan_json.setReadOnly(True)
        self.plan_json.setPlaceholderText("Reglas y plan de clasificacion industrial")
        self.plan_json.setMaximumHeight(200)
        layout.addWidget(self.plan_json)
        self.simulate_button = QPushButton("Simular un paso")
        self.simulate_button.clicked.connect(self._simulate_step)
        layout.addWidget(self.simulate_button)
        return card

    # ---- camera -------------------------------------------------------------
    def _on_camera_update(self, image, snapshot: WorldSnapshot) -> None:
        self.video.setPixmap(QPixmap.fromImage(image).scaled(
            self.video.size(), Qt.KeepAspectRatio, Qt.FastTransformation))
        self.fps_label.setText(f"Camara {self.camera.fps:.1f} FPS | {self.camera.latency_ms:.1f} ms")
        self._update_inventory(snapshot)
        self._update_summary(snapshot)
        self._update_plan(snapshot)

    def _update_plan(self, snapshot):
        import json
        plan = self.sorting.build(snapshot)
        payload = {"rules": self.sorting.dialogue.active, "plan": plan, "execution": "simulation_only"}
        text = json.dumps(payload, ensure_ascii=False, indent=2)
        if self.plan_json.toPlainText() != text:
            self.plan_json.setPlainText(text)
        self.simulate_button.setEnabled(any(i["status"] == "ready_for_simulation" for i in plan))

    def _simulate_step(self):
        import json
        response = self.sorting.simulate_one(self.world.snapshot())
        self.result_text.setText(json.dumps(response, ensure_ascii=False, indent=2))
        self._update_plan(self.world.snapshot())

    def _on_camera_error(self, message: str) -> None:
        self.video.setPixmap(QPixmap())
        self.video.setText(f"Camara no disponible\n\n{message}")
        self.statusBar().showMessage(message)

    def _update_inventory(self, snapshot: WorldSnapshot) -> None:
        objects = snapshot.objects
        self.table.blockSignals(True)
        self.table.setRowCount(len(objects))
        self.table.clearSelection()  # selection is restored from the world state below
        for row, obj in enumerate(objects):
            values = [
                obj.object_id, obj.object_type, obj.color or "?", obj.shape,
                QUALITY_LABEL_ES.get(obj.quality, obj.quality), obj.detection_status,
                obj.motion, f"{obj.confidence:.0%}",
            ]
            for col, text in enumerate(values):
                item = self.table.item(row, col)
                if item is None:
                    item = QTableWidgetItem()
                    self.table.setItem(row, col, item)
                item.setText(text)
                if col == 4:
                    fg, _ = QUALITY_STYLE.get(obj.quality, ("#1e293b", ""))
                    item.setForeground(QColor(fg))
                if obj.selected:
                    item.setSelected(True)
        self.table.blockSignals(False)

    def _on_table_selection(self) -> None:
        rows = {index.row() for index in self.table.selectedIndexes()}
        ids = [self.table.item(row, 0).text() for row in sorted(rows) if self.table.item(row, 0)]
        self.world.set_selection(ids)

    def _update_summary(self, snapshot: WorldSnapshot) -> None:
        counts = snapshot.counts()
        self.stat_labels["visible"].setText(str(counts["visible"]))
        self.stat_labels["GOOD"].setText(str(counts["GOOD"]))
        self.stat_labels["DEFECTIVE"].setText(str(counts["DEFECTIVE"]))
        self.stat_labels["UNCERTAIN"].setText(str(counts["UNCERTAIN"]))

    # ---- voice --------------------------------------------------------------
    def _load_model(self) -> None:
        self.model_worker = ModelLoadWorker(self.model_dir, self.preferred_model)
        self.model_worker.loaded.connect(self._on_model_loaded)
        self.model_worker.error.connect(self._on_model_error)
        self.model_worker.start()

    def _on_model_loaded(self, engine) -> None:
        self.engine = engine
        self.statusBar().showMessage(f"Modelo {MODEL_LABELS.get(engine.kind, engine.kind)} listo.")

    def _on_model_error(self, message: str) -> None:
        self.statusBar().showMessage(f"Sin modelo de intenciones ({message}). Se usan solo reglas.")

    def _populate_mics(self) -> None:
        try:
            from roboqc.speech.recorder import list_devices
            devices = list_devices()
        except Exception:
            devices = []
        self.mic_combo.clear()
        for d in devices:
            self.mic_combo.addItem(f"[{d['index']}] {' '.join(str(d['name']).split())}", d["index"])
        if not devices:
            self.mic_combo.addItem("Sin microfono", None)
        self._has_mic = bool(devices)

    def _populate_engines(self) -> None:
        self.engine_combo.clear()
        available = {name: engine.available() for name, engine in ENGINES.items()}
        for name, engine in ENGINES.items():
            label = engine.label if available[name] else f"{engine.label} (no instalado)"
            self.engine_combo.addItem(label, name)
            self.engine_combo.model().item(self.engine_combo.count() - 1).setEnabled(available[name])
        self._has_engine = any(available.values())
        if self._has_engine:
            preferred = DEFAULT_ENGINE if available[DEFAULT_ENGINE] else next(n for n, ok in available.items() if ok)
            self.engine_combo.setCurrentIndex(self.engine_combo.findData(preferred))

    def _set_state(self, state: str, message: str) -> None:
        self._state = state
        self.statusBar().showMessage(message)
        self._refresh_controls()

    def _refresh_controls(self) -> None:
        idle = self._state == IDLE
        recording = self._state == RECORDING
        self.btn_record.setEnabled(self._has_mic and self._has_engine and (idle or recording))
        self.btn_record.setText("Detener" if recording else "Grabar")
        self.btn_record.setStyleSheet(RECORD_ACTIVE_STYLE if recording else "")
        self.engine_combo.setEnabled(idle)
        self.mic_combo.setEnabled(idle)
        self.btn_run.setEnabled(idle)

    def toggle_recording(self) -> None:
        if self._state == RECORDING:
            self.rec_worker.stop()
            self._set_state(PROCESSING, "Procesando audio...")
            return
        _wait_for(self.rec_worker)
        self.rec_worker = RecordingWorker(device=self.mic_combo.currentData())
        self.rec_worker.recorded.connect(self._on_recorded)
        self.rec_worker.error.connect(self._on_voice_error)
        self.rec_worker.start()
        self._set_state(RECORDING, "Grabando... pulsa Detener cuando termines.")

    def _on_recorded(self, pcm_bytes: bytes, samplerate: int) -> None:
        if not pcm_bytes:
            self._set_state(IDLE, "No se capturo audio. Revisa el microfono.")
            return
        _wait_for(self.trans_worker)
        engine = get_engine(self.engine_combo.currentData())
        self.trans_worker = TranscriptionWorker(engine, pcm_bytes, samplerate)
        self.trans_worker.transcribed.connect(self._on_transcribed)
        self.trans_worker.error.connect(self._on_voice_error)
        self.trans_worker.start()
        self._set_state(PROCESSING, "Transcribiendo...")

    def _on_transcribed(self, text: str, latency: float) -> None:
        self.trans_edit.setPlainText(text)
        self.pages.setCurrentIndex(0)
        self.mode_group.button(0).setChecked(True)
        if not text.strip():
            self._set_state(IDLE, "No se reconocio ninguna palabra.")
            return
        self._set_state(IDLE, f"Transcripcion lista ({latency:.1f} s). Revisa el texto y pulsa Ejecutar.")

    def _on_voice_error(self, message: str) -> None:
        self._set_state(IDLE, "Error de voz.")
        QMessageBox.critical(self, "Error de voz", message)

    # ---- commands -----------------------------------------------------------
    def run_input(self) -> None:
        if self._state != IDLE:
            return
        texts = {"voice": self.trans_edit.toPlainText().strip(), "text": self.manual_edit.toPlainText().strip()}
        order = ("voice", "text") if self.pages.currentIndex() == 0 else ("text", "voice")
        source = next((s for s in order if texts[s]), None)
        if source is None:
            self.statusBar().showMessage("Escribe o graba una instruccion.")
            return
        _wait_for(self.cls_worker)
        self.cls_worker = ClassificationWorker(texts[source], source, self.engine, dialogue=self.sorting.dialogue)
        self.cls_worker.classified.connect(self._on_classified)
        self.cls_worker.error.connect(self._on_classify_error)
        self.cls_worker.start()
        self._set_state(CLASSIFYING, "Interpretando...")

    def show_result(self, cmd: StructuredCommand) -> VisionResult:
        """Resolve a command against the live world state and show it in the result panel."""
        envelope = next((a.parameters["industrial"] for a in cmd.actions if "industrial" in a.parameters), None)
        if envelope is not None:
            import json
            self.sorting.accept(envelope)
            message = envelope["clarification"] or json.dumps(envelope, ensure_ascii=False, indent=2)
            result = VisionResult(status=OK if envelope["status"] == "valid" else CLARIFY, messages=[message])
            self._update_plan(self.world.snapshot())
        else:
            result = resolve_command(cmd, self.world)
        self._render_result(cmd, result)
        return result

    def _on_classified(self, cmd: StructuredCommand, topk) -> None:
        result = self.show_result(cmd)
        self._set_state(IDLE, f"Comando resuelto: {RESULT_STYLE[result.status][0].lower()}.")

    def _on_classify_error(self, message: str) -> None:
        self._set_state(IDLE, "Error al interpretar la instruccion.")
        self.result_text.setText(f"Error al interpretar la instruccion: {message}")

    def _render_result(self, cmd: StructuredCommand, result: VisionResult) -> None:
        label, fg, bg = RESULT_STYLE.get(result.status, (result.status, "#475569", "#e2e8f0"))
        _set_badge(self.result_badge, label, fg, bg)
        envelope = next((a.parameters["industrial"] for a in cmd.actions if "industrial" in a.parameters), None)
        if envelope is not None:
            command = envelope["command"] or {}
            self.interpretation.setText(f"Clasificacion industrial: {envelope['status']} | {command.get('intent','-')} | {command.get('mode','-')}")
        else:
            self.interpretation.setText(_interpretation_text(cmd))
        self.result_text.setText(result.text or "Sin respuesta.")

    def closeEvent(self, event) -> None:
        self.camera.stop()
        self.camera.wait(3000)
        if self._state == RECORDING:
            self.rec_worker.stop()
        for worker in (self.rec_worker, self.trans_worker, self.cls_worker, self.model_worker):
            _wait_for(worker)
        super().closeEvent(event)
