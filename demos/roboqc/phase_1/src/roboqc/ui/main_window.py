"""PySide6 main window for RoboQC: Spanish single-column UI for command analysis."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from PySide6.QtCore import Qt
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QButtonGroup, QComboBox, QFrame, QGridLayout, QHBoxLayout, QLabel, QMainWindow, QMessageBox,
    QPlainTextEdit, QPushButton, QProgressBar, QScrollArea, QStackedWidget,
    QSizePolicy, QVBoxLayout, QWidget,
)

from roboqc.commands.schema import StructuredCommand
from roboqc.commands.taxonomy import INTENT_DESCRIPTIONS_ES, SAFETY_CRITICAL
from roboqc.intent.inference import available_models
from roboqc.speech.engines import DEFAULT_ENGINE, ENGINES, get_engine
from roboqc.ui.workers import ClassificationWorker, ModelLoadWorker, RecordingWorker, TranscriptionWorker


STYLESHEET = """
QWidget#page { background: #f1f5f9; }
QWidget { font-family: "Segoe UI", Arial, sans-serif; font-size: 13px; color: #1e293b; }
QFrame#card { background: #ffffff; border: 1px solid #e2e8f0; border-radius: 14px; }
QFrame#subcard { background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 10px; }
QScrollArea { background: transparent; border: none; }
QWidget#resultHost, QWidget#qt_scrollarea_viewport { background: transparent; }
QLabel#title { font-size: 22px; font-weight: 700; color: #0f172a; }
QLabel#muted { color: #64748b; }
QLabel#section { font-size: 14px; font-weight: 700; color: #0f172a; }
QLabel#value { font-weight: 600; color: #0f172a; }
QPushButton { background: #2563eb; color: #ffffff; border: none; border-radius: 8px; padding: 9px 16px; font-weight: 600; }
QPushButton:hover { background: #1d4ed8; }
QPushButton:disabled { background: #cbd5e1; color: #64748b; }
QPlainTextEdit { background: #ffffff; border: 1px solid #cbd5e1; border-radius: 8px; padding: 6px 8px; }
QPlainTextEdit:focus { border: 1px solid #2563eb; }
QComboBox { background: #ffffff; border: 1px solid #cbd5e1; border-radius: 8px; padding: 7px 34px 7px 10px; min-height: 20px; }
QComboBox:hover { border-color: #94a3b8; }
QComboBox:focus, QComboBox:on { border: 1px solid #2563eb; }
QComboBox:disabled { background: #f1f5f9; color: #94a3b8; }
QComboBox::drop-down { subcontrol-origin: padding; subcontrol-position: center right; width: 30px; border: none; }
QComboBox QAbstractItemView { background: #ffffff; border: 1px solid #cbd5e1; border-radius: 8px; padding: 4px; outline: none; selection-background-color: #dbeafe; selection-color: #1e3a8a; }
QComboBox QAbstractItemView::item { min-height: 24px; padding: 4px 8px; border-radius: 6px; }
QComboBox QAbstractItemView::item:disabled { color: #94a3b8; }
QTableWidget { background: #ffffff; border: 1px solid #e2e8f0; border-radius: 8px; gridline-color: #f1f5f9; }
QHeaderView::section { background: #f8fafc; border: none; border-bottom: 1px solid #e2e8f0; padding: 6px; font-weight: 600; color: #475569; }
QProgressBar { background: #e2e8f0; border: none; border-radius: 5px; }
QProgressBar::chunk { background: #2563eb; border-radius: 5px; }
QPushButton#segment { background: #f1f5f9; color: #475569; border: 1px solid #cbd5e1; border-radius: 8px; padding: 6px 16px; font-weight: 600; }
QPushButton#segment:hover { background: #e2e8f0; }
QPushButton#segment:checked { background: #2563eb; color: #ffffff; border-color: #2563eb; }
"""

ICON_DIR = os.path.join(os.path.dirname(__file__), "icons").replace("\\", "/")
STYLESHEET += f"""
QComboBox::down-arrow {{ image: url({ICON_DIR}/chevron_down.svg); width: 12px; height: 8px; }}
QComboBox::down-arrow:disabled {{ image: url({ICON_DIR}/chevron_down_disabled.svg); }}
"""

RECORD_ACTIVE_STYLE ="QPushButton { background: #dc2626; } QPushButton:hover { background: #b91c1c; }"

IDLE, RECORDING, PROCESSING, CLASSIFYING, LOADING = "idle", "recording", "processing", "classifying", "loading"
RULES_ONLY = "rules"

STATUS_STYLE = {
    "VALID": ("Valido", "#15803d", "#dcfce7"),
    "AMBIGUOUS": ("Ambiguo", "#a16207", "#fef9c3"),
    "UNKNOWN": ("No reconocido", "#475569", "#e2e8f0"),
    "INVALID": ("Invalido", "#b91c1c", "#fee2e2"),
}
MODEL_LABELS = {"baseline": "TF-IDF", "minilm": "MiniLM", "jepa": "JEPA", "industrial": "Industrial (Colab)"}
SOURCE_LABELS = {"voice": "Voz", "text": "Texto"}
PARAM_LABELS = {
    "object_type": "Tipo de objeto",
    "object_color": "Color del objeto",
    "quantity": "Cantidad",
    "object_quality": "Calidad",
    "destination_type": "Tipo de destino",
    "destination_color": "Color del destino",
    "destination_role": "Papel del destino",
}
ISSUE_LABELS = {"correccion_contextual": "Posible correccion contextual: revisa si la instruccion sustituye a una anterior."}
UNSET_VALUES = (None, "", "UNSPECIFIED")
ROW_SPACING = 12
INTENT_CARD_WIDTH = 200
ACTIONS_PER_VIEW = 3
ACTION_MIN_WIDTH = 240


def _wait_for(worker) -> None:
    if worker is not None:
        worker.wait()


def _clear(layout) -> None:
    while layout.count():
        item = layout.takeAt(0)
        if item.widget() is not None:
            item.widget().deleteLater()
        elif item.layout() is not None:
            _clear(item.layout())


def _card(name: str = "card") -> tuple[QFrame, QVBoxLayout]:
    frame = QFrame()
    frame.setObjectName(name)
    layout = QVBoxLayout(frame)
    layout.setContentsMargins(16, 14, 16, 14)
    layout.setSpacing(10)
    return frame, layout


def _label(text: str, name: str | None = None, wrap: bool = True) -> QLabel:
    lbl = QLabel(text)
    lbl.setTextFormat(Qt.PlainText)
    lbl.setTextInteractionFlags(Qt.TextSelectableByMouse)
    lbl.setWordWrap(wrap)
    if name:
        lbl.setObjectName(name)
    return lbl


def _badge(text: str, fg: str, bg: str) -> QLabel:
    lbl = QLabel(text)
    lbl.setStyleSheet(
        f"QLabel {{ color: {fg}; background: {bg}; border-radius: 10px; padding: 3px 10px; font-weight: 600; }}"
    )
    return lbl


def _meter(fraction: float, text: str) -> QWidget:
    holder = QWidget()
    row = QHBoxLayout(holder)
    row.setContentsMargins(6, 0, 6, 0)
    bar = QProgressBar()
    bar.setRange(0, 1000)
    bar.setValue(round(max(0.0, min(1.0, fraction)) * 1000))
    bar.setTextVisible(False)
    bar.setFixedHeight(10)
    bar.setMinimumWidth(60)
    row.addWidget(bar, 1)
    row.addWidget(_label(text, "muted", wrap=False))
    return holder


class _CardRow(QScrollArea):
    """Horizontal row of cards that scrolls left and right.

    With per_view set, the cards share the visible width (at most per_view at once);
    otherwise every card keeps card_width.
    """

    def __init__(self):
        super().__init__()
        self.setWidgetResizable(False)
        self.setFrameShape(QFrame.NoFrame)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
        self._host = QWidget()
        self._host.setObjectName("resultHost")
        self._row_layout = QHBoxLayout(self._host)
        self._row_layout.setContentsMargins(0, 0, 0, 0)
        self._row_layout.setSpacing(ROW_SPACING)
        self._row_layout.setAlignment(Qt.AlignTop)
        self.setWidget(self._host)
        self._cards = []
        self._per_view = None
        self._card_width = INTENT_CARD_WIDTH

    def set_cards(self, cards, per_view=None, card_width=INTENT_CARD_WIDTH):
        self._cards = cards
        self._per_view = per_view
        self._card_width = card_width
        for card in cards:
            self._row_layout.addWidget(card)
        self.relayout()

    def relayout(self):
        if not self._cards:
            return
        if self._per_view:
            count = self._per_view
            width = max(ACTION_MIN_WIDTH, (self.viewport().width() - ROW_SPACING * (count - 1)) // count)
        else:
            width = self._card_width
        for card in self._cards:
            card.setFixedWidth(width)
        height = max(card.heightForWidth(width) if card.hasHeightForWidth() else card.sizeHint().height()
                     for card in self._cards)
        for card in self._cards:
            card.setFixedHeight(height)
        total = len(self._cards) * width + ROW_SPACING * (len(self._cards) - 1)
        self._host.setFixedSize(total, height)
        self.setFixedHeight(height + self.horizontalScrollBar().sizeHint().height() + 2)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.relayout()

    def wheelEvent(self, event):
        bar = self.horizontalScrollBar()
        if bar.maximum() == 0:
            event.ignore()
            return
        delta = event.angleDelta().y() or event.angleDelta().x()
        bar.setValue(bar.value() - delta)
        event.accept()


def _info_panel(title: str, value: str, note: str | None = None) -> QFrame:
    panel, layout = _card("subcard")
    layout.addWidget(_label(title, "muted"))
    layout.addWidget(_label(value, "value"))
    if note:
        layout.addWidget(_label(note, "muted"))
    return panel


def _intent_card(intent: str, prob: float, selected: bool) -> QFrame:
    card, layout = _card("subcard")
    layout.addWidget(_label(intent, "value" if selected else None))
    layout.addWidget(_label(INTENT_DESCRIPTIONS_ES.get(intent, ""), "muted"))
    layout.addWidget(_meter(prob, f"{prob * 100:.1f} %"))
    return card


def _format_value(value) -> str:
    return "-" if value in UNSET_VALUES else str(value)


def _segmented(labels) -> tuple[QHBoxLayout, QButtonGroup]:
    row = QHBoxLayout()
    row.setSpacing(6)
    group = QButtonGroup()
    for i, text in enumerate(labels):
        btn = QPushButton(text)
        btn.setObjectName("segment")
        btn.setCheckable(True)
        group.addButton(btn, i)
        row.addWidget(btn)
    group.button(0).setChecked(True)
    return row, group


class MainWindow(QMainWindow):
    def __init__(self, model_dir: str = "models", preferred_model=None):
        super().__init__()
        self.setWindowTitle("RoboQC")
        self.resize(900, 860)
        self.setMinimumSize(640, 640)
        self.setStyleSheet(STYLESHEET)

        self.model_dir = model_dir
        self.preferred_model = preferred_model
        self.engine = None
        self.engine_error = ""

        self.rec_worker = None
        self.trans_worker = None
        self.cls_worker = None
        self.model_worker = None
        self._state = IDLE
        self._has_mic = False
        self._has_engine = False

        self._build_ui()
        self._populate_mics()
        self._populate_engines()
        self._show_placeholder("Introduce o graba una instruccion y pulsa Clasificar.")
        if self.engine is None:
            self.statusBar().showMessage(f"Modelo no disponible ({self.engine_error}). Se usa el clasificador por reglas.")
        else:
            self.statusBar().showMessage("Listo.")
        self._refresh_controls()
        self._load_engine()

    def _load_engine(self):
        self.model_worker = ModelLoadWorker(self.model_dir, self.preferred_model)
        self.model_worker.loaded.connect(self._on_model_loaded)
        self.model_worker.error.connect(self._on_initial_model_error)
        self.model_worker.start()
        self._set_state(LOADING, "Cargando modelo de intenciones...")

    def _on_initial_model_error(self, message):
        self.engine_error = message
        self._set_state(IDLE, f"Modelo no disponible: {message}. Se usan reglas.")

    def _build_ui(self):
        page = QWidget()
        page.setObjectName("page")
        self.setCentralWidget(page)
        root = QVBoxLayout(page)
        root.setContentsMargins(20, 16, 20, 12)
        root.setSpacing(14)
        root.addLayout(self._build_header())

        root.addWidget(self._build_input_panel(), 1)
        root.addWidget(self._build_result_panel(), 2)

    def _build_header(self) -> QHBoxLayout:
        row = QHBoxLayout()
        row.addWidget(_label("RoboQC", "title", wrap=False), 1)
        return row

    def _build_model_combo(self) -> QComboBox:
        self.model_combo = QComboBox()
        self.model_combo.setToolTip("Clasificador de intenciones. Solo reglas no calcula probabilidades.")
        for name, ok in available_models(self.model_dir).items():
            label = MODEL_LABELS[name] if ok else f"{MODEL_LABELS[name]} (no instalado)"
            self.model_combo.addItem(label, name)
            self.model_combo.model().item(self.model_combo.count() - 1).setEnabled(ok)
        self.model_combo.addItem("Solo reglas", RULES_ONLY)
        self._select_model(self.engine.kind if self.engine is not None else RULES_ONLY)
        self.model_combo.activated.connect(self._on_model_activated)
        return self.model_combo

    def _select_model(self, name: str):
        self.model_combo.setCurrentIndex(self.model_combo.findData(name))

    def _on_model_activated(self, index: int):
        name = self.model_combo.itemData(index)
        if name == RULES_ONLY:
            self.engine = None
            self._set_state(IDLE, "Clasificador por reglas activo. No calcula probabilidades.")
            return
        _wait_for(self.model_worker)
        self.model_worker = ModelLoadWorker(self.model_dir, name)
        self.model_worker.loaded.connect(self._on_model_loaded)
        self.model_worker.error.connect(self._on_model_error)
        self.model_worker.start()
        self._set_state(LOADING, f"Cargando modelo {MODEL_LABELS[name]}...")

    def _on_model_loaded(self, engine):
        self.engine = engine
        self._select_model(engine.kind)
        self._set_state(IDLE, f"Modelo {MODEL_LABELS[engine.kind]} listo.")

    def _on_model_error(self, msg):
        self._select_model(self.engine.kind if self.engine is not None else RULES_ONLY)
        self._set_state(IDLE, "No se pudo cargar el modelo. Se mantiene el anterior.")
        QMessageBox.critical(self, "Error al cargar el modelo", msg)

    def closeEvent(self, event):
        if self._state == RECORDING:
            self.rec_worker.stop()
        for worker in (self.rec_worker, self.trans_worker, self.cls_worker, self.model_worker):
            _wait_for(worker)
        super().closeEvent(event)

    def _build_input_panel(self) -> QFrame:
        card, layout = _card()
        head = QGridLayout()
        head.addWidget(_label("Instruccion", "section", wrap=False), 0, 0)
        modes, self.mode_group = _segmented(("Voz", "Texto"))
        head.addLayout(modes, 0, 1, Qt.AlignCenter)
        self.btn_classify = QPushButton("Clasificar  (Ctrl+Enter)")
        self.btn_classify.clicked.connect(self.classify_input)
        head.addWidget(self.btn_classify, 0, 2, Qt.AlignRight)
        head.setColumnStretch(0, 1)
        head.setColumnStretch(2, 1)
        layout.addLayout(head)
        shortcut = QShortcut(QKeySequence("Ctrl+Return"), self)
        shortcut.activated.connect(self.classify_input)

        self.pages = QStackedWidget()
        self.pages.addWidget(self._build_voice_tab())
        self.pages.addWidget(self._build_text_tab())
        self.mode_group.idClicked.connect(self.pages.setCurrentIndex)
        layout.addWidget(self.pages, 1)
        return card

    def _build_voice_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setSpacing(10)

        row = QHBoxLayout()
        self.engine_combo = QComboBox()
        self.engine_combo.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.engine_combo.setMinimumContentsLength(14)
        self.engine_combo.setToolTip("Motor de reconocimiento de voz. Los no instalados aparecen deshabilitados.")
        row.addWidget(self.engine_combo)
        self.mic_combo = QComboBox()
        self.mic_combo.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.mic_combo.setMinimumContentsLength(24)
        row.addWidget(self.mic_combo, 1)
        self.btn_record = QPushButton("Grabar")
        self.btn_record.setMinimumWidth(110)
        self.btn_record.clicked.connect(self.toggle_recording)
        row.addWidget(self.btn_record)
        layout.addLayout(row)

        self.trans_edit = QPlainTextEdit()
        self.trans_edit.setPlaceholderText("La transcripcion aparecera aqui. Puedes corregirla antes de clasificar.")
        layout.addWidget(self.trans_edit, 1)
        return tab

    def _build_text_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        self.manual_edit = QPlainTextEdit()
        self.manual_edit.setPlaceholderText("Escribe una instruccion, por ejemplo: pon la pieza roja en el destino azul")
        layout.addWidget(self.manual_edit, 1)
        return tab

    def _build_result_panel(self) -> QFrame:
        card, layout = _card()
        head = QHBoxLayout()
        head.addWidget(_label("Resultado", "section", wrap=False))
        head.addStretch(1)
        head.addWidget(_label("Modelo", "muted", wrap=False))
        head.addWidget(self._build_model_combo())
        head.addSpacing(12)
        tabs, self.result_group = _segmented(("Resumen", "Intenciones", "Acciones"))
        head.addLayout(tabs)
        layout.addLayout(head)

        self.result_pages = QStackedWidget()
        self.summary_layout = self._add_result_page()
        self.intents_layout = self._add_result_page()
        self.actions_layout = self._add_result_page()
        self.result_group.idClicked.connect(self.result_pages.setCurrentIndex)
        layout.addWidget(self.result_pages, 1)
        return card

    def _add_result_page(self) -> QVBoxLayout:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        host = QWidget()
        host.setObjectName("resultHost")
        page_layout = QVBoxLayout(host)
        page_layout.setContentsMargins(0, 0, 6, 0)
        page_layout.setSpacing(12)
        scroll.setWidget(host)
        self.result_pages.addWidget(scroll)
        return page_layout

    def _clear_results(self):
        for layout in (self.summary_layout, self.intents_layout, self.actions_layout):
            _clear(layout)

    def _populate_mics(self):
        try:
            from roboqc.speech.recorder import list_devices
            devices = list_devices()
        except Exception:
            devices = []
        self.mic_combo.clear()
        for d in devices:
            name = " ".join(str(d["name"]).split())
            self.mic_combo.addItem(f"[{d['index']}] {name}", d["index"])
            self.mic_combo.setItemData(self.mic_combo.count() - 1, name, Qt.ToolTipRole)
        if not devices:
            self.mic_combo.addItem("Sin microfono", None)
        self._has_mic = bool(devices)

    def _populate_engines(self):
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
        else:
            self.engine_combo.setCurrentIndex(-1)

    def _set_state(self, state: str, message: str):
        self._state = state
        self.statusBar().showMessage(message)
        self._refresh_controls()

    def _refresh_controls(self):
        idle = self._state == IDLE
        recording = self._state == RECORDING
        self.btn_record.setEnabled(self._has_mic and self._has_engine and (idle or recording))
        self.btn_record.setText("Detener" if recording else "Grabar")
        self.btn_record.setStyleSheet(RECORD_ACTIVE_STYLE if recording else "")
        self.mic_combo.setEnabled(idle)
        self.engine_combo.setEnabled(idle)
        self.model_combo.setEnabled(idle)
        self.btn_classify.setEnabled(idle)

    def toggle_recording(self):
        if self._state == RECORDING:
            self.rec_worker.stop()
            self._set_state(PROCESSING, "Procesando audio...")
            return
        _wait_for(self.rec_worker)
        self.rec_worker = RecordingWorker(device=self.mic_combo.currentData())
        self.rec_worker.recorded.connect(self._on_record_finished)
        self.rec_worker.error.connect(self._on_record_error)
        self.rec_worker.start()
        self._set_state(RECORDING, "Grabando... pulsa Detener cuando termines.")

    def _on_record_finished(self, pcm_bytes, samplerate):
        if not pcm_bytes:
            self._set_state(IDLE, "No se capturo audio. Revisa el microfono e intentalo de nuevo.")
            return
        _wait_for(self.trans_worker)
        engine = get_engine(self.engine_combo.currentData())
        self.trans_worker = TranscriptionWorker(engine, pcm_bytes, samplerate)
        self.trans_worker.transcribed.connect(self._on_trans_finished)
        self.trans_worker.error.connect(self._on_trans_error)
        self.trans_worker.start()
        if engine.name.startswith("whisper"):
            self._set_state(PROCESSING, "Transcribiendo... (la primera vez descarga el modelo de voz)")
        else:
            self._set_state(PROCESSING, "Transcribiendo...")

    def _on_record_error(self, msg):
        self._set_state(IDLE, "Error de grabacion.")
        QMessageBox.critical(self, "Error de grabacion", msg)

    def _on_trans_finished(self, text, latency):
        self.trans_edit.setPlainText(text)
        if not text.strip():
            self._set_state(IDLE, "No se reconocio ninguna palabra. Intentalo de nuevo.")
            return
        self.mode_group.button(0).setChecked(True)
        self.pages.setCurrentIndex(0)
        self._set_state(IDLE, f"Transcripcion lista ({latency:.1f} s). Revisa el texto y pulsa Clasificar.")

    def _on_trans_error(self, msg):
        self._set_state(IDLE, "Error de transcripcion.")
        QMessageBox.critical(self, "Error de transcripcion", msg)

    def classify_input(self):
        if self._state != IDLE:
            return
        texts = {"voice": self.trans_edit.toPlainText().strip(), "text": self.manual_edit.toPlainText().strip()}
        order = ("voice", "text") if self.pages.currentIndex() == 0 else ("text", "voice")
        source = next((s for s in order if texts[s]), None)
        if source is None:
            QMessageBox.warning(self, "Aviso", "Introduce o graba una instruccion.")
            return
        _wait_for(self.cls_worker)
        from roboqc.industrial.voice import Dialogue
        if not hasattr(self, "industrial_dialogue"):
            self.industrial_dialogue = Dialogue()
        self.cls_worker = ClassificationWorker(texts[source], source, self.engine, dialogue=self.industrial_dialogue)
        self.cls_worker.classified.connect(self._on_classify_finished)
        self.cls_worker.error.connect(self._on_classify_error)
        self.cls_worker.start()
        self._set_state(CLASSIFYING, "Clasificando...")

    def _on_classify_finished(self, cmd: StructuredCommand, topk):
        self._render_result(cmd, topk)
        self._set_state(IDLE, f"Clasificacion completada: {len(cmd.actions)} accion(es).")

    def _on_classify_error(self, msg):
        self._set_state(IDLE, "Error de clasificacion.")
        QMessageBox.critical(self, "Error de clasificacion", msg)

    def _show_placeholder(self, text: str):
        self._clear_results()
        lbl = _label(text, "muted")
        lbl.setAlignment(Qt.AlignCenter)
        self.summary_layout.addWidget(lbl)
        self.summary_layout.addStretch(1)

    def _render_result(self, cmd: StructuredCommand, topk):
        self._clear_results()
        self.summary_layout.addWidget(self._summary_box(cmd))
        industrial = next((a.parameters["industrial"] for a in cmd.actions if "industrial" in a.parameters), None)
        if industrial is not None:
            import json
            details = QPlainTextEdit(json.dumps(industrial, ensure_ascii=False, indent=2))
            details.setReadOnly(True)
            details.setMinimumHeight(240)
            self.summary_layout.addWidget(details)
        if cmd.unresolved_references or cmd.validation_errors:
            self.summary_layout.addWidget(self._issues_box(cmd))
        self.summary_layout.addStretch(1)

        if topk:
            selected = cmd.actions[0].intent if cmd.actions else None
            intents = _CardRow()
            intents.set_cards([_intent_card(intent, float(prob), intent == selected) for intent, prob in topk])
            self.intents_layout.addWidget(intents)
        else:
            self.intents_layout.addWidget(_label("No disponible: el clasificador por reglas no calcula probabilidades.", "muted"))
        self.intents_layout.addStretch(1)

        if cmd.actions:
            actions = _CardRow()
            actions.set_cards(
                [self._action_box(i, action) for i, action in enumerate(cmd.actions, 1)],
                per_view=ACTIONS_PER_VIEW,
            )
            self.actions_layout.addWidget(actions)
        else:
            self.actions_layout.addWidget(_label("No se ha reconocido ninguna accion.", "muted"))
        self.actions_layout.addStretch(1)

    def _summary_box(self, cmd: StructuredCommand) -> QFrame:
        box, layout = _card("subcard")
        head = QHBoxLayout()
        head.addWidget(_label("Estado del comando", "section"), 1)
        status_text, fg, bg = STATUS_STYLE.get(cmd.status, (cmd.status, "#475569", "#e2e8f0"))
        head.addWidget(_badge(status_text, fg, bg))
        layout.addLayout(head)

        normalized = cmd.normalized_text if cmd.normalized_text != cmd.transcription else None
        time_text = f"{cmd.processing_time_ms:.1f} ms" if cmd.processing_time_ms is not None else "-"
        panels = QHBoxLayout()
        panels.setSpacing(ROW_SPACING)
        panels.addWidget(_info_panel("Instruccion", cmd.transcription or "-", normalized and f"Normalizada: {normalized}"), 2)
        panels.addWidget(_info_panel("Origen", SOURCE_LABELS.get(cmd.source, cmd.source)), 1)
        panels.addWidget(_info_panel("Tiempo de proceso", time_text), 1)
        layout.addLayout(panels)
        return box

    def _action_box(self, index: int, action) -> QFrame:
        box, layout = _card("subcard")
        layout.addWidget(_label(f"Accion {index}: {action.intent}", "section"))
        layout.addWidget(_label(INTENT_DESCRIPTIONS_ES.get(action.intent, "Sin descripcion"), "muted"))

        flags = QHBoxLayout()
        if action.negated:
            flags.addWidget(_badge("Negada", "#b91c1c", "#fee2e2"))
        if action.requires_confirmation:
            flags.addWidget(_badge("Requiere confirmacion", "#a16207", "#fef9c3"))
        if action.intent in SAFETY_CRITICAL:
            flags.addWidget(_badge("Critica para la seguridad", "#b91c1c", "#fee2e2"))
        flags.addStretch(1)
        layout.addLayout(flags)

        if action.confidence is not None:
            pct = max(0.0, min(1.0, float(action.confidence)))
            layout.addWidget(_label("Confianza del modelo", "muted", wrap=False))
            layout.addWidget(_meter(pct, f"{pct * 100:.1f} %"))
        layout.addLayout(self._params_grid(action.parameters))
        return box

    def _params_grid(self, params: dict) -> QGridLayout:
        grid = QGridLayout()
        grid.setHorizontalSpacing(16)
        grid.setVerticalSpacing(6)
        for i, (key, text) in enumerate(PARAM_LABELS.items()):
            row, col = divmod(i, 2)
            grid.addWidget(_label(text, "muted"), row, col * 2)
            grid.addWidget(_label(_format_value(params.get(key)), "value"), row, col * 2 + 1)
        grid.setColumnStretch(1, 1)
        grid.setColumnStretch(3, 1)
        return grid

    def _issues_box(self, cmd: StructuredCommand) -> QFrame:
        box, layout = _card("subcard")
        has_errors = bool(cmd.validation_errors)
        box.setStyleSheet(
            "QFrame#subcard { background: %s; border-color: %s; }"
            % (("#fef2f2", "#fecaca") if has_errors else ("#fffbeb", "#fde68a"))
        )
        lines = [ISSUE_LABELS.get(r, f"Referencia no resuelta: {r}") for r in cmd.unresolved_references]
        lines += cmd.validation_errors
        layout.addWidget(_label("Avisos y errores", "section"))
        layout.addWidget(_label("\n".join(f"- {line}" for line in lines)))
        return box

