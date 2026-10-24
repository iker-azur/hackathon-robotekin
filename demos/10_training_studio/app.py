"""Desktop studio for bin-picking ACT imitation learning, training and results."""

from __future__ import annotations

import json
import os
import random
import re
import sys
from datetime import datetime
from pathlib import Path

from PySide6.QtCore import QPointF, QProcess, QProcessEnvironment, QTimer, Qt, QUrl
from PySide6.QtGui import QColor, QDesktopServices, QFont, QPainter, QPainterPath, QPen, QPixmap, QTextCursor
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QTabWidget,
    QScrollArea,
    QPushButton,
    QSpinBox,
    QStackedWidget,
    QToolButton,
    QVBoxLayout,
    QWidget,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
BINPICK_DIR = REPO_ROOT / "demos" / "09_voice_bin_picking"
TRAIN_SCRIPT = BINPICK_DIR / "tools" / "train_act.py"
EVALUATE_SCRIPT = BINPICK_DIR / "tools" / "eval_sorting.py"
OFFLINE_EVAL_SCRIPT = BINPICK_DIR / "tools" / "offline_eval_policy.py"
HF_DATASET_HELPER = Path(__file__).resolve().with_name("huggingface_datasets.py")
DATA_DIR = BINPICK_DIR / "data"
OUTPUTS_DIR = BINPICK_DIR / "outputs"
BINPICK_RESULTS_DIR = BINPICK_DIR / "results"
BINPICK_OUTPUTS_DIR = BINPICK_DIR / "outputs"
EVALUATIONS_DIR = BINPICK_RESULTS_DIR
_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
OUTPUT_TASKS = ("Pick-up", "Up-down")


class KeyboardOnlyComboBox(QComboBox):
    """Keep Setup choices keyboard-editable without mouse dropdown changes."""

    def mousePressEvent(self, event) -> None:
        self.setFocus(Qt.FocusReason.MouseFocusReason)
        event.accept()

    def wheelEvent(self, event) -> None:
        event.ignore()


class KeyboardOnlyCheckBox(QCheckBox):
    """Allow Space/keyboard toggles while preventing accidental mouse toggles."""

    def mousePressEvent(self, event) -> None:
        self.setFocus(Qt.FocusReason.MouseFocusReason)
        event.accept()

    def mouseReleaseEvent(self, event) -> None:
        event.accept()


class KeyboardOnlyLineEdit(QLineEdit):
    """Let mouse clicks focus a value without placing/editing its text."""

    def mousePressEvent(self, event) -> None:
        self.setFocus(Qt.FocusReason.MouseFocusReason)
        event.accept()

    def mouseDoubleClickEvent(self, event) -> None:
        self.setFocus(Qt.FocusReason.MouseFocusReason)
        event.accept()

    def contextMenuEvent(self, event) -> None:
        event.accept()


class KeyboardOnlySpinBox(QSpinBox):
    def mousePressEvent(self, event) -> None:
        self.setFocus(Qt.FocusReason.MouseFocusReason)
        event.accept()

    def mouseReleaseEvent(self, event) -> None:
        event.accept()

    def wheelEvent(self, event) -> None:
        event.ignore()


class KeyboardOnlyDoubleSpinBox(QDoubleSpinBox):
    def mousePressEvent(self, event) -> None:
        self.setFocus(Qt.FocusReason.MouseFocusReason)
        event.accept()

    def mouseReleaseEvent(self, event) -> None:
        event.accept()

    def wheelEvent(self, event) -> None:
        event.ignore()


class LiveLossChart(QWidget):
    """Small dependency-free line chart for train and validation loss."""

    SERIES_COLORS = {"Training loss": QColor("#198754"), "Validation loss": QColor("#e07a24")}

    def __init__(self) -> None:
        super().__init__()
        self.series: dict[str, list[tuple[int, float]]] = {key: [] for key in self.SERIES_COLORS}
        self.setMinimumHeight(190)

    def clear(self) -> None:
        for values in self.series.values():
            values.clear()
        self.update()

    def add_value(self, name: str, step: int, value: float) -> None:
        points = self.series[name]
        points.append((step, value))
        del points[:-400]
        self.update()

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor("#ffffff"))
        area = self.rect().adjusted(52, 20, -18, -35)
        all_points = [point for series in self.series.values() for point in series]
        if not all_points:
            painter.setPen(QColor("#7a808a"))
            painter.drawText(area, Qt.AlignmentFlag.AlignCenter, "Loss curves will appear as training logs arrive")
            return
        min_x = min(x for x, _ in all_points)
        max_x = max(x for x, _ in all_points)
        max_y = max(y for _, y in all_points)
        min_y = min(0.0, min(y for _, y in all_points))
        if max_y <= min_y:
            max_y = min_y + 1.0
        max_y = max_y + (max_y - min_y) * 0.08
        if max_x <= min_x:
            max_x = min_x + 1
        painter.setPen(QPen(QColor("#e7e9ed"), 1))
        for index in range(5):
            y = area.top() + area.height() * index / 4
            painter.drawLine(QPointF(area.left(), y), QPointF(area.right(), y))
        painter.setPen(QColor("#747b85"))
        painter.drawText(4, area.top() + 5, f"{max_y:.3g}")
        painter.drawText(4, area.bottom(), f"{min_y:.3g}")
        painter.drawText(area.left(), self.height() - 8, f"Step {min_x}")
        painter.drawText(area.right() - 70, self.height() - 8, f"Step {max_x}")
        for name, points in self.series.items():
            if not points:
                continue
            path = QPainterPath()
            for index, (x, y) in enumerate(points):
                px = area.left() + (x - min_x) / (max_x - min_x) * area.width()
                py = area.bottom() - (y - min_y) / (max_y - min_y) * area.height()
                (path.moveTo if index == 0 else path.lineTo)(QPointF(px, py))
            painter.setPen(QPen(self.SERIES_COLORS[name], 2))
            painter.drawPath(path)
        legend_x = area.left()
        for name, color in self.SERIES_COLORS.items():
            if not self.series[name]:
                continue
            painter.setPen(QPen(color, 3))
            painter.drawLine(legend_x, 10, legend_x + 18, 10)
            painter.setPen(QColor("#525965"))
            painter.drawText(legend_x + 24, 14, name)
            legend_x += 145


class MetricBarChart(QWidget):
    """Compact horizontal bars for the selected bin-picking report."""

    def __init__(self) -> None:
        super().__init__()
        self.values: list[tuple[str, float, str]] = []
        self.setMinimumHeight(230)

    def set_values(self, values: list[tuple[str, float, str]]) -> None:
        self.values = [(name, value, shown) for name, value, shown in values if value >= 0]
        self.setMinimumHeight(max(130, 36 * len(self.values) + 32))
        self.update()

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor("#ffffff"))
        if not self.values:
            painter.setPen(QColor("#7a808a"))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "No chartable metrics in this report")
            return
        max_value = max(value for _, value, _ in self.values) or 1.0
        name_width = min(190, max(110, self.width() // 3))
        bar_width = max(40, self.width() - name_width - 82)
        y = 18
        for index, (name, value, shown) in enumerate(self.values):
            painter.setPen(QColor("#414751"))
            painter.drawText(8, y + 15, name[:28])
            bar_rect = self.rect().adjusted(name_width, y, -68, -(self.height() - y - 23))
            bar_rect.setHeight(15)
            bar_rect.setWidth(bar_width)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor("#edf0f2"))
            painter.drawRoundedRect(bar_rect, 5, 5)
            fill = bar_rect.adjusted(0, 0, -bar_width + max(1, int(bar_width * value / max_value)), 0)
            painter.setBrush(QColor("#27866b" if index % 2 == 0 else "#4f8dc9"))
            painter.drawRoundedRect(fill, 5, 5)
            painter.setPen(QColor("#414751"))
            painter.drawText(bar_rect.right() + 8, y + 15, shown)
            y += 34


class CollapsiblePanel(QFrame):
    """Small dropdown section that can hide secondary chart content."""

    def __init__(self, title: str, content: QWidget, expanded: bool = False) -> None:
        super().__init__()
        self.setObjectName("collapsiblePanel")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 6, 10, 8)
        layout.setSpacing(6)
        self.toggle = QToolButton()
        self.toggle.setText(title)
        self.toggle.setCheckable(True)
        self.toggle.setChecked(expanded)
        self.toggle.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.toggle.setArrowType(Qt.ArrowType.DownArrow if expanded else Qt.ArrowType.RightArrow)
        self.toggle.setObjectName("chartDropdown")
        self.toggle.toggled.connect(self._set_expanded)
        layout.addWidget(self.toggle)
        self.content = content
        self.content.setVisible(expanded)
        layout.addWidget(self.content)

    def _set_expanded(self, expanded: bool) -> None:
        self.content.setVisible(expanded)
        self.toggle.setArrowType(Qt.ArrowType.DownArrow if expanded else Qt.ArrowType.RightArrow)


class TrainingWindow(QMainWindow):
    NAV_GROUPS = (
        ("WORKSPACE", (("Overview", 0, None),)),
        ("DATA", (
            ("Dataset", 1, None),
            ("Processing", 2, None),
        )),
        ("TRAINING", (
            ("Setup", 3, None),
            ("Dashboard", 5, None),
            ("Training History", 4, 0),
        )),
        ("RESULTS", (
            ("Experiments", 4, 1),
            ("Inference", 4, 0),
        )),
        ("SETTINGS", (("Runtime Environment", 3, None),)),
    )

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Rubinnova | Robotekin Training Studio")
        self.resize(1320, 860)
        self.setMinimumSize(1080, 720)
        self._task_kind = ""
        self._task_output = ""
        self._live_output_buffer = ""
        self._metric_step = 0
        self._task_total_steps = 0
        self._pending_training_metadata = None
        self.metric_refresh_timer = QTimer(self)
        self.metric_refresh_timer.setInterval(2500)
        self.metric_refresh_timer.timeout.connect(self._refresh_binpick_reports)
        self.process = QProcess(self)
        self.process.setProcessChannelMode(QProcess.MergedChannels)
        self.process.readyReadStandardOutput.connect(self._read_output)
        self.process.finished.connect(self._finished)
        self.process.errorOccurred.connect(self._process_error)
        self.hub_process = QProcess(self)
        self.hub_process.setProcessChannelMode(QProcess.MergedChannels)
        self.hub_process.readyReadStandardOutput.connect(self._hub_read_output)
        self.hub_process.finished.connect(self._hub_finished)
        self.hub_process.errorOccurred.connect(self._hub_process_error)
        self._hub_action = ""
        self._hub_output = ""
        self._hub_download_queue: list[str] = []
        self._hub_downloaded: list[str] = []
        self._hub_download_failed: list[str] = []
        self._hub_current_repo = ""

        shell = QWidget()
        shell_layout = QHBoxLayout(shell)
        shell_layout.setContentsMargins(0, 0, 0, 0)
        shell_layout.setSpacing(0)
        self.sidebar = self._build_sidebar()
        shell_layout.addWidget(self.sidebar)
        self.pages = QStackedWidget()
        self.pages.addWidget(self._build_overview())
        self.pages.addWidget(self._build_datasets())
        self.pages.addWidget(self._build_preparation())
        self.pages.addWidget(self._build_training())
        self.pages.addWidget(self._build_results())
        self.pages.addWidget(self._build_console())
        shell_layout.addWidget(self.pages, 1)
        self.setCentralWidget(shell)

        self.statusBar().showMessage("Ready to configure a training run")
        self.setStyleSheet(self._stylesheet())
        self._refresh_datasets()
        self._refresh_results()
        self._update_summary()
        self._update_dataset_status()
        self._update_output_path()

    def _build_sidebar(self) -> QFrame:
        sidebar = QFrame()
        sidebar.setObjectName("sidebar")
        sidebar.setFixedWidth(242)
        layout = QVBoxLayout(sidebar)
        layout.setContentsMargins(18, 22, 18, 18)
        layout.setSpacing(8)

        brand = QHBoxLayout()
        logo = QLabel()
        pixmap = QPixmap(str(REPO_ROOT / "assets" / "logo.png"))
        if not pixmap.isNull():
            logo.setPixmap(pixmap.scaled(
                48, 48, Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            ))
        logo.setFixedSize(50, 50)
        brand_text = QVBoxLayout()
        company = QLabel("RUBINNOVA")
        company.setObjectName("brandName")
        event = QLabel("ROBOTEKIN HACKATHON")
        event.setObjectName("brandEvent")
        brand_text.addWidget(company)
        brand_text.addWidget(event)
        brand.addWidget(logo)
        brand.addLayout(brand_text, 1)
        layout.addLayout(brand)

        label = QLabel("SO-101  /  ROBOT LEARNING")
        label.setObjectName("sidebarCaption")
        layout.addWidget(label)
        divider = QFrame()
        divider.setObjectName("sidebarDivider")
        divider.setFrameShape(QFrame.HLine)
        layout.addWidget(divider)

        self.nav_buttons: list[tuple[QPushButton, int, int | None]] = []
        for group_name, entries in self.NAV_GROUPS:
            group_label = QLabel(group_name)
            group_label.setObjectName("sidebarGroup")
            layout.addWidget(group_label)
            for title, page_index, tab_index in entries:
                button = QPushButton(title)
                button.setObjectName("navButton")
                button.setCheckable(True)
                button.clicked.connect(
                    lambda _checked=False, page=page_index, tab=tab_index, item=button: self._show_page(page, tab, item)
                )
                layout.addWidget(button)
                self.nav_buttons.append((button, page_index, tab_index))
        self.nav_buttons[0][0].setChecked(True)
        layout.addStretch(1)

        note = QLabel("ROBOT LEARNING\nLeRobot  ·  ACT  ·  SO-101")
        note.setObjectName("sidebarFootnote")
        note.setWordWrap(True)
        layout.addWidget(note)
        return sidebar

    def _new_page(self, title: str, subtitle: str) -> tuple[QWidget, QVBoxLayout]:
        page = QScrollArea()
        page.setWidgetResizable(True)
        page.setFrameShape(QFrame.NoFrame)
        surface = QWidget()
        layout = QVBoxLayout(surface)
        layout.setContentsMargins(32, 26, 32, 26)
        layout.setSpacing(18)
        eyebrow = QLabel("RUBINNOVA   /   ROBOTEKIN 2026")
        eyebrow.setObjectName("eyebrow")
        heading = QLabel(title)
        heading.setObjectName("pageTitle")
        description = QLabel(subtitle)
        description.setObjectName("pageSubtitle")
        description.setWordWrap(True)
        layout.addWidget(eyebrow)
        layout.addWidget(heading)
        layout.addWidget(description)
        page.setWidget(surface)
        return page, layout

    @staticmethod
    def _card(title: str, description: str = "") -> tuple[QFrame, QVBoxLayout]:
        frame = QFrame()
        frame.setObjectName("card")
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(18, 16, 18, 16)
        layout.setSpacing(9)
        heading = QLabel(title)
        heading.setObjectName("cardTitle")
        layout.addWidget(heading)
        if description:
            text = QLabel(description)
            text.setObjectName("mutedText")
            text.setWordWrap(True)
            layout.addWidget(text)
        return frame, layout

    def _build_overview(self) -> QWidget:
        page, layout = self._new_page(
            "Training Studio",
            "Train an ACT bin-picking policy from SO-101 demonstrations and evaluate it in Isaac Sim.",
        )

        hero = QFrame()
        hero.setObjectName("hero")
        hero_layout = QHBoxLayout(hero)
        hero_layout.setContentsMargins(24, 22, 24, 22)
        copy = QVBoxLayout()
        tagline = QLabel("ROBOT LEARNING  /  HACKATHON")
        tagline.setObjectName("heroEyebrow")
        headline = QLabel("Demonstrate. Train. Evaluate.")
        headline.setObjectName("heroTitle")
        detail = QLabel(
            "Record demonstrations, train the ACT imitation policy and measure sorting performance across randomized scenes."
        )
        detail.setObjectName("heroDetail")
        detail.setWordWrap(True)
        copy.addWidget(tagline)
        copy.addWidget(headline)
        copy.addWidget(detail)
        self.go_training_button = QPushButton("Configure training  →")
        self.go_training_button.setObjectName("primaryButton")
        self.go_training_button.clicked.connect(lambda: self._show_page(3))
        hero_layout.addLayout(copy, 1)
        hero_layout.addWidget(self.go_training_button, 0, Qt.AlignBottom)
        layout.addWidget(hero)

        stats = QHBoxLayout()
        self.stat_dataset_count = self._stat_card("DATASETS", "0", "local datasets available")
        self.stat_latest_run = self._stat_card("LATEST RUN", "—", "no checkpoints yet")
        self.stat_python = self._stat_card("ENVIRONMENT", "—", "select the training environment")
        for card in (self.stat_dataset_count[0], self.stat_latest_run[0], self.stat_python[0]):
            stats.addWidget(card, 1)
        layout.addLayout(stats)

        pipeline, pipeline_layout = self._card("Project workflow")
        steps = QHBoxLayout()
        for n, title, detail in (
            ("1", "Demonstrate", "Move the arm and record episodes"),
            ("2", "Prepare", "Review the dataset and its features"),
            ("3", "Train", "Fit ACT with LeRobot"),
            ("4", "Evaluate", "Measure the checkpoint on the task"),
        ):
            tile = QFrame()
            tile.setObjectName("pipelineStep")
            tile_layout = QVBoxLayout(tile)
            marker = QLabel(n)
            marker.setObjectName("stepNumber")
            name = QLabel(title)
            name.setObjectName("stepTitle")
            body = QLabel(detail)
            body.setObjectName("mutedText")
            body.setWordWrap(True)
            tile_layout.addWidget(marker)
            tile_layout.addWidget(name)
            tile_layout.addWidget(body)
            steps.addWidget(tile, 1)
        pipeline_layout.addLayout(steps)
        layout.addWidget(pipeline)

        layout.addStretch(1)
        return page

    def _stat_card(self, label: str, value: str, detail: str) -> tuple[QFrame, QLabel, QLabel]:
        card, layout = self._card(label)
        value_label = QLabel(value)
        value_label.setObjectName("statValue")
        detail_label = QLabel(detail)
        detail_label.setObjectName("mutedText")
        detail_label.setWordWrap(True)
        layout.addWidget(value_label)
        layout.addWidget(detail_label)
        return card, value_label, detail_label

    def _build_datasets(self) -> QWidget:
        page, layout = self._new_page(
            "Bin-Picking Datasets",
            "Browse local LeRobot datasets recorded for the bin-picking ACT policy.",
        )
        actions = QHBoxLayout()
        refresh = QPushButton("Refresh local datasets")
        refresh.clicked.connect(self._refresh_datasets)
        actions.addWidget(refresh)
        actions.addStretch(1)
        layout.addLayout(actions)

        columns = QHBoxLayout()
        hub_card, hub_layout = self._card(
            "Hugging Face · MikelRubi",
            "Browse datasets in the supplied profile and download LeRobot datasets into the local training data folder."
        )
        hub_actions = QHBoxLayout()
        self.hub_refresh_button = QPushButton("Refresh Hub list")
        self.hub_refresh_button.clicked.connect(self._refresh_hub_datasets)
        self.hub_download_button = QPushButton("Download checked")
        self.hub_download_button.setEnabled(False)
        self.hub_download_button.clicked.connect(self._download_hub_dataset)
        self.hub_select_all_button = QPushButton("Select all")
        self.hub_select_all_button.clicked.connect(lambda: self._set_hub_selection(True))
        self.hub_clear_button = QPushButton("Clear selection")
        self.hub_clear_button.clicked.connect(lambda: self._set_hub_selection(False))
        hub_open = QPushButton("Open profile")
        hub_open.clicked.connect(lambda: QDesktopServices.openUrl(QUrl("https://huggingface.co/MikelRubi/datasets")))
        hub_actions.addWidget(self.hub_refresh_button)
        hub_actions.addWidget(self.hub_download_button)
        hub_actions.addStretch(1)
        hub_layout.addLayout(hub_actions)
        hub_selection_actions = QHBoxLayout()
        self.hub_pickcube_button = QPushButton("Pick-cube")
        self.hub_pickcube_button.setToolTip("Select repositories whose names contain 'pick-cube' or 'pickcube'.")
        self.hub_pickcube_button.clicked.connect(lambda: self._set_hub_task_selection("pickcube"))
        self.hub_updown_button = QPushButton("Up-down")
        self.hub_updown_button.setToolTip("Select repositories whose names contain 'up down' or 'updown'.")
        self.hub_updown_button.clicked.connect(lambda: self._set_hub_task_selection("updown"))
        hub_selection_actions.addWidget(self.hub_select_all_button)
        hub_selection_actions.addWidget(self.hub_pickcube_button)
        hub_selection_actions.addWidget(self.hub_updown_button)
        hub_selection_actions.addWidget(self.hub_clear_button)
        hub_selection_actions.addWidget(hub_open)
        hub_selection_actions.addStretch(1)
        hub_layout.addLayout(hub_selection_actions)
        self.hub_dataset_list = QListWidget()
        self.hub_dataset_list.currentItemChanged.connect(self._hub_dataset_selected)
        self.hub_dataset_list.itemChanged.connect(self._hub_dataset_selection_changed)
        hub_layout.addWidget(self.hub_dataset_list, 1)
        self.hub_status = QLabel("Use Refresh Hub list to load the repositories.")
        self.hub_status.setObjectName("mutedText")
        self.hub_status.setWordWrap(True)
        hub_layout.addWidget(self.hub_status)
        self.hub_log = QPlainTextEdit()
        self.hub_log.setReadOnly(True)
        self.hub_log.setMaximumHeight(90)
        self.hub_log.setPlaceholderText("Hub list and download status will appear here.")
        hub_layout.addWidget(self.hub_log)
        columns.addWidget(hub_card, 1)

        detail_card, detail_layout = self._card("Dataset details")
        detail_layout.addWidget(QLabel("Local dataset for the next training run"))
        self.local_dataset_combo = QComboBox()
        self.local_dataset_combo.currentIndexChanged.connect(self._dataset_selected)
        detail_layout.addWidget(self.local_dataset_combo)
        self.dataset_details = QPlainTextEdit()
        self.dataset_details.setReadOnly(True)
        self.dataset_details.setPlaceholderText("Select a dataset to view its metadata.")
        detail_layout.addWidget(self.dataset_details)
        detail_layout.addWidget(QLabel("Task groups to include (all selected by default)"))
        self.task_group_list = QListWidget()
        self.task_group_list.setMaximumHeight(150)
        self.task_group_list.itemChanged.connect(self._task_group_changed)
        detail_layout.addWidget(self.task_group_list)
        detail_layout.addWidget(QLabel("Episodes to include in the training split"))
        episode_actions = QHBoxLayout()
        select_episodes = QPushButton("Select all")
        select_episodes.clicked.connect(lambda: self._set_episode_selection(True))
        clear_episodes = QPushButton("Clear selection")
        clear_episodes.clicked.connect(lambda: self._set_episode_selection(False))
        episode_actions.addWidget(select_episodes)
        episode_actions.addWidget(clear_episodes)
        self.episode_selection_status = QLabel("Select a local dataset to list its episodes.")
        self.episode_selection_status.setObjectName("mutedText")
        episode_actions.addWidget(self.episode_selection_status, 1)
        detail_layout.addLayout(episode_actions)
        self.episode_selection_list = QListWidget()
        self.episode_selection_list.setMaximumHeight(210)
        self.episode_selection_list.itemChanged.connect(self._update_episode_selection_status)
        detail_layout.addWidget(self.episode_selection_list)
        self.use_dataset_button = QPushButton("Use dataset for training")
        self.use_dataset_button.clicked.connect(lambda: self._show_page(3))
        detail_layout.addWidget(self.use_dataset_button)
        columns.addWidget(detail_card, 1)
        layout.addLayout(columns, 1)
        return page

    def _build_preparation(self) -> QWidget:
        page, layout = self._new_page(
            "Data Preparation",
            "Review the data pipeline before spending GPU time on training.",
        )
        stages = QGridLayout()
        for row, (number, title, detail, script) in enumerate((
            ("01", "Record bin-picking demos", "Capture successful ACT episodes with RGB, target mask, joint state and actions.", "tools/collect_demos.py"),
            ("02", "Validate dataset", "Check episode metadata, features and demonstration success.", "tools/validate_dataset.py"),
            ("03", "Train ACT policy", "Train directly from LeRobot episodes; reserve held-out episodes for evaluation.", "tools/train_act.py"),
        )):
            card, card_layout = self._card(f"{number}  {title}", detail)
            code = QLabel(script)
            code.setObjectName("codeLabel")
            card_layout.addWidget(code)
            stages.addWidget(card, row // 2, row % 2)
        layout.addLayout(stages)

        data_card, data_layout = self._card("Selected dataset")
        self.preparation_status = QLabel()
        self.preparation_status.setObjectName("mutedText")
        self.preparation_status.setWordWrap(True)
        data_layout.addWidget(self.preparation_status)
        layout.addWidget(data_card)

        layout.addStretch(1)
        return page

    def _build_training(self) -> QWidget:
        page, layout = self._new_page(
            "Configure Training",
            "Train a LeRobot ACT imitation-learning policy for bin picking from recorded robot demonstrations.",
        )
        environment, env_layout = self._card("Runtime environment")
        python_row = QHBoxLayout()
        conda_prefix = Path(os.environ.get("CONDA_PREFIX", ""))
        conda_python = conda_prefix / "python.exe"
        self.python_path = QLineEdit(str(conda_python) if conda_python.is_file() else "")
        self.python_path.setPlaceholderText("Select python.exe from the hackathon-robotekin Conda environment")
        self.python_path.textChanged.connect(self._update_summary)
        browse_python = QPushButton("Browse for Python…")
        browse_python.clicked.connect(self._browse_python)
        python_row.addWidget(self.python_path, 1)
        python_row.addWidget(browse_python)
        env_layout.addLayout(python_row)
        env_note = QLabel(
            "Select the bin-picking environment with LeRobot 0.4.4; the full sorting evaluation also needs Isaac Sim dependencies."
        )
        env_note.setObjectName("mutedText")
        env_note.setWordWrap(True)
        env_layout.addWidget(env_note)
        layout.addWidget(environment)

        dataset_card, dataset_layout = self._card("Training dataset")
        self.dataset = QLineEdit("bin_pick_v1")
        self.dataset.setReadOnly(True)
        self.dataset.textChanged.connect(self._update_dataset_status)
        dataset_layout.addWidget(QLabel("Dataset folder inside demos/09_voice_bin_picking/data"))
        dataset_layout.addWidget(self.dataset)
        self.dataset_status = QLabel()
        self.dataset_status.setWordWrap(True)
        dataset_layout.addWidget(self.dataset_status)
        layout.addWidget(dataset_card)

        model_card, model_layout = self._card("Model and architecture", "Policy architecture for imitation learning.")
        model_grid = QGridLayout()
        self.training_mode = KeyboardOnlyComboBox()
        self.training_mode.addItem("New training", "new")
        self.training_mode.addItem("Fine-tune from checkpoint", "transfer")
        self.training_mode.currentIndexChanged.connect(self._training_mode_changed)
        self.policy_model = KeyboardOnlyComboBox()
        self.policy_model.addItem("ACT — Action Chunking Transformer", "act")
        self.policy_model.setEnabled(False)
        self.backbone = KeyboardOnlyComboBox()
        for backbone in ("resnet18", "resnet34", "resnet50"):
            self.backbone.addItem(backbone.upper(), backbone)
        model_grid.addWidget(QLabel("Policy"), 0, 0)
        model_grid.addWidget(self.policy_model, 1, 0)
        model_grid.addWidget(QLabel("Vision backbone"), 0, 1)
        model_grid.addWidget(self.backbone, 1, 1)
        model_grid.addWidget(QLabel("Training mode"), 0, 2)
        model_grid.addWidget(self.training_mode, 1, 2)
        self.observation_mode = KeyboardOnlyComboBox()
        self.observation_mode.addItem("RGB + target mask (recommended)", "rgb_mask")
        self.observation_mode.addItem("RGB only (comparison)", "rgb_only")
        self.observation_mode.addItem("Target mask only (comparison)", "mask_only")
        model_grid.addWidget(QLabel("ACT observation inputs"), 0, 3)
        model_grid.addWidget(self.observation_mode, 1, 3)
        model_layout.addLayout(model_grid)
        checkpoint_row = QHBoxLayout()
        self.pretrained_policy = QLineEdit()
        self.pretrained_policy.setPlaceholderText("Optional pretrained ACT checkpoint folder")
        self.pretrained_policy.setEnabled(False)
        browse_policy = QPushButton("Choose checkpoint…")
        browse_policy.clicked.connect(self._browse_pretrained_policy)
        browse_policy.setEnabled(False)
        self.browse_policy_button = browse_policy
        checkpoint_row.addWidget(self.pretrained_policy, 1)
        checkpoint_row.addWidget(browse_policy)
        model_layout.addLayout(checkpoint_row)
        self.pretrained_backbone = KeyboardOnlyCheckBox("Initialize backbone with ImageNet weights")
        self.pretrained_backbone.setChecked(True)
        model_layout.addWidget(self.pretrained_backbone)
        model_note = QLabel("Larger ResNet backbones use more GPU memory and training time. ImageNet weights are available for all listed backbones.")
        model_note.setObjectName("mutedText")
        model_note.setWordWrap(True)
        model_layout.addWidget(model_note)
        layout.addWidget(model_card)

        params_card, params_layout = self._card(
            "Training configuration",
            "First-run profile: 10,000 steps, batch 8, chunk 50, replan every 10 actions, learning rate 1e-5 and pretrained ResNet18. "
            "Change model and training parameters with the keyboard; dataset and output folder selection still use the mouse.",
        )
        params = QGridLayout()
        self.steps = self._spin(1, 1_000_000, 10_000)
        self.batch = self._spin(1, 512, 8)
        self.chunk = self._spin(1, 1000, 50)
        self.action_steps = self._spin(1, 1000, 10)
        self.seed = self._spin(0, 2_147_483_647, 1000)
        self.device = KeyboardOnlyComboBox()
        self.device.addItems(["cuda", "cpu"])
        self.learning_rate = KeyboardOnlyLineEdit("1e-5")
        self.episodes = QLineEdit()
        self.episodes.setPlaceholderText("Optional additional filter: 0-39")
        fields = (
            ("Training steps", self.steps),
            ("Batch size", self.batch),
            ("Action chunk size", self.chunk),
            ("Actions per prediction", self.action_steps),
            ("Device", self.device),
            ("Learning rate", self.learning_rate),
            ("Random seed", self.seed),
            ("Episodes", self.episodes),
        )
        for index, (label, widget) in enumerate(fields):
            cell = QVBoxLayout()
            cell.addWidget(QLabel(label))
            cell.addWidget(widget)
            params.addLayout(cell, index // 4, index % 4)
        params_layout.addLayout(params)
        split_card, split_layout = self._card(
            "Dataset split",
            "Match SkinEye's automatic/manual split workflow. Splits are episode-based to keep robot trajectories intact."
        )
        self.split_mode = KeyboardOnlyComboBox()
        self.split_mode.addItem("Automatic split", "automatic")
        self.split_mode.addItem("Manual episode lists", "manual")
        split_layout.addWidget(self.split_mode)
        self.split_percent_widgets = []
        percent_grid = QGridLayout()
        for column, (label, default) in enumerate((("Training", 80), ("Validation", 20), ("Test", 0))):
            percent_grid.addWidget(QLabel(f"{label} (%)"), 0, column)
            value = self._spin(0, 100, default)
            value.valueChanged.connect(self._normalize_split_percentages)
            percent_grid.addWidget(value, 1, column)
            self.split_percent_widgets.append(value)
        split_layout.addLayout(percent_grid)
        self.manual_episode_edits = []
        manual_form = QFormLayout()
        for label in ("Training episodes", "Validation episodes", "Test episodes"):
            edit = QLineEdit()
            edit.setPlaceholderText("Example: 0-39 or 0,2,5")
            self.manual_episode_edits.append(edit)
            manual_form.addRow(label, edit)
        split_layout.addLayout(manual_form)
        self.split_note = QLabel("Automatic split defaults to 80% training, 20% validation and 0% test.")
        self.split_note.setObjectName("mutedText")
        self.split_note.setWordWrap(True)
        split_layout.addWidget(self.split_note)
        self.split_mode.currentIndexChanged.connect(self._update_split_mode)
        layout.addWidget(split_card)
        self._update_split_mode()
        skin_eye_params = QGridLayout()
        self.image_size_info = QLabel("Dataset-defined")
        self.image_size_info.setToolTip("ACT uses the camera feature dimensions stored in the LeRobot dataset; image size is not an independent Ultralytics-style parameter.")
        self.patience_info = QSpinBox()
        self.patience_info.setRange(0, 100000)
        self.patience_info.setValue(0)
        self.patience_info.setEnabled(False)
        self.patience_info.setToolTip("LeRobot ACT training is step-based and this trainer has no epoch-based early-stopping callback.")
        skin_eye_params.addWidget(QLabel("Image size"), 0, 0)
        skin_eye_params.addWidget(self.image_size_info, 1, 0)
        skin_eye_params.addWidget(QLabel("Early stopping patience"), 0, 1)
        skin_eye_params.addWidget(self.patience_info, 1, 1)
        params_layout.addLayout(skin_eye_params)
        self.advanced_options_button = QPushButton("More training options…")
        self.advanced_options_button.clicked.connect(self._open_advanced_training_options)
        params_layout.addWidget(self.advanced_options_button)
        self.no_vae = KeyboardOnlyCheckBox("Disable ACT VAE (ablation option)")
        self.no_vae.setChecked(False)
        params_layout.addWidget(self.no_vae)
        self.use_robot_state = KeyboardOnlyCheckBox("Include robot joint state as ACT input")
        self.use_robot_state.setChecked(True)
        self.use_robot_state.setToolTip("Disable as an ablation to measure whether the policy relies too strongly on its current pose.")
        params_layout.addWidget(self.use_robot_state)
        layout.addWidget(params_card)

        advanced_card, advanced_layout = self._card("ACT architecture options", "Policy architecture settings that do not appear in SkinEye's detector training form.")
        advanced = QGridLayout()
        self.weight_decay = KeyboardOnlyLineEdit("1e-4")
        self.weight_decay.setToolTip("AdamW weight decay. Set 0 to disable this regularization.")
        self.backbone_lr = KeyboardOnlyLineEdit("1e-5")
        self.backbone_lr.setToolTip("Separate learning rate for the pretrained ResNet vision backbone.")
        self.dropout = KeyboardOnlyLineEdit("0.1")
        self.dropout.setToolTip("Dropout probability used inside ACT's transformer.")
        self.kl_weight = KeyboardOnlyLineEdit("10.0")
        self.dim_model = self._spin(64, 2048, 512)
        self.encoder_layers = self._spin(1, 16, 4)
        self.decoder_layers = self._spin(1, 8, 1)
        self.attention_heads = self._spin(1, 32, 8)
        self.latent_dim = self._spin(1, 512, 32)
        self.feedforward_dim = self._spin(128, 16384, 3200)
        self.vae_encoder_layers = self._spin(1, 16, 4)
        self.feedforward_activation = KeyboardOnlyComboBox()
        self.feedforward_activation.addItems(["relu", "gelu"])
        self.replace_final_stride = KeyboardOnlyCheckBox("Use dilation in the final ResNet stage")
        self.replace_final_stride.setToolTip("Preserves more spatial detail and increases compute/memory use.")
        self.temporal_ensemble = KeyboardOnlyCheckBox("Enable temporal action ensembling")
        self.temporal_ensemble.setToolTip("Smooths overlapping action chunks at inference; requires one action per prediction.")
        self.temporal_ensemble_coeff = KeyboardOnlyDoubleSpinBox()
        self.temporal_ensemble_coeff.setRange(0.001, 1.0)
        self.temporal_ensemble_coeff.setSingleStep(0.01)
        self.temporal_ensemble_coeff.setDecimals(3)
        self.temporal_ensemble_coeff.setValue(0.01)
        self.temporal_ensemble_coeff.setEnabled(False)
        self.temporal_ensemble.toggled.connect(self.temporal_ensemble_coeff.setEnabled)
        self.pre_norm = KeyboardOnlyCheckBox("Pre-normalize transformer inputs")
        self.pre_norm.setToolTip("Changes transformer normalization order; retain the default unless comparing a separate experiment.")
        self.use_amp = KeyboardOnlyCheckBox("Use mixed precision when supported")
        self.use_amp.setToolTip("Uses lower precision on supported CUDA devices to reduce memory and often improve throughput.")
        self.optimizer = KeyboardOnlyComboBox()
        self.optimizer.addItem("AdamW (ACT policy preset)")
        self.optimizer.setEnabled(False)
        self.save_freq = self._spin(1, 1_000_000, 2500)
        self.log_freq = self._spin(1, 100_000, 200)
        self.workers = self._spin(0, 64, 4)
        self.lr_final_fraction = 0.01
        self.momentum = 0.937
        self.warmup_epochs = 3.0
        self.warmup_momentum = 0.8
        self.warmup_bias_lr = 0.1
        self.cosine_schedule = False
        self.dataset_fraction = 1.0
        advanced_fields = (
            ("Backbone learning rate", self.backbone_lr),
            ("Dropout", self.dropout),
            ("KL loss weight", self.kl_weight),
            ("Transformer hidden dimension", self.dim_model),
            ("Transformer feed-forward dimension", self.feedforward_dim),
            ("Feed-forward activation", self.feedforward_activation),
            ("Transformer encoder layers", self.encoder_layers),
            ("Transformer decoder layers", self.decoder_layers),
            ("Attention heads", self.attention_heads),
            ("VAE latent dimension", self.latent_dim),
            ("VAE encoder layers", self.vae_encoder_layers),
            ("Optimizer", self.optimizer),
            ("Log metrics every N steps", self.log_freq),
            ("Data loader workers", self.workers),
            ("Checkpoint period (steps)", self.save_freq),
        )
        for index, (label, widget) in enumerate(advanced_fields):
            cell = QVBoxLayout()
            cell.addWidget(QLabel(label))
            cell.addWidget(widget)
            advanced.addLayout(cell, index // 3, index % 3)
        advanced.addWidget(self.pre_norm, 4, 0, 1, 2)
        advanced.addWidget(self.use_amp, 4, 2)
        advanced.addWidget(self.replace_final_stride, 5, 0, 1, 3)
        advanced.addWidget(self.temporal_ensemble, 6, 0, 1, 2)
        advanced.addWidget(self.temporal_ensemble_coeff, 6, 2)
        advanced_layout.addLayout(advanced)
        self.deterministic = QCheckBox("Deterministic run (more reproducible; may be slower)")
        self.deterministic.setEnabled(False)
        self.deterministic.setToolTip("The bin-picking ACT trainer does not expose a deterministic-mode flag.")
        self.wandb_enabled = QCheckBox("Log metrics to Weights & Biases")
        self.wandb_enabled.setEnabled(False)
        self.wandb_enabled.setToolTip("This trainer does not configure W&B from the studio.")
        self.wandb_project = QLineEdit("robotekin-bin-picking")
        self.wandb_project.setEnabled(False)
        self.wandb_entity = QLineEdit()
        self.wandb_entity.setEnabled(False)
        self.wandb_entity.setPlaceholderText("Entity (optional; uses the authenticated account)")
        wandb_form = QFormLayout()
        wandb_form.addRow(self.wandb_enabled)
        wandb_form.addRow("W&B project", self.wandb_project)
        wandb_form.addRow("W&B entity", self.wandb_entity)
        advanced_layout.addLayout(wandb_form)
        supported_note = QLabel(
            "ACT architecture, optimizer regularization, observation inputs, data selection, mixed precision and run intervals are passed to the trainer. "
            "The script checks the selected LeRobot version and reports unsupported options before training. "
            "Setup values are changed with the keyboard; mouse wheel, spinner arrows and dropdown clicks are disabled."
        )
        supported_note.setWordWrap(True)
        supported_note.setObjectName("mutedText")
        advanced_layout.addWidget(supported_note)
        layout.addWidget(advanced_card)

        output_card, output_layout = self._card("Experiment output")
        self.output_stage = QComboBox()
        self.output_stage.addItem("Pick-up — grasp and present", "Pick-up")
        self.output_stage.addItem("Up-down — sort into bins", "Up-down")
        self.output_stage.setToolTip("Selects the output folder for this run. Choose task groups and episodes separately in Datasets.")
        self.output_stage.currentIndexChanged.connect(self._update_output_path)
        output_layout.addWidget(QLabel("Training task / output folder"))
        output_layout.addWidget(self.output_stage)
        self.output_name = QLineEdit("act_baseline_10k")
        self.output_name.textChanged.connect(self._update_output_path)
        output_layout.addWidget(QLabel("Use a new name; existing runs will not be overwritten"))
        output_layout.addWidget(self.output_name)
        self.output_path = QLabel()
        self.output_path.setObjectName("mutedText")
        self.output_path.setWordWrap(True)
        output_layout.addWidget(self.output_path)
        layout.addWidget(output_card)

        controls = QHBoxLayout()
        self.start_button = QPushButton("Start training")
        self.start_button.setObjectName("primaryButton")
        self.start_button.clicked.connect(self._start_training)
        go_console = QPushButton("Open console")
        go_console.clicked.connect(lambda: self._show_page(5))
        controls.addWidget(self.start_button)
        controls.addWidget(go_console)
        controls.addStretch(1)
        layout.addLayout(controls)
        layout.addStretch(1)
        return page

    def _build_results(self) -> QWidget:
        page, layout = self._new_page(
            "Models and Evaluation",
            "Review training runs, checkpoints and recorded evaluation reports.",
        )
        refresh = QPushButton("Refresh results")
        refresh.clicked.connect(self._refresh_results)
        refresh_row = QHBoxLayout()
        refresh_row.addStretch(1)
        refresh_row.addWidget(refresh)
        layout.addLayout(refresh_row)

        self.results_tabs = QTabWidget()
        training_tab = QWidget()
        training_layout = QVBoxLayout(training_tab)
        training_split = QHBoxLayout()
        run_list_card, run_list_layout = self._card("Training runs")
        self.results_list = QListWidget()
        self.results_list.currentItemChanged.connect(self._result_selected)
        run_list_layout.addWidget(self.results_list)
        training_split.addWidget(run_list_card, 1)
        run_details_card, run_details_layout = self._card("Run details and training metrics")
        self.run_details = QPlainTextEdit()
        self.run_details.setReadOnly(True)
        self.run_details.setPlaceholderText("Select a run to view its configuration, artifacts and latest logged metrics.")
        run_details_layout.addWidget(self.run_details)
        training_split.addWidget(run_details_card, 2)
        training_layout.addLayout(training_split, 1)
        controls = QHBoxLayout()
        self.eval_episodes = self._spin(1, 1000, 20)
        self.eval_headless = QCheckBox("Headless simulation")
        self.eval_headless.setChecked(True)
        self.eval_segmenter = QLineEdit("hsv")
        self.eval_segmenter.setPlaceholderText("hsv or yolo:<weights.pt>")
        self.open_output_button = QPushButton("Open run folder")
        self.open_output_button.clicked.connect(self._open_selected_output)
        self.evaluate_button = QPushButton("Run full sorting evaluation")
        self.evaluate_button.setObjectName("primaryButton")
        self.evaluate_button.clicked.connect(self._start_evaluation)
        self.offline_evaluate_button = QPushButton("Evaluate held-out ACT actions")
        self.offline_evaluate_button.clicked.connect(self._start_offline_evaluation)
        controls.addWidget(QLabel("Evaluation seeds"))
        controls.addWidget(self.eval_episodes)
        controls.addWidget(QLabel("Segmenter"))
        controls.addWidget(self.eval_segmenter)
        controls.addWidget(self.eval_headless)
        controls.addWidget(self.evaluate_button)
        controls.addWidget(self.offline_evaluate_button)
        controls.addWidget(self.open_output_button)
        training_layout.addLayout(controls)
        self.results_tabs.addTab(training_tab, "Training Runs")

        evaluation_tab = QWidget()
        evaluation_layout = QHBoxLayout(evaluation_tab)
        report_list_card, report_list_layout = self._card("Evaluation reports")
        self.evaluation_list = QListWidget()
        self.evaluation_list.currentItemChanged.connect(self._evaluation_selected)
        report_list_layout.addWidget(self.evaluation_list)
        evaluation_layout.addWidget(report_list_card, 1)
        report_detail_card, report_detail_layout = self._card("Evaluation results")
        self.evaluation_details = QPlainTextEdit()
        self.evaluation_details.setReadOnly(True)
        self.evaluation_details.setPlaceholderText("Select a report to view success rate and per-run details.")
        report_detail_layout.addWidget(self.evaluation_details)
        evaluation_layout.addWidget(report_detail_card, 2)
        self.results_tabs.addTab(evaluation_tab, "Evaluation History")

        binpick_tab = QWidget()
        binpick_layout = QVBoxLayout(binpick_tab)
        binpick_intro = QLabel(
            "Metrics from the bin-picking perception, ACT policy and full sorting evaluations. "
            "Refresh after running the evaluators to load their reports."
        )
        binpick_intro.setObjectName("mutedText")
        binpick_intro.setWordWrap(True)
        binpick_layout.addWidget(binpick_intro)
        binpick_split = QHBoxLayout()
        binpick_list_card, binpick_list_layout = self._card("Bin-picking reports")
        self.binpick_report_list = QListWidget()
        self.binpick_report_list.currentItemChanged.connect(self._binpick_report_selected)
        binpick_list_layout.addWidget(self.binpick_report_list)
        binpick_split.addWidget(binpick_list_card, 1)
        binpick_detail_card, binpick_detail_layout = self._card("Perception, policy and task metrics")
        chart_content = QWidget()
        chart_content_layout = QVBoxLayout(chart_content)
        chart_content_layout.setContentsMargins(0, 0, 0, 0)
        self.binpick_chart_note = QLabel("Bars show comparable rates or action errors for the selected report.")
        self.binpick_chart_note.setObjectName("mutedText")
        chart_content_layout.addWidget(self.binpick_chart_note)
        self.binpick_metric_chart = MetricBarChart()
        chart_content_layout.addWidget(self.binpick_metric_chart)
        self.binpick_chart_dropdown = CollapsiblePanel("Show report chart", chart_content, expanded=False)
        binpick_detail_layout.addWidget(self.binpick_chart_dropdown)
        self.binpick_report_details = QPlainTextEdit()
        self.binpick_report_details.setReadOnly(True)
        self.binpick_report_details.setPlaceholderText("Select a bin-picking report to inspect its metrics.")
        self.binpick_report_details.setMinimumHeight(190)
        binpick_detail_layout.addWidget(self.binpick_report_details)
        binpick_split.addWidget(binpick_detail_card, 2)
        binpick_layout.addLayout(binpick_split, 1)
        self.results_tabs.addTab(binpick_tab, "Bin Picking Metrics")
        layout.addWidget(self.results_tabs, 1)
        return page

    def _build_console(self) -> QWidget:
        page, layout = self._new_page(
            "Training Dashboard",
            "Follow the latest training metrics and inspect the live process output.",
        )
        bar = QHBoxLayout()
        self.task_label = QLabel("No active process")
        self.task_label.setObjectName("consoleStatus")
        self.live_metric_labels = {}
        metrics_card, metrics_layout = self._card(
            "Live training metrics",
            "The bin-picking ACT trainer reports training loss during the run. Validation is performed offline after training."
        )
        live_grid = QGridLayout()
        for index, (key, title, initial) in enumerate((
            ("step", "Step", "—"),
            ("loss", "Train loss", "—"),
            ("eval_loss", "Validation loss", "—"),
            ("grad_norm", "Gradient norm", "—"),
            ("lr", "Learning rate", "—"),
            ("samples_per_s", "Samples / sec", "—"),
            ("gpu_mem_gb", "GPU memory (GB)", "—"),
        )):
            cell = QVBoxLayout()
            title_label = QLabel(title)
            title_label.setObjectName("mutedText")
            value_label = QLabel(initial)
            value_label.setObjectName("metricValue")
            cell.addWidget(title_label)
            cell.addWidget(value_label)
            live_grid.addLayout(cell, index // 4, index % 4)
            self.live_metric_labels[key] = value_label
        metrics_layout.addLayout(live_grid)
        layout.addWidget(metrics_card)
        loss_content = QWidget()
        loss_layout = QVBoxLayout(loss_content)
        loss_layout.setContentsMargins(0, 0, 0, 0)
        loss_note = QLabel("Lower validation loss is useful only when measured on episodes held out from training.")
        loss_note.setObjectName("mutedText")
        loss_layout.addWidget(loss_note)
        self.live_loss_chart = LiveLossChart()
        loss_layout.addWidget(self.live_loss_chart)
        self.live_loss_dropdown = CollapsiblePanel("Show training and validation loss chart", loss_content, expanded=False)
        layout.addWidget(self.live_loss_dropdown)
        self.stop_button = QPushButton("Stop process")
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(self._stop_process)
        clear = QPushButton("Clear console")
        clear.clicked.connect(lambda: self.log.clear())
        self.progress = QProgressBar()
        self.progress.setRange(0, 0)
        self.progress.setTextVisible(False)
        self.progress.hide()
        bar.addWidget(self.task_label)
        bar.addStretch(1)
        bar.addWidget(self.progress, 1)
        bar.addWidget(self.stop_button)
        bar.addWidget(clear)
        layout.addLayout(bar)

        console, console_layout = self._card("Console")
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(8000)
        self.log.setPlaceholderText("Start a training or evaluation run to view its output.")
        console_layout.addWidget(self.log)
        layout.addWidget(console, 1)
        return page

    def _show_page(
        self, index: int, tab_index: int | None = None, active_button: QPushButton | None = None
    ) -> None:
        self.pages.setCurrentIndex(index)
        if index == 6 and tab_index is not None:
            self.results_tabs.setCurrentIndex(tab_index)
        if active_button is None:
            active_button = next((
                button for button, page_index, route_tab in self.nav_buttons
                if page_index == index and (route_tab == tab_index or (tab_index is None and route_tab is None))
            ), None)
        for button, page_index, route_tab in self.nav_buttons:
            button.setChecked(button is active_button)

    @staticmethod
    def _spin(minimum: int, maximum: int, value: int) -> QSpinBox:
        widget = KeyboardOnlySpinBox()
        widget.setRange(minimum, maximum)
        widget.setValue(value)
        return widget

    def _browse_python(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Select the training environment's Python interpreter", "", "Python (python.exe)"
        )
        if path:
            self.python_path.setText(path)

    def _browse_pretrained_policy(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Select a pretrained ACT checkpoint")
        if path:
            self.pretrained_policy.setText(path)

    def _normalize_split_percentages(self, value: int) -> None:
        if not hasattr(self, "split_percent_widgets"):
            return
        source = self.sender()
        train, validation, test = self.split_percent_widgets
        if source is train:
            remainder = max(0, 100 - train.value() - test.value())
            validation.blockSignals(True)
            validation.setValue(remainder)
            validation.blockSignals(False)
        elif source is validation:
            remainder = max(0, 100 - validation.value() - test.value())
            train.blockSignals(True)
            train.setValue(remainder)
            train.blockSignals(False)
        else:
            remainder = max(0, 100 - validation.value() - test.value())
            train.blockSignals(True)
            train.setValue(remainder)
            train.blockSignals(False)

    @staticmethod
    def _parse_episode_ids(text: str) -> list[int]:
        ids = []
        for part in text.split(","):
            part = part.strip()
            if not part:
                continue
            if "-" in part:
                first, last = (int(value.strip()) for value in part.split("-", 1))
                if last < first:
                    raise ValueError("Episode ranges must be in ascending order.")
                ids.extend(range(first, last + 1))
            else:
                ids.append(int(part))
        return sorted(set(ids))

    @staticmethod
    def _compress_episode_ids(ids: list[int]) -> str:
        if not ids:
            return ""
        values = sorted(set(ids))
        ranges = []
        start = previous = values[0]
        for value in values[1:]:
            if value == previous + 1:
                previous = value
                continue
            ranges.append(f"{start}-{previous}" if start != previous else str(start))
            start = previous = value
        ranges.append(f"{start}-{previous}" if start != previous else str(start))
        return ",".join(ranges)

    def _prepare_episode_split(self) -> dict:
        metadata_path = self._dataset_dir() / "meta" / "info.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        total = int(metadata.get("total_episodes", 0))
        if total < 1:
            raise ValueError("The selected dataset does not report any episodes.")
        available = self._selected_dataset_episode_ids()
        if not available:
            raise ValueError("Select at least one episode from the dataset page.")
        selected_text = self.episodes.text().strip()
        if selected_text:
            requested_ids = self._parse_episode_ids(selected_text)
            if any(index < 0 or index >= total for index in requested_ids):
                raise ValueError(f"Episode IDs must be between 0 and {total - 1}.")
            if not set(requested_ids).issubset(available):
                raise ValueError("The episode list includes episodes outside the selected task groups.")
            available = requested_ids
        if self.split_mode.currentData() == "manual":
            train_text, validation_text, test_text = (edit.text().strip() for edit in self.manual_episode_edits)
            if not train_text:
                raise ValueError("Enter the training episode IDs for a manual split.")
            train_ids = self._parse_episode_ids(train_text)
            validation_ids = self._parse_episode_ids(validation_text)
            test_ids = self._parse_episode_ids(test_text)
            sets = [set(train_ids), set(validation_ids), set(test_ids)]
            if any(sets[i] & sets[j] for i in range(3) for j in range(i + 1, 3)):
                raise ValueError("Training, validation and test episode lists must not overlap.")
            if any(index < 0 or index >= total for ids in sets for index in ids):
                raise ValueError(f"Episode IDs must be between 0 and {total - 1}.")
            if any(not ids.issubset(available) for ids in sets):
                raise ValueError("Manual episode lists must stay within the selected task groups and episode filter.")
        else:
            percentages = [widget.value() for widget in self.split_percent_widgets]
            if sum(percentages) != 100 or percentages[0] <= 0:
                raise ValueError("Automatic train/validation/test percentages must total 100 and training must be positive.")
            rng = random.Random(self.seed.value())
            rng.shuffle(available)
            keep = max(1, int(len(available) * self.dataset_fraction))
            available = available[:keep]
            n_train = max(1, int(len(available) * percentages[0] / 100))
            n_validation = int(len(available) * percentages[1] / 100)
            if n_train + n_validation > len(available):
                n_validation = max(0, len(available) - n_train)
            train_ids = available[:n_train]
            validation_ids = available[n_train:n_train + n_validation]
            test_ids = available[n_train + n_validation:]
        if not train_ids:
            raise ValueError("The split produced no training episodes.")
        if self.split_mode.currentData() == "manual" and self.dataset_fraction < 1.0:
            rng = random.Random(self.seed.value())
            rng.shuffle(train_ids)
            train_ids = train_ids[:max(1, int(len(train_ids) * self.dataset_fraction))]
        return {
            "dataset": self.dataset.text().strip(),
            "repo_id": self._dataset_repo_id(),
            "task_groups": sorted({
                group for episode in available
                for group in self._episode_group_map.get(episode, set())
            }),
            "total_episodes": total,
            "train_episodes": sorted(train_ids),
            "validation_episodes": sorted(validation_ids),
            "test_episodes": sorted(test_ids),
            "seed": self.seed.value(),
            "observation_mode": self.observation_mode.currentData(),
        }
    def _update_split_mode(self, *_args) -> None:
        manual = self.split_mode.currentData() == "manual"
        for widget in self.split_percent_widgets:
            widget.setEnabled(not manual)
        for widget in self.manual_episode_edits:
            widget.setEnabled(manual)
        self.split_note.setText(
            "Training episodes are passed to ACT; validation/test episodes remain held out for offline policy or full sorting evaluation."
            if manual else "Episodes are split before training. Validation/test sets are held out for offline or full sorting evaluation; ACT has no in-loop validation."
        )

    def _training_mode_changed(self, _index: int) -> None:
        transfer = self.training_mode.currentData() == "transfer"
        self.pretrained_policy.setEnabled(transfer)
        self.browse_policy_button.setEnabled(transfer)

    def _open_advanced_training_options(self) -> None:
        dialog = QDialog(self)
        dialog.setWindowTitle("More training options")
        dialog.setMinimumWidth(560)
        layout = QVBoxLayout(dialog)
        intro = QLabel(
            "ACT imitation-learning controls supported by the bin-picking trainer are enabled. "
            "Values that the trainer cannot apply remain disabled."
        )
        intro.setWordWrap(True)
        intro.setObjectName("mutedText")
        layout.addWidget(intro)
        form = QFormLayout()
        layout.addLayout(form)

        optimizer = QComboBox()
        optimizer.addItems(["auto", "SGD", "Adam", "AdamW", "RMSProp"])
        optimizer.setCurrentText("AdamW")
        optimizer.setEnabled(False)
        optimizer.setToolTip("ACT's policy preset uses AdamW. The other Ultralytics optimizer choices cannot train this ACT policy.")
        form.addRow("Optimizer", optimizer)

        def float_control(value: float, minimum: float, maximum: float, step: float, decimals: int = 4) -> QDoubleSpinBox:
            widget = KeyboardOnlyDoubleSpinBox()
            widget.setRange(minimum, maximum)
            widget.setSingleStep(step)
            widget.setDecimals(decimals)
            widget.setValue(value)
            return widget

        def integer_control(value: int, minimum: int, maximum: int) -> QSpinBox:
            widget = KeyboardOnlySpinBox()
            widget.setRange(minimum, maximum)
            widget.setValue(value)
            return widget

        epochs = QLineEdit("Not used (step-based)")
        epochs.setReadOnly(True)
        epochs.setToolTip("ACT runs for a fixed number of optimizer steps; the training dataset does not use image epochs.")
        form.insertRow(0, "Epochs", epochs)

        try:
            initial_lr = float(self.learning_rate.text().strip())
        except ValueError:
            initial_lr = 1e-5
        lr0 = float_control(initial_lr, 0.000001, 10.0, 0.0001, 6)
        form.addRow("Initial learning rate", lr0)
        lrf = float_control(self.lr_final_fraction, 0.000001, 1.0, 0.001, 6)
        lrf.setEnabled(False)
        lrf.setToolTip("Applied when cosine learning-rate scheduling is enabled.")
        form.addRow("Final LR fraction", lrf)
        momentum = float_control(self.momentum, 0.0, 0.999, 0.001, 4)
        momentum.setEnabled(False)
        momentum.setToolTip("The bin-picking trainer keeps AdamW beta1 fixed.")
        form.addRow("Momentum / Adam beta1", momentum)
        try:
            weight_decay_value = float(self.weight_decay.text().strip())
        except ValueError:
            weight_decay_value = 1e-4
        weight_decay = float_control(weight_decay_value, 0.0, 1.0, 0.0001, 6)
        weight_decay.setToolTip("AdamW weight decay; set to 0 to disable.")
        form.addRow("Weight decay", weight_decay)
        warmup_epochs = float_control(self.warmup_epochs, 0.0, 1000.0, 0.5, 1)
        warmup_epochs.setEnabled(False)
        warmup_epochs.setToolTip("Applied when cosine learning-rate scheduling is enabled.")
        form.addRow("Warmup epochs", warmup_epochs)
        warmup_momentum = float_control(self.warmup_momentum, 0.0, 1.0, 0.01, 3)
        warmup_momentum.setEnabled(False)
        warmup_momentum.setToolTip("Ultralytics/SGD-only. ACT uses AdamW without a momentum warmup parameter.")
        form.addRow("Warmup momentum", warmup_momentum)
        warmup_bias_lr = float_control(self.warmup_bias_lr, 0.0, 1.0, 0.01, 3)
        warmup_bias_lr.setEnabled(False)
        warmup_bias_lr.setToolTip("Ultralytics detector-specific bias warmup; ACT does not expose this parameter.")
        form.addRow("Warmup bias LR", warmup_bias_lr)
        dataset_fraction = float_control(self.dataset_fraction, 0.01, 1.0, 0.05, 2)
        dataset_fraction.setToolTip("Subsamples episodes deterministically before the train/validation/test split.")
        form.addRow("Dataset fraction", dataset_fraction)
        close_mosaic = integer_control(10, 0, 100000)
        close_mosaic.setEnabled(False)
        close_mosaic.setToolTip("Mosaic is an object-detection augmentation and is not used by ACT.")
        form.addRow("Disable mosaic in final epochs", close_mosaic)
        workers = integer_control(self.workers.value(), 0, 128)
        form.addRow("Data loader workers", workers)
        save_period = integer_control(self.save_freq.value(), 1, 1000000)
        save_period.setToolTip("LeRobot saves by training step rather than epoch.")
        form.addRow("Checkpoint period (steps)", save_period)

        checks = {}
        for key, label, checked, enabled, tip in (
            ("cos_lr", "Cosine learning-rate schedule", self.cosine_schedule, False,
             "This bin-picking ACT trainer does not expose scheduler overrides."),
            ("amp", "Automatic mixed precision", self.use_amp.isChecked(), True,
             "Uses mixed precision on supported CUDA devices."),
            ("deterministic", "Deterministic training", self.deterministic.isChecked(), False,
             "This bin-picking ACT trainer does not expose a deterministic-mode flag."),
            ("cache", "Cache training images", False, False,
             "The ACT pipeline does not expose SkinEye's image-cache control."),
            ("val", "Validate during training", False, False,
             "ACT validation is an offline evaluation after training in this project."),
            ("plots", "Write training plots", False, False,
             "The ACT trainer does not generate SkinEye/Ultralytics training plots."),
        ):
            check = KeyboardOnlyCheckBox(label)
            check.setChecked(checked)
            check.setEnabled(enabled)
            check.setToolTip(tip)
            checks[key] = check
            form.addRow(check)

        buttons = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        if dialog.exec() != QDialog.Accepted:
            return
        self.learning_rate.setText(f"{lr0.value():.8g}")
        self.weight_decay.setText(f"{weight_decay.value():.8g}")
        self.use_amp.setChecked(checks["amp"].isChecked())
        self.dataset_fraction = dataset_fraction.value()
        self.workers.setValue(workers.value())
        self.save_freq.setValue(save_period.value())
        self.statusBar().showMessage("Advanced training options saved for the next run")

    def _dataset_dir(self) -> Path:
        return DATA_DIR / self.dataset.text().strip()

    def _dataset_repo_id(self) -> str:
        manifest_path = self._dataset_dir() / "huggingface_dataset.json"
        if manifest_path.is_file():
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                if manifest.get("repo_id"):
                    return str(manifest["repo_id"])
            except (OSError, ValueError):
                pass
        return self.dataset.text().strip()

    def _update_dataset_status(self) -> None:
        dataset_dir = self._dataset_dir()
        ready = (dataset_dir / "meta" / "info.json").is_file()
        message = (
            f"Dataset available: {dataset_dir}"
            if ready else
            f"Could not find {dataset_dir / 'meta' / 'info.json'}. Select or prepare a dataset first."
        )
        self.dataset_status.setText(message)
        self.dataset_status.setStyleSheet("color: #26734d;" if ready else "color: #9a5a00;")
        if hasattr(self, "preparation_status"):
            self.preparation_status.setText(message)
            self.preparation_status.setStyleSheet("color: #26734d;" if ready else "color: #9a5a00;")

    def _refresh_datasets(self) -> None:
        previous_dataset = self.dataset.text().strip()
        self.local_dataset_combo.clear()
        datasets = []
        if DATA_DIR.is_dir():
            datasets = sorted(
                folder for folder in DATA_DIR.iterdir()
                if folder.is_dir() and (folder / "meta" / "info.json").is_file()
            )
        for folder in datasets:
            self.local_dataset_combo.addItem(folder.name, str(folder))
        if datasets:
            selected_index = self.local_dataset_combo.findText(previous_dataset)
            self.local_dataset_combo.setCurrentIndex(max(0, selected_index))
        else:
            self.dataset.clear()
            self.dataset_details.setPlainText(
                "No local datasets found yet.\n\n"
                "Use demos/09_voice_bin_picking/tools/collect_demos.py to record and validate bin-picking episodes before training."
            )
            self.task_group_list.clear()
            self.episode_selection_list.clear()
            self._update_episode_selection_status()
        self._update_summary()

    def _refresh_hub_datasets(self) -> None:
        self._launch_hub_action("list")

    def _hub_dataset_selected(self, current: QListWidgetItem | None, _previous: QListWidgetItem | None) -> None:
        self._hub_dataset_selection_changed(current)

    def _hub_dataset_selection_changed(self, *_args, update_status: bool = True) -> None:
        selected = sum(
            self.hub_dataset_list.item(row).checkState() == Qt.CheckState.Checked
            for row in range(self.hub_dataset_list.count())
        )
        total = self.hub_dataset_list.count()
        if update_status:
            self.hub_status.setText(f"{selected} of {total} datasets checked.")
        self.hub_download_button.setEnabled(
            selected > 0 and self.hub_process.state() == QProcess.NotRunning
        )

    def _set_hub_selection(self, selected: bool) -> None:
        state = Qt.CheckState.Checked if selected else Qt.CheckState.Unchecked
        self.hub_dataset_list.blockSignals(True)
        for row in range(self.hub_dataset_list.count()):
            self.hub_dataset_list.item(row).setCheckState(state)
        self.hub_dataset_list.blockSignals(False)
        self._hub_dataset_selection_changed()

    def _set_hub_task_selection(self, task: str) -> None:
        keywords = {"pickcube": "pickcube", "updown": "updown"}
        display_names = {"pickcube": "Pick-cube", "updown": "Up-down"}
        keyword = keywords.get(task)
        if keyword is None:
            return
        selected_count = 0
        self.hub_dataset_list.blockSignals(True)
        for row in range(self.hub_dataset_list.count()):
            item = self.hub_dataset_list.item(row)
            repo_id = str(item.data(Qt.ItemDataRole.UserRole) or "")
            normalized_name = re.sub(r"[^a-z0-9]", "", repo_id.casefold())
            matches = keyword in normalized_name
            item.setCheckState(Qt.CheckState.Checked if matches else Qt.CheckState.Unchecked)
            selected_count += int(matches)
        self.hub_dataset_list.blockSignals(False)
        self._hub_dataset_selection_changed()
        if selected_count:
            self.hub_status.setText(
                f"Selected {selected_count} {display_names[task]} dataset(s) by name. "
                "You can still change individual selections."
            )
        else:
            self.hub_status.setText(f"No dataset repository name matched {display_names[task]}.")

    def _download_hub_dataset(self) -> None:
        self._hub_download_queue = [
            str(item.data(Qt.ItemDataRole.UserRole))
            for row in range(self.hub_dataset_list.count())
            if (item := self.hub_dataset_list.item(row)).checkState() == Qt.CheckState.Checked
        ]
        if not self._hub_download_queue:
            return
        self._hub_downloaded = []
        self._hub_download_failed = []
        self._download_next_hub_dataset()

    def _download_next_hub_dataset(self) -> None:
        if self._hub_download_queue:
            self._hub_current_repo = self._hub_download_queue.pop(0)
            self._launch_hub_action("download", repo_id=self._hub_current_repo, clear_log=False)
            self.hub_status.setText(
                f"Downloading {self._hub_current_repo} ({len(self._hub_downloaded) + len(self._hub_download_failed) + 1} of "
                f"{len(self._hub_downloaded) + len(self._hub_download_failed) + len(self._hub_download_queue) + 1})…"
            )
            return
        summary = f"Finished: {len(self._hub_downloaded)} downloaded"
        if self._hub_download_failed:
            summary += f", {len(self._hub_download_failed)} failed"
        self.hub_status.setText(summary + ". Select one downloaded dataset in the local list for each training run.")
        self.hub_refresh_button.setEnabled(True)
        self._hub_dataset_selection_changed(update_status=False)

    def _launch_hub_action(self, action: str, repo_id: str | None = None, clear_log: bool = True) -> None:
        if self.hub_process.state() != QProcess.NotRunning:
            return
        if not HF_DATASET_HELPER.is_file():
            self.hub_status.setText(f"Hugging Face helper not found: {HF_DATASET_HELPER}")
            return
        args = [str(HF_DATASET_HELPER), action, "--author", "MikelRubi", "--data-root", str(DATA_DIR)]
        if repo_id:
            args.extend(["--repo-id", str(repo_id)])
        self._hub_action = action
        self._hub_output = ""
        if clear_log:
            self.hub_log.clear()
        self.hub_status.setText("Loading Hugging Face datasets…" if action == "list" else f"Downloading {repo_id}…")
        self.hub_refresh_button.setEnabled(False)
        self.hub_download_button.setEnabled(False)
        self.hub_process.start(sys.executable, args)

    def _hub_read_output(self) -> None:
        text = bytes(self.hub_process.readAllStandardOutput()).decode("utf-8", errors="replace")
        self._hub_output += text
        cursor = self.hub_log.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        cursor.insertText(text)
        self.hub_log.setTextCursor(cursor)
        self.hub_log.ensureCursorVisible()

    def _hub_finished(self, exit_code: int, _exit_status: QProcess.ExitStatus) -> None:
        results = re.findall(r"@@HF_RESULT@@(\{[^\r\n]*\})", self._hub_output)
        result = None
        if results:
            try:
                result = json.loads(results[-1])
            except ValueError:
                result = None
        self.hub_refresh_button.setEnabled(True)
        if not result or not result.get("ok") or exit_code != 0:
            message = result.get("error") if result else "Could not read the Hugging Face helper response."
            if self._hub_action == "download":
                self._hub_download_failed.append(self._hub_current_repo)
                self.hub_log.appendPlainText(f"Download failed for {self._hub_current_repo}: {message}")
                self._download_next_hub_dataset()
            else:
                self.hub_status.setText(
                    f"Hugging Face request failed: {message} Sign in with `hf auth login` if the dataset is private."
                )
                self._hub_dataset_selected(self.hub_dataset_list.currentItem(), None)
            return
        if self._hub_action == "list":
            datasets = result.get("datasets", [])
            self.hub_dataset_list.clear()
            for dataset in datasets:
                repo_id = str(dataset.get("id", ""))
                storage = dataset.get("used_storage")
                size_text = self._format_byte_size(storage) if isinstance(storage, (int, float)) else "size unknown"
                visibility = "private" if dataset.get("private") else "public"
                item = QListWidgetItem(f"{repo_id}  ·  {visibility}  ·  {size_text}")
                item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                item.setCheckState(Qt.CheckState.Unchecked)
                item.setData(Qt.ItemDataRole.UserRole, repo_id)
                self.hub_dataset_list.addItem(item)
            self.hub_status.setText(
                f"Found {len(datasets)} dataset repositories for {result.get('author', 'MikelRubi')}. "
                "Check the repositories you want to download. Private repositories require a Hugging Face login on this computer."
            )
        elif self._hub_action == "download":
            local_name = result.get("local_name", "")
            if result.get("repo_id"):
                self._hub_downloaded.append(str(result["repo_id"]))
            self._refresh_datasets()
            if local_name:
                local_index = self.local_dataset_combo.findText(local_name)
                if local_index >= 0:
                    self.local_dataset_combo.setCurrentIndex(local_index)
            for row in range(self.hub_dataset_list.count()):
                item = self.hub_dataset_list.item(row)
                if item.data(Qt.ItemDataRole.UserRole) == result.get("repo_id"):
                    item.setCheckState(Qt.CheckState.Unchecked)
                    break
            if self._hub_download_queue:
                self._download_next_hub_dataset()
                return
            self._download_next_hub_dataset()
            return
        self._hub_dataset_selected(self.hub_dataset_list.currentItem(), None)

    def _hub_process_error(self, error: QProcess.ProcessError) -> None:
        if error == QProcess.ProcessError.FailedToStart:
            self.hub_refresh_button.setEnabled(True)
            message = f"Could not start Hugging Face helper: {self.hub_process.errorString()}"
            if self._hub_action == "download":
                self._hub_download_failed.append(self._hub_current_repo)
                self.hub_log.appendPlainText(message)
                self._download_next_hub_dataset()
            else:
                self._hub_dataset_selected(self.hub_dataset_list.currentItem(), None)
                self.hub_status.setText(message)

    @staticmethod
    def _format_byte_size(value: int | float) -> str:
        size = float(value)
        for unit in ("B", "KB", "MB", "GB", "TB"):
            if size < 1024 or unit == "TB":
                return f"{size:.1f} {unit}"
            size /= 1024
        return f"{size:.1f} TB"

    def _dataset_selected(self, index: int) -> None:
        if index < 0:
            return
        folder = Path(self.local_dataset_combo.currentData())
        self.dataset.setText(folder.name)
        info_path = folder / "meta" / "info.json"
        try:
            info = json.loads(info_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            self.dataset_details.setPlainText(f"Could not read dataset metadata:\n{exc}")
            self.task_group_list.clear()
            self.episode_selection_list.clear()
            self._update_episode_selection_status()
            return
        features = info.get("features") or {}
        feature_names = list(features) if isinstance(features, dict) else []
        details = [
            f"Name: {folder.name}",
            f"Path: {folder}",
            f"Robot type: {info.get('robot_type', 'not specified')}",
            f"Frequency: {info.get('fps', 'not specified')} Hz",
            f"Episodes: {info.get('total_episodes', 'not specified')}",
            f"Frames: {info.get('total_frames', 'not specified')}",
            "Features: " + (", ".join(feature_names) if feature_names else "not specified"),
        ]
        manifest_path = folder / "huggingface_dataset.json"
        if manifest_path.is_file():
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                details.extend((
                    f"Hugging Face dataset: {manifest.get('repo_id', 'not specified')}",
                    f"Hub revision: {manifest.get('revision', 'not specified')}",
                ))
            except (OSError, ValueError):
                details.append("Hugging Face manifest: could not be read")
        self.dataset_details.setPlainText("\n".join(details))
        self._refresh_task_groups(folder)
        self._update_dataset_status()
        self._update_summary()

    def _refresh_task_groups(self, folder: Path) -> None:
        self.task_group_list.clear()
        groups: dict[str, set[int]] = {}
        metadata_dir = folder / "meta"

        tasks_path = metadata_dir / "tasks.parquet"
        task_names: dict[int, str] = {}
        legacy_tasks = metadata_dir / "tasks.jsonl"
        if legacy_tasks.is_file():
            try:
                for line in legacy_tasks.read_text(encoding="utf-8").splitlines():
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    task_index = row.get("task_index", row.get("index"))
                    task_name = row.get("task") or row.get("name")
                    if task_index is not None and task_name:
                        task_names[int(task_index)] = str(task_name)
            except (OSError, ValueError, TypeError):
                task_names.clear()
        if tasks_path.is_file():
            try:
                import pyarrow.parquet as pq
                for row in pq.read_table(tasks_path).to_pylist():
                    task_index = row.get("task_index")
                    task_name = row.get("task") or row.get("name")
                    if task_index is not None and task_name:
                        task_names[int(task_index)] = str(task_name)
            except (ImportError, OSError, ValueError):
                pass

        episode_files = sorted((metadata_dir / "episodes").glob("chunk-*/file-*.parquet"))
        if episode_files:
            try:
                import pyarrow.parquet as pq
                for path in episode_files:
                    for row in pq.read_table(path).to_pylist():
                        episode = int(row.get("episode_index", -1))
                        if episode < 0:
                            continue
                        values = row.get("tasks") or []
                        if isinstance(values, str):
                            values = [values]
                        for value in values:
                            if isinstance(value, int):
                                value = task_names.get(value, str(value))
                            if isinstance(value, dict):
                                value = value.get("task") or value.get("name")
                            if value:
                                groups.setdefault(str(value), set()).add(episode)
                        task_index = row.get("task_index")
                        if task_index is not None and not values:
                            name = task_names.get(int(task_index), str(task_index))
                            groups.setdefault(name, set()).add(episode)
            except (ImportError, OSError, ValueError, TypeError):
                groups.clear()

        legacy_episodes = metadata_dir / "episodes.jsonl"
        if not groups and legacy_episodes.is_file():
            try:
                for line in legacy_episodes.read_text(encoding="utf-8").splitlines():
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    episode = int(row.get("episode_index", -1))
                    values = row.get("tasks") or []
                    if isinstance(values, str):
                        values = [values]
                    for value in values:
                        if isinstance(value, int):
                            value = task_names.get(value, str(value))
                        if isinstance(value, dict):
                            value = value.get("task") or value.get("name")
                        if value and episode >= 0:
                            groups.setdefault(str(value), set()).add(episode)
            except (OSError, ValueError, TypeError):
                groups.clear()

        if not groups:
            self.task_group_list.addItem("No task groups found; all dataset episodes will be available.")
        else:
            self.task_group_list.blockSignals(True)
            for name, episode_ids in sorted(groups.items()):
                item = QListWidgetItem(f"{name} ({len(episode_ids)} episodes)")
                item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                item.setCheckState(Qt.CheckState.Checked)
                item.setData(Qt.ItemDataRole.UserRole, sorted(episode_ids))
                item.setData(Qt.ItemDataRole.UserRole + 1, name)
                self.task_group_list.addItem(item)
            self.task_group_list.blockSignals(False)
        self._episode_group_map = {
            episode: {name for name, ids in groups.items() if episode in ids}
            for ids in groups.values() for episode in ids
        }
        self._populate_episode_selection(folder, self._episode_group_map)

    def _selected_task_group_names(self) -> list[str]:
        return [
            str(item.data(Qt.ItemDataRole.UserRole + 1))
            for row in range(self.task_group_list.count())
            if (item := self.task_group_list.item(row)).flags() & Qt.ItemFlag.ItemIsUserCheckable
            and item.checkState() == Qt.CheckState.Checked
        ]

    def _populate_episode_selection(self, folder: Path, episode_groups: dict[int, set[str]]) -> None:
        self.episode_selection_list.blockSignals(True)
        self.episode_selection_list.clear()
        try:
            info = json.loads((folder / "meta" / "info.json").read_text(encoding="utf-8"))
            count = int(info.get("total_episodes", 0))
        except (OSError, ValueError, TypeError):
            count = 0
        for episode in range(count):
            groups = sorted(episode_groups.get(episode, set()))
            group_text = f" · {' / '.join(groups)}" if groups else " · ungrouped"
            item = QListWidgetItem(f"Episode {episode:04d}{group_text}")
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Checked)
            item.setData(Qt.ItemDataRole.UserRole, episode)
            self.episode_selection_list.addItem(item)
        self.episode_selection_list.blockSignals(False)
        self._update_episode_selection_status()

    def _task_group_changed(self, _item: QListWidgetItem) -> None:
        selected_groups = set(self._selected_task_group_names())
        self.episode_selection_list.blockSignals(True)
        for row in range(self.episode_selection_list.count()):
            episode_item = self.episode_selection_list.item(row)
            episode = int(episode_item.data(Qt.ItemDataRole.UserRole))
            groups = self._episode_group_map.get(episode, set())
            if groups:
                episode_item.setCheckState(
                    Qt.CheckState.Checked if groups & selected_groups else Qt.CheckState.Unchecked
                )
        self.episode_selection_list.blockSignals(False)
        self._update_episode_selection_status()

    def _set_episode_selection(self, selected: bool) -> None:
        state = Qt.CheckState.Checked if selected else Qt.CheckState.Unchecked
        self.episode_selection_list.blockSignals(True)
        for row in range(self.episode_selection_list.count()):
            self.episode_selection_list.item(row).setCheckState(state)
        self.episode_selection_list.blockSignals(False)
        self._update_episode_selection_status()

    def _update_episode_selection_status(self, *_args) -> None:
        if not hasattr(self, "episode_selection_status"):
            return
        total = self.episode_selection_list.count()
        selected = sum(
            self.episode_selection_list.item(row).checkState() == Qt.CheckState.Checked
            for row in range(total)
        )
        self.episode_selection_status.setText(f"{selected} of {total} episodes selected")

    def _selected_dataset_episode_ids(self) -> list[int]:
        return [
            int(item.data(Qt.ItemDataRole.UserRole))
            for row in range(self.episode_selection_list.count())
            if (item := self.episode_selection_list.item(row)).checkState() == Qt.CheckState.Checked
        ]

    def _update_output_path(self) -> None:
        name = self.output_name.text().strip()
        if _NAME.fullmatch(name):
            stage = self.output_stage.currentData() if hasattr(self, "output_stage") else OUTPUT_TASKS[0]
            path = OUTPUTS_DIR / str(stage) / name
            suffix = " — already exists; it will not be overwritten" if path.exists() else ""
            self.output_path.setText(str(path) + suffix)
        else:
            self.output_path.setText("Use letters, numbers, periods, hyphens or underscores.")

    def _refresh_results(self) -> None:
        if not hasattr(self, "results_list"):
            return
        self.results_list.clear()
        runs = []
        for folder in self._training_run_dirs():
            checkpoint = folder / "checkpoints" / "last" / "pretrained_model"
            runs.append((folder, checkpoint if checkpoint.is_dir() else None))
        for folder, checkpoint in sorted(runs, key=lambda row: row[0].stat().st_mtime, reverse=True):
            modified = datetime.fromtimestamp(folder.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
            status = "checkpoint ready" if checkpoint else "no final checkpoint"
            stage = folder.parent.name if folder.parent.name in OUTPUT_TASKS else "Legacy"
            item = QListWidgetItem(f"[{stage}]  {folder.name}    ·    {modified}    ·    {status}")
            item.setData(Qt.ItemDataRole.UserRole, str(checkpoint) if checkpoint else None)
            item.setData(Qt.ItemDataRole.UserRole + 1, str(folder))
            self.results_list.addItem(item)
        if not runs:
            self.results_list.addItem("No training runs found yet.")
        elif self.results_list.count():
            self.results_list.setCurrentRow(0)
        self._refresh_evaluation_reports()
        self._refresh_binpick_reports()
        self._result_selected(self.results_list.currentItem(), None)
        self._update_summary()

    @staticmethod
    def _training_run_dirs() -> list[Path]:
        if not OUTPUTS_DIR.is_dir():
            return []
        runs = []
        for entry in OUTPUTS_DIR.iterdir():
            if not entry.is_dir():
                continue
            candidates = entry.iterdir() if entry.name in OUTPUT_TASKS else (entry,)
            for folder in candidates:
                if folder.is_dir() and (
                    (folder / "checkpoints").is_dir() or (folder / "studio_run_config.json").is_file()
                ):
                    runs.append(folder)
        return runs

    def _refresh_evaluation_reports(self) -> None:
        self.evaluation_list.clear()
        reports = sorted(
            EVALUATIONS_DIR.glob("eval_sorting_*.json"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        ) if EVALUATIONS_DIR.is_dir() else []
        for report_path in reports:
            try:
                record = json.loads(report_path.read_text(encoding="utf-8"))
                if "resumen" in record:
                    summary = record.get("resumen", {})
                    success_key = next((k for k in summary if "xito_seg" in str(k).lower()), None)
                    success = summary.get(success_key, "—") if success_key else "—"
                    name = report_path.stem.removeprefix("eval_sorting_")
                    self.evaluation_list.addItem(QListWidgetItem(f"{name}    ·    task success {success}"))
                    self.evaluation_list.item(self.evaluation_list.count() - 1).setData(Qt.ItemDataRole.UserRole, str(report_path))
                    continue
                run = record.get("run", {})
                summary = record.get("summary", {})
                name = run.get("name", report_path.stem)
                successes = summary.get("successes", 0)
                episodes = summary.get("episodes", 0)
                rate = float(summary.get("rate", 0)) * 100
                item = QListWidgetItem(f"{name}    ·    {successes}/{episodes} ({rate:.0f}%)")
                item.setData(Qt.ItemDataRole.UserRole, str(report_path))
                self.evaluation_list.addItem(item)
            except (OSError, ValueError, TypeError):
                continue
        if not reports:
            self.evaluation_list.addItem("No evaluation reports found yet.")
        elif self.evaluation_list.count():
            self.evaluation_list.setCurrentRow(0)
        self._evaluation_selected(self.evaluation_list.currentItem(), None)

    def _refresh_binpick_reports(self) -> None:
        if not hasattr(self, "binpick_report_list"):
            return
        previous = self.binpick_report_list.currentItem()
        previous_path = previous.data(Qt.ItemDataRole.UserRole) if previous else None
        reports = []
        if BINPICK_RESULTS_DIR.is_dir():
            reports.extend(BINPICK_RESULTS_DIR.glob("segmentation_*.json"))
            reports.extend(BINPICK_RESULTS_DIR.glob("eval_sorting_*.json"))
            reports.extend(BINPICK_RESULTS_DIR.glob("*offline_eval*.json"))
        if BINPICK_OUTPUTS_DIR.is_dir():
            reports.extend(BINPICK_OUTPUTS_DIR.rglob("offline_eval*.json"))
        reports = sorted(set(reports), key=lambda path: path.stat().st_mtime, reverse=True)
        self.binpick_report_list.blockSignals(True)
        self.binpick_report_list.clear()
        selected_row = 0
        for index, path in enumerate(reports):
            try:
                record = json.loads(path.read_text(encoding="utf-8"))
                label = self._binpick_report_label(path, record)
                item = QListWidgetItem(label)
                item.setData(Qt.ItemDataRole.UserRole, str(path))
                self.binpick_report_list.addItem(item)
                if str(path) == previous_path:
                    selected_row = index
            except (OSError, ValueError, TypeError):
                continue
        if not reports:
            self.binpick_report_list.addItem("No bin-picking reports found. Run segmentation or sorting evaluation first.")
        elif self.binpick_report_list.count():
            self.binpick_report_list.setCurrentRow(selected_row)
        self.binpick_report_list.blockSignals(False)
        self._binpick_report_selected(self.binpick_report_list.currentItem(), None)

    @staticmethod
    def _binpick_report_label(path: Path, record: dict) -> str:
        if path.name.startswith("segmentation_"):
            recall = record.get("recall", "—")
            precision = next((v for k, v in record.items() if str(k).lower().startswith("precisi")), "—")
            return f"Vision · {record.get('segmentador', path.stem)} · recall {recall} · precision {precision}"
        if path.name.startswith("eval_sorting_"):
            summary = record.get("resumen", {})
            success = next((v for k, v in summary.items() if "xito_seg" in str(k).lower()), "—")
            return f"End-to-end · {summary.get('política', summary.get('modo', path.stem))} · success {success}"
        summary = record.get("resumen", {})
        error = next((v for k, v in summary.items() if str(k).lower().startswith("error_pol")), "—")
        return f"ACT offline · policy action error {error} · {path.parent.name}"

    def _binpick_report_selected(self, current: QListWidgetItem | None, _previous: QListWidgetItem | None) -> None:
        if current is None:
            self.binpick_report_details.clear()
            self.binpick_metric_chart.set_values([])
            return
        report_path = current.data(Qt.ItemDataRole.UserRole)
        if not report_path:
            self.binpick_report_details.setPlainText("No bin-picking reports found yet.")
            self.binpick_metric_chart.set_values([])
            return
        path = Path(report_path)
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
            if path.name.startswith("segmentation_"):
                details = self._format_segmentation_report(record)
                chart_values = self._segmentation_chart_values(record)
                self.binpick_chart_note.setText("Vision rates and mask IoU (%) · compare occlusion and scene groups to find weak cases.")
            elif path.name.startswith("eval_sorting_"):
                details = self._format_sorting_report(record)
                chart_values = self._sorting_chart_values(record)
                self.binpick_chart_note.setText("Task, completion and grasp success rates (%) for this evaluation run.")
            else:
                details = self._format_policy_report(record)
                chart_values = self._policy_chart_values(record)
                self.binpick_chart_note.setText("Offline mean absolute action error · lower is better; compare policy with simple baselines.")
            self.binpick_metric_chart.set_values(chart_values)
            self.binpick_report_details.setPlainText(f"Report: {path.name}\n\n{details}\n\nFile: {path}")
        except (OSError, ValueError, TypeError) as error:
            self.binpick_report_details.setPlainText(f"Could not read report:\n{error}")

    @staticmethod
    def _as_float(value) -> float | None:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None
        return number if number == number and abs(number) != float("inf") else None

    @staticmethod
    def _ratio(value) -> float | None:
        if isinstance(value, str) and "/" in value:
            try:
                numerator, denominator = value.split("/", 1)
                denominator = float(denominator)
                return float(numerator) / denominator if denominator else None
            except ValueError:
                return None
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None
        return number if 0 <= number <= 1 else None

    @classmethod
    def _segmentation_chart_values(cls, record: dict) -> list[tuple[str, float, str]]:
        rows = []

        def add(label: str, value) -> None:
            number = cls._as_float(value)
            if number is not None:
                rows.append((label, number, f"{number * 100:.1f}%"))

        add("Precision", next((v for k, v in record.items() if str(k).lower().startswith("precisi")), None))
        add("Overall recall", record.get("recall"))
        add("Mean mask IoU", record.get("iou_medio"))
        for group, values in record.get("recall_por_visibilidad", {}).items():
            if isinstance(values, dict):
                add(f"Visible {group}", values.get("recall"))
        for group, values in record.get("recall_por_tipo_de_escena", {}).items():
            if isinstance(values, dict):
                add(f"Scene {group}", values.get("recall"))
        return rows

    @classmethod
    def _sorting_chart_values(cls, record: dict) -> list[tuple[str, float, str]]:
        summary = record.get("resumen", {})
        rows = []

        def add_ratio(label: str, value) -> None:
            ratio = cls._ratio(value)
            if ratio is not None:
                rows.append((label, ratio, f"{ratio * 100:.1f}%"))

        task_success = next((v for k, v in summary.items() if "xito_seg" in str(k).lower()), None)
        add_ratio("Task success", task_success)
        add_ratio("Completion diagnosis", summary.get("el_robot_acierta_al_creerlo"))
        for key, label in (("agarres_act", "ACT grasp success"), ("agarres_guion", "Scripted grasp success")):
            add_ratio(label, summary.get(key))
        return rows

    @classmethod
    def _policy_chart_values(cls, record: dict) -> list[tuple[str, float, str]]:
        summary = record.get("resumen", {})
        rows = []
        for key, raw_value in summary.items():
            normalized = str(key).lower()
            label = None
            if normalized.startswith("error_pol"):
                label = "Policy error"
            elif normalized.startswith("error_copiar"):
                label = "Copy-state baseline"
            elif normalized.startswith("error_mantener"):
                label = "Hold-action baseline"
            value = cls._as_float(raw_value)
            if label and value is not None and not any(existing[0] == label for existing in rows):
                rows.append((label, value, f"{value:.3f}"))
        return rows

    @staticmethod
    def _format_segmentation_report(record: dict) -> str:
        def find(prefix: str, default="—"):
            return next((value for key, value in record.items() if str(key).lower().startswith(prefix)), default)

        lines = [
            "PERCEPTION / INSTANCE SEGMENTATION",
            f"Split / images / IoU threshold: {record.get('conjunto', '—')} / {find('im')} / {record.get('iou_umbral', '—')}",
            f"Recall: {record.get('recall', '—')}    Precision: {find('precisi')}",
            f"Mean mask IoU: {record.get('iou_medio', '—')}    False positives: {find('falsos_positivos')}    False negatives: {find('falsos_negativos')}",
            f"Color accuracy: {find('acierto_de_color')}    Shape accuracy: {find('acierto_de_forma')}",
        ]
        position = find("error_de_posici")
        if isinstance(position, dict):
            lines.append(f"Position error (mm): mean {position.get('media', '—')}, p95 {position.get('p95', '—')}")
        latency = record.get("latencia_ms")
        if isinstance(latency, dict):
            lines.append(f"Inference latency (ms): mean {latency.get('media', '—')}, p95 {latency.get('p95', '—')}")
        for key, title in (("recall_por_visibilidad", "Recall by visibility"), ("recall_por_tipo_de_escena", "Recall by scene layout")):
            groups = record.get(key)
            if isinstance(groups, dict):
                lines.extend(("", title))
                lines.extend(f"  {label}: recall {values.get('recall', '—')} (n={values.get('n', '—')})"
                             for label, values in groups.items() if isinstance(values, dict))
        return "\n".join(lines)

    @staticmethod
    def _format_sorting_report(record: dict) -> str:
        summary = record.get("resumen", {})
        success = next((v for k, v in summary.items() if "xito_seg" in str(k).lower()), "—")
        lines = [
            "END-TO-END BIN-PICKING / SORTING",
            f"Policy: {summary.get('política', summary.get('tipo', '—'))}",
            f"Segmenter: {summary.get('segmentador', '—')}    Evaluations: {summary.get('ejecuciones', '—')}",
            f"Correct task success: {success}",
            f"Robot completion states: {json.dumps(summary.get('estado_del_robot', {}), ensure_ascii=False)}",
            f"ACT grasp success: {summary.get('agarres_act', '—')}    Scripted grasp success: {summary.get('agarres_guion', '—')}",
            f"Pieces in wrong bins: {summary.get('piezas_en_caja_equivocada', '—')}",
            f"Robot reports correct completion: {summary.get('el_robot_acierta_al_creerlo', '—')}",
            f"Recovery attempts: {summary.get('recuperaciones_totales', '—')}    Mean simulated cycle time (s): {summary.get('t_sim_medio_s', '—')}",
            "",
            "Per-scene breakdown is available in the JSON report and the individual episode records.",
        ]
        return "\n".join(lines)

    @staticmethod
    def _format_policy_report(record: dict) -> str:
        summary = record.get("resumen", {})
        lines = ["ACT POLICY / OFFLINE ACTION ERROR", f"Held-out episodes: {record.get('episodios', '—')}",
                 f"Frames evaluated: {record.get('fotogramas_evaluados', '—')}    Action chunk: {record.get('bloque', '—')}"]
        for key, value in summary.items():
            lines.append(f"{key}: {value}")
        errors = next((value for key, value in record.items()
                       if str(key).startswith("error_medio_por_articul")), None)
        if isinstance(errors, dict):
            for horizon, values in errors.items():
                lines.extend(("", horizon))
                for model, per_joint in values.items():
                    lines.append(f"  {model}: {json.dumps(per_joint, ensure_ascii=False)}")
        return "\n".join(lines)

    def _result_selected(self, current: QListWidgetItem | None, _previous: QListWidgetItem | None) -> None:
        checkpoint_ready = current is not None and current.data(Qt.ItemDataRole.UserRole) is not None
        output_path = current.data(Qt.ItemDataRole.UserRole + 1) if current is not None else None
        self.evaluate_button.setEnabled(bool(checkpoint_ready) and self.process.state() == QProcess.NotRunning)
        self.offline_evaluate_button.setEnabled(bool(checkpoint_ready) and self.process.state() == QProcess.NotRunning)
        self.open_output_button.setEnabled(bool(output_path))
        self.run_details.setPlainText(self._format_training_run(Path(output_path)) if output_path else "Select a training run.")

    def _format_training_run(self, folder: Path) -> str:
        checkpoint = folder / "checkpoints" / "last" / "pretrained_model"
        lines = [
            f"Run: {folder.name}",
            f"Updated: {datetime.fromtimestamp(folder.stat().st_mtime).strftime('%Y-%m-%d %H:%M')}",
            f"Output folder: {folder}",
            f"Final checkpoint: {'available' if checkpoint.is_dir() else 'not found'}",
        ]
        studio_config_path = folder / "studio_run_config.json"
        if studio_config_path.is_file():
            try:
                studio_config = json.loads(studio_config_path.read_text(encoding="utf-8"))
                lines.extend((
                    "",
                    "ACT bin-picking setup:",
                    f"  dataset: {studio_config.get('dataset', '—')}",
                    f"  observations: {studio_config.get('observation_mode', '—')}",
                    f"  episodes train/validation/test: {len(studio_config.get('train_episodes', []))} / "
                    f"{len(studio_config.get('validation_episodes', []))} / {len(studio_config.get('test_episodes', []))}",
                    f"  selected task groups: {', '.join(studio_config.get('task_groups', [])) or 'all'}",
                    f"  steps / batch / chunk: {studio_config.get('steps', '—')} / {studio_config.get('batch', '—')} / "
                    f"{studio_config.get('action_chunk', '—')}",
                    f"  learning rate: {studio_config.get('learning_rate', '—')}",
                    f"  backbone / hidden size / heads: {studio_config.get('vision_backbone', '—')} / "
                    f"{studio_config.get('dim_model', '—')} / {studio_config.get('attention_heads', '—')}",
                    f"  encoder / decoder layers: {studio_config.get('encoder_layers', '—')} / "
                    f"{studio_config.get('decoder_layers', '—')} · dropout {studio_config.get('dropout', '—')}",
                    f"  batch / AMP / weight decay: {studio_config.get('batch', '—')} / "
                    f"{studio_config.get('use_amp', '—')} / {studio_config.get('weight_decay', '—')}",
                ))
            except (OSError, ValueError):
                lines.append("ACT bin-picking setup could not be read.")
        config_path = next((path for path in (
            folder / "train_config.json",
            folder / "config.json",
            folder / "train_config.yaml",
            folder / "checkpoints" / "last" / "pretrained_model" / "train_config.json",
            folder / "checkpoints" / "last" / "pretrained_model" / "config.json",
        ) if path.is_file()), None)
        if config_path and config_path.suffix == ".json":
            try:
                config = json.loads(config_path.read_text(encoding="utf-8"))
                lines.extend(("", "Training configuration:"))
                for key in ("job_name", "seed", "steps", "batch_size", "num_workers", "save_freq", "log_freq"):
                    if key in config:
                        lines.append(f"  {key}: {config[key]}")
                policy = config.get("policy", {})
                if isinstance(policy, dict):
                    for key in (
                        "type", "chunk_size", "n_action_steps", "vision_backbone", "use_vae",
                        "optimizer_lr", "optimizer_weight_decay", "dim_model", "n_encoder_layers",
                        "n_decoder_layers", "n_heads", "latent_dim", "pretrained_path",
                    ):
                        if key in policy:
                            lines.append(f"  policy.{key}: {policy[key]}")
                dataset = config.get("dataset", {})
                if isinstance(dataset, dict):
                    if dataset.get("repo_id"):
                        lines.append(f"  dataset: {dataset['repo_id']}")
                    if dataset.get("eval_split"):
                        lines.append(f"  validation split: {float(dataset['eval_split']) * 100:.1f}%")
                    transforms = dataset.get("image_transforms", {})
                    if isinstance(transforms, dict) and transforms.get("enable"):
                        lines.append(f"  image transforms: {', '.join(transforms.get('tfs', {})) or 'enabled'}")
            except (OSError, ValueError):
                lines.extend(("", "Training configuration could not be read."))
        elif config_path:
            lines.extend(("", f"Training config: {config_path.name}"))

        metric_files = [*folder.rglob("*.jsonl"), *folder.rglob("*.csv")]
        if metric_files:
            latest = max(metric_files, key=lambda path: path.stat().st_mtime)
            lines.extend(("", f"Metric log: {latest.name}"))
            try:
                if latest.suffix == ".jsonl":
                    records = [line for line in latest.read_text(encoding="utf-8", errors="replace").splitlines() if line.strip()]
                    record = json.loads(records[-1]) if records else {}
                    lines.append("Latest logged values:")
                    lines.extend(f"  {key}: {value}" for key, value in record.items() if isinstance(value, (str, int, float, bool)))
                else:
                    import csv
                    with latest.open(newline="", encoding="utf-8", errors="replace") as handle:
                        rows = list(csv.DictReader(handle))
                    if rows:
                        lines.append("Latest logged values:")
                        lines.extend(f"  {key}: {value}" for key, value in rows[-1].items() if value)
            except (OSError, ValueError, IndexError):
                lines.append("Could not read the latest metric record.")
        else:
            lines.extend(("", "No training metric log was found in this run folder."))
        return "\n".join(lines)

    def _evaluation_selected(self, current: QListWidgetItem | None, _previous: QListWidgetItem | None) -> None:
        if current is None:
            self.evaluation_details.clear()
            return
        report_path = current.data(Qt.ItemDataRole.UserRole)
        if not report_path:
            self.evaluation_details.setPlainText("No evaluation report selected.")
            return
        try:
            record = json.loads(Path(report_path).read_text(encoding="utf-8"))
            if "resumen" in record:
                self.evaluation_details.setPlainText(self._format_sorting_report(record))
                return
            run = record.get("run", {})
            summary = record.get("summary", {})
            episodes = record.get("episodes", [])
            lines = [
                f"Evaluation: {run.get('name', Path(report_path).stem)}",
                f"Date (UTC): {run.get('utc', 'not recorded')}",
                f"Policy type: {run.get('type', 'not recorded')}",
                f"Policy inputs: {run.get('inputs', 'not recorded')}",
                f"Test set: {run.get('spots_description', run.get('spots', 'not recorded'))}",
                f"Episodes: {summary.get('episodes', len(episodes))}",
                f"Successes: {summary.get('successes', 0)}",
                f"Success rate: {float(summary.get('rate', 0)) * 100:.1f}%",
                f"Failures: {summary.get('failures') or 'None'}",
                f"Report file: {report_path}",
            ]
            self.evaluation_details.setPlainText("\n".join(lines))
        except (OSError, ValueError, TypeError) as error:
            self.evaluation_details.setPlainText(f"Could not read evaluation report:\n{error}")

    def _selected_checkpoint(self) -> tuple[Path, Path] | None:
        item = self.results_list.currentItem()
        if item is None:
            return None
        checkpoint = item.data(Qt.ItemDataRole.UserRole)
        output = item.data(Qt.ItemDataRole.UserRole + 1)
        if not checkpoint or not output:
            return None
        return Path(checkpoint), Path(output)

    def _open_selected_output(self) -> None:
        selected = self._selected_checkpoint()
        if selected:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(selected[1])))

    def _start_training(self) -> None:
        python = Path(self.python_path.text().strip().strip('"'))
        dataset = self.dataset.text().strip()
        output = self.output_name.text().strip()
        learning_rate = self.learning_rate.text().strip()
        if not python.is_file():
            self._show_error("Select an existing python.exe executable.")
            return
        if not TRAIN_SCRIPT.is_file():
            self._show_error(f"Training script not found:\n{TRAIN_SCRIPT}")
            return
        if not _NAME.fullmatch(dataset):
            self._show_error("Dataset names may contain only letters, numbers, periods, hyphens and underscores.")
            return
        dataset_info_path = self._dataset_dir() / "meta" / "info.json"
        if not dataset_info_path.is_file():
            self._show_error("The selected local dataset does not exist. Record demonstrations or select another dataset.")
            return
        if not _NAME.fullmatch(output):
            self._show_error("Run names may contain only letters, numbers, periods, hyphens and underscores.")
            return
        task_stage = str(self.output_stage.currentData())
        output_dir = OUTPUTS_DIR / task_stage / output
        if output_dir.exists():
            self._show_error(f"The output folder already exists and will not be overwritten:\n{output_dir}")
            return
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        try:
            learning_rate_value = float(learning_rate)
            weight_decay_value = float(self.weight_decay.text().strip())
            backbone_lr_value = float(self.backbone_lr.text().strip())
            dropout_value = float(self.dropout.text().strip())
            kl_weight_value = float(self.kl_weight.text().strip())
            if learning_rate_value <= 0 or backbone_lr_value <= 0:
                raise ValueError
        except ValueError:
            self._show_error("Learning rates must be positive, weight decay nonnegative, and numeric ACT settings must be valid.")
            return
        if weight_decay_value < 0 or kl_weight_value < 0:
            self._show_error("Weight decay and KL loss weight must be greater than or equal to zero.")
            return
        if not 0.0 <= dropout_value < 1.0:
            self._show_error("Dropout must be at least 0 and less than 1.")
            return
        if self.dim_model.value() % self.attention_heads.value():
            self._show_error("Transformer hidden dimension must be divisible by the number of attention heads.")
            return
        if self.action_steps.value() > self.chunk.value():
            self._show_error("Actions per prediction cannot exceed the action chunk size.")
            return
        if self.temporal_ensemble.isChecked() and self.action_steps.value() != 1:
            self._show_error("Temporal action ensembling requires exactly 1 action per prediction.")
            return
        if self.training_mode.currentData() == "transfer":
            pretrained = Path(self.pretrained_policy.text().strip().strip('"'))
            if not pretrained.is_dir():
                self._show_error("Select an existing pretrained ACT checkpoint folder for fine-tuning.")
                return
            if not any((pretrained / name).is_file() for name in ("config.json", "policy_config.json")):
                self._show_error("The selected folder does not contain an ACT policy config file.")
                return
        try:
            episode_split = self._prepare_episode_split()
        except (OSError, ValueError, json.JSONDecodeError) as error:
            self._show_error(str(error))
            return
        args = [
            str(TRAIN_SCRIPT),
            "--dataset-root", str(self._dataset_dir()),
            "--repo-id", episode_split["repo_id"],
            "--steps", str(self.steps.value()),
            "--batch", str(self.batch.value()),
            "--chunk", str(self.chunk.value()),
            "--action-steps", str(self.action_steps.value()),
            "--device", self.device.currentText(),
            "--output", str(output_dir),
            "--seed", str(self.seed.value()),
            "--obs-mode", str(self.observation_mode.currentData()),
            "--kl-weight", self.kl_weight.text().strip(),
            "--vision-backbone", str(self.backbone.currentData()),
            "--feedforward-activation", self.feedforward_activation.currentText(),
            "--weight-decay", str(weight_decay_value),
            "--backbone-lr", str(backbone_lr_value),
            "--dropout", str(dropout_value),
            "--dim-model", str(self.dim_model.value()),
            "--attention-heads", str(self.attention_heads.value()),
            "--feedforward-dim", str(self.feedforward_dim.value()),
            "--encoder-layers", str(self.encoder_layers.value()),
            "--decoder-layers", str(self.decoder_layers.value()),
            "--latent-dim", str(self.latent_dim.value()),
            "--vae-encoder-layers", str(self.vae_encoder_layers.value()),
            "--workers", str(self.workers.value()),
            "--save-freq", str(self.save_freq.value()),
            "--log-freq", str(self.log_freq.value()),
            "--episodes", self._compress_episode_ids(episode_split["train_episodes"]),
        ]
        args.extend(["--lr", learning_rate])
        if not self.pretrained_backbone.isChecked():
            args.append("--no-pretrained-backbone")
        if self.pre_norm.isChecked():
            args.append("--pre-norm")
        if self.replace_final_stride.isChecked():
            args.append("--replace-final-stride")
        if self.use_amp.isChecked():
            args.append("--use-amp")
        if self.temporal_ensemble.isChecked():
            args.extend(["--temporal-ensemble-coeff", str(self.temporal_ensemble_coeff.value())])
        if self.training_mode.currentData() == "transfer":
            args.extend(["--init-from", str(pretrained)])
        if self.no_vae.isChecked():
            args.append("--no-vae")
        if not self.use_robot_state.isChecked():
            args.append("--no-state")
        self._pending_training_metadata = {
            **episode_split,
            "output": str(output_dir),
            "task_stage": task_stage,
            "device": self.device.currentText(),
            "steps": self.steps.value(),
            "batch": self.batch.value(),
            "workers": self.workers.value(),
            "save_freq": self.save_freq.value(),
            "log_freq": self.log_freq.value(),
            "dataset_fraction": self.dataset_fraction,
            "action_chunk": self.chunk.value(),
            "action_steps": self.action_steps.value(),
            "learning_rate": learning_rate_value,
            "kl_weight": kl_weight_value,
            "vision_backbone": self.backbone.currentData(),
            "feedforward_activation": self.feedforward_activation.currentText(),
            "replace_final_stride_with_dilation": self.replace_final_stride.isChecked(),
            "pretrained_backbone_weights": self.pretrained_backbone.isChecked(),
            "weight_decay": weight_decay_value,
            "backbone_learning_rate": backbone_lr_value,
            "dropout": dropout_value,
            "dim_model": self.dim_model.value(),
            "attention_heads": self.attention_heads.value(),
            "feedforward_dim": self.feedforward_dim.value(),
            "encoder_layers": self.encoder_layers.value(),
            "decoder_layers": self.decoder_layers.value(),
            "latent_dim": self.latent_dim.value(),
            "vae_encoder_layers": self.vae_encoder_layers.value(),
            "pre_norm": self.pre_norm.isChecked(),
            "use_amp": self.use_amp.isChecked(),
            "temporal_ensemble_coeff": self.temporal_ensemble_coeff.value() if self.temporal_ensemble.isChecked() else None,
            "use_vae": not self.no_vae.isChecked(),
            "use_robot_state": self.use_robot_state.isChecked(),
            "fine_tuned_from": str(pretrained) if self.training_mode.currentData() == "transfer" else None,
        }
        self._launch_process(python, args, "training", output)

    def _start_evaluation(self) -> None:
        selected = self._selected_checkpoint()
        if not selected:
            self._show_error("Select a checkpoint from the list.")
            return
        python = Path(self.python_path.text().strip().strip('"'))
        if not python.is_file():
            self._show_error("Select an existing python.exe in the Training section.")
            return
        if not EVALUATE_SCRIPT.is_file():
            self._show_error(f"Evaluation script not found:\n{EVALUATE_SCRIPT}")
            return
        checkpoint, output_dir = selected
        if not checkpoint.is_dir():
            self._show_error(f"Checkpoint not found:\n{checkpoint}")
            return
        segmenter = self.eval_segmenter.text().strip()
        if not segmenter:
            self._show_error("Enter a segmenter name, for example hsv or yolo:weights.pt.")
            return
        report_name = f"studio_{output_dir.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        args = [
            str(EVALUATE_SCRIPT),
            "--policy", f"act:{checkpoint}",
            "--segmenter", segmenter,
            "--seeds", str(self.eval_episodes.value()),
            "--name", report_name,
        ]
        if self.eval_headless.isChecked():
            args.append("--headless")
        self._launch_process(python, args, "evaluation", output_dir.name)

    def _start_offline_evaluation(self) -> None:
        selected = self._selected_checkpoint()
        if not selected:
            self._show_error("Select a trained ACT checkpoint first.")
            return
        checkpoint, output_dir = selected
        split_path = output_dir / "studio_run_config.json"
        if not split_path.is_file():
            self._show_error("This run has no saved episode split. Train it from the studio or choose a run with held-out episodes.")
            return
        try:
            run_config = json.loads(split_path.read_text(encoding="utf-8"))
            episode_ids = run_config.get("test_episodes") or run_config.get("validation_episodes") or []
        except (OSError, ValueError) as error:
            self._show_error(f"Could not read the held-out episode list:\n{error}")
            return
        if not episode_ids:
            self._show_error("This run has no validation or test episodes held out from training.")
            return
        python = Path(self.python_path.text().strip().strip('"'))
        if not python.is_file():
            self._show_error("Select an existing Python executable in the Training section.")
            return
        if not OFFLINE_EVAL_SCRIPT.is_file():
            self._show_error(f"Offline ACT evaluator not found:\n{OFFLINE_EVAL_SCRIPT}")
            return
        output_path = output_dir / f"offline_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        args = [
            str(OFFLINE_EVAL_SCRIPT),
            "--policy", str(checkpoint),
            "--dataset-root", str(DATA_DIR / run_config["dataset"]),
            "--repo-id", run_config.get("repo_id", run_config["dataset"]),
            "--episodes", self._compress_episode_ids(episode_ids),
            "--device", self.device.currentText(),
            "--out", str(output_path),
        ]
        self._launch_process(python, args, "offline evaluation", output_dir.name)

    def _launch_process(self, python: Path, args: list[str], kind: str, name: str) -> None:
        if self.process.state() != QProcess.NotRunning:
            return
        self.log.clear()
        self._live_output_buffer = ""
        self._metric_step = 0
        self.live_loss_chart.clear()
        for label in self.live_metric_labels.values():
            label.setText("—")
        self._task_kind = kind
        self._task_output = name
        self._task_total_steps = self.steps.value() if kind == "training" and hasattr(self, "steps") else 0
        if self._task_total_steps:
            self.progress.setRange(0, self._task_total_steps)
            self.progress.setValue(0)
        else:
            self.progress.setRange(0, 0)
        self._append_text(f"Python: {python}\nCommand: {python} {' '.join(args)}\n\n")
        environment = QProcessEnvironment.systemEnvironment()
        environment.insert("PYTHONIOENCODING", "utf-8:replace")
        self.process.setProcessEnvironment(environment)
        self.process.setWorkingDirectory(str(REPO_ROOT))
        self.process.start(str(python), args)
        self.task_label.setText(f"{kind.title()} running · {name}")
        self.stop_button.setEnabled(True)
        self.progress.show()
        self.metric_refresh_timer.start()
        self.evaluate_button.setEnabled(False)
        self.start_button.setEnabled(False)
        self.statusBar().showMessage(f"{kind.title()} running")
        self._show_page(5)

    def _stop_process(self) -> None:
        if self.process.state() == QProcess.NotRunning:
            return
        answer = QMessageBox.question(
            self,
            "Stop process",
            "Stop the process now? If interrupted while saving a checkpoint, the checkpoint may be incomplete.",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if answer != QMessageBox.Yes:
            return
        self.process.terminate()
        QTimer.singleShot(5000, self._kill_if_running)
        self.statusBar().showMessage("Stopping process…")

    def _kill_if_running(self) -> None:
        if self.process.state() != QProcess.NotRunning:
            self.process.kill()

    def _read_output(self) -> None:
        text = bytes(self.process.readAllStandardOutput()).decode("utf-8", errors="replace")
        self._append_text(text)
        self._consume_live_metrics(text)

    def _consume_live_metrics(self, text: str) -> None:
        self._live_output_buffer += text
        lines = self._live_output_buffer.splitlines(keepends=True)
        self._live_output_buffer = ""
        if lines and not lines[-1].endswith(("\n", "\r")):
            self._live_output_buffer = lines.pop()
        patterns = {
            "loss": r"\bloss\s*[:=]\s*([-+0-9.eE]+)",
            "eval_loss": r"\beval_loss\s*[:=]\s*([-+0-9.eE]+)",
            "grad_norm": r"\b(?:grad_norm|grdn)\s*[:=]\s*([-+0-9.eE]+)",
            "lr": r"\blr\s*[:=]\s*([-+0-9.eE]+)",
            "samples_per_s": r"\b(?:samples_per_s|smp/s)\s*[:=]\s*([-+0-9.eE]+)",
            "gpu_mem_gb": r"\b(?:gpu_mem_gb|mem_gb)\s*[:=]\s*([-+0-9.eE]+)",
        }
        for line in lines:
            step_match = re.search(r"\bstep\s*[:=]?\s*(\d+)", line, re.IGNORECASE)
            if step_match:
                self._metric_step = int(step_match.group(1))
                self.live_metric_labels["step"].setText(
                    f"{self._metric_step} / {self._task_total_steps}" if self._task_total_steps else str(self._metric_step)
                )
                if self._task_total_steps:
                    self.progress.setValue(min(self._metric_step, self._task_total_steps))
            for key, pattern in patterns.items():
                match = re.search(pattern, line, re.IGNORECASE)
                if match:
                    raw_value = match.group(1)
                    self.live_metric_labels[key].setText(raw_value)
                    if key in {"loss", "eval_loss"} and self._metric_step:
                        try:
                            self.live_loss_chart.add_value(
                                "Training loss" if key == "loss" else "Validation loss",
                                self._metric_step,
                                float(raw_value),
                            )
                        except ValueError:
                            pass

    def _append_text(self, text: str) -> None:
        cursor = self.log.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        cursor.insertText(text)
        self.log.setTextCursor(cursor)
        self.log.ensureCursorVisible()

    def _finished(self, exit_code: int, _exit_status: QProcess.ExitStatus) -> None:
        self._consume_live_metrics("\n")
        self.metric_refresh_timer.stop()
        self.stop_button.setEnabled(False)
        self.progress.hide()
        self.start_button.setEnabled(True)
        self.evaluate_button.setEnabled(self._selected_checkpoint() is not None)
        if exit_code == 0:
            message = f"{self._task_kind.title()} completed successfully · {self._task_output}"
            self._append_text(f"\n{message}\n")
            self.statusBar().showMessage(message)
            self.task_label.setText(message)
        else:
            message = f"{self._task_kind.title()} exited with code {exit_code}"
            self._append_text(f"\n{message}\n")
            self.statusBar().showMessage(message)
            self.task_label.setText(message)
        if self._task_kind == "training" and exit_code == 0 and self._pending_training_metadata:
            metadata_path = Path(self._pending_training_metadata["output"]) / "studio_run_config.json"
            try:
                metadata_path.write_text(json.dumps(self._pending_training_metadata, indent=2), encoding="utf-8")
                self._append_text(f"Saved held-out episode split: {metadata_path}\n")
            except OSError as error:
                self._append_text(f"Could not save episode split metadata: {error}\n")
        self._pending_training_metadata = None
        self._refresh_results()
        self._update_summary()

    def _process_error(self, error: QProcess.ProcessError) -> None:
        if error == QProcess.ProcessError.FailedToStart:
            self.metric_refresh_timer.stop()
            message = self.process.errorString()
            self._append_text(f"\nCould not start process: {message}\n")
            self.stop_button.setEnabled(False)
            self.start_button.setEnabled(True)
            self.progress.hide()
            self.task_label.setText("Could not start process")
            self.statusBar().showMessage("Could not start process")

    def _update_summary(self) -> None:
        if not hasattr(self, "stat_dataset_count"):
            return
        count = sum(1 for _ in DATA_DIR.glob("*/meta/info.json")) if DATA_DIR.is_dir() else 0
        self.stat_dataset_count[1].setText(str(count))
        self.stat_dataset_count[2].setText(
            "local datasets available" if count else "no local datasets yet"
        )
        latest = None
        if OUTPUTS_DIR.is_dir():
            folders = self._training_run_dirs()
            if folders:
                latest = max(folders, key=lambda path: path.stat().st_mtime)
        self.stat_latest_run[1].setText(latest.name if latest else "—")
        if hasattr(self, "python_path"):
            raw_python_path = self.python_path.text().strip()
            if raw_python_path:
                python_path = Path(raw_python_path)
                environment_name = python_path.parent.parent.name if python_path.parent.name.lower() == "scripts" else python_path.parent.name
                self.stat_python[1].setText(environment_name)
                self.stat_python[2].setText("training interpreter selected")
            else:
                self.stat_python[1].setText("—")
                self.stat_python[2].setText("select the demo's Conda environment")

    @staticmethod
    def _show_error(message: str) -> None:
        QMessageBox.warning(None, "Cannot continue", message)

    @staticmethod
    def _stylesheet() -> str:
        return """
            QMainWindow { background: #f2f3f5; color: #252932; }
            QWidget { color: #252932; }
            QFrame#sidebar { background: #20232b; color: #f7f7f8; }
            QScrollArea { background: transparent; border: 0; }
            QLabel#brandName { color: #ffffff; font-size: 17px; font-weight: 800; letter-spacing: 1px; }
            QLabel#brandEvent { color: #ed3446; font-size: 9px; font-weight: 800; letter-spacing: 1px; }
            QLabel#sidebarCaption { color: #aeb3bf; font-size: 10px; font-weight: 700; margin-top: 22px; }
            QLabel#sidebarGroup { color: #777f8d; font-size: 9px; font-weight: 800; letter-spacing: 1.2px; margin-top: 9px; margin-left: 5px; }
            QFrame#sidebarDivider { color: #3b3f49; margin: 8px 0 12px 0; }
            QPushButton#navButton { color: #c8cbd2; background: transparent; border: 0; border-radius: 7px;
                text-align: left; padding: 12px 12px; font-weight: 600; }
            QPushButton#navButton:hover { color: #ffffff; background: #2b2f39; }
            QPushButton#navButton:checked { color: #ffffff; background: #353944; border-left: 3px solid #d9192e; }
            QLabel#sidebarFootnote { color: #969ba7; font-size: 10px; line-height: 1.5; }
            QLabel#eyebrow { color: #bd1e31; font-size: 10px; font-weight: 800; letter-spacing: 1.4px; }
            QLabel#pageTitle { color: #1d222b; font-size: 28px; font-weight: 750; }
            QLabel#pageSubtitle { color: #6d7380; font-size: 13px; }
            QFrame#hero { background: #242832; border-radius: 14px; }
            QLabel#heroEyebrow { color: #f05a67; font-size: 10px; font-weight: 800; letter-spacing: 1.4px; }
            QLabel#heroTitle { color: #ffffff; font-size: 27px; font-weight: 750; }
            QLabel#heroDetail { color: #d1d4da; font-size: 13px; }
            QFrame#card { background: #ffffff; border: 1px solid #e0e2e7; border-radius: 11px; }
            QFrame#collapsiblePanel { background: #ffffff; border: 1px solid #e0e2e7; border-radius: 9px; }
            QToolButton#chartDropdown { color: #363b45; border: 0; text-align: left; padding: 5px; font-weight: 700; }
            QToolButton#chartDropdown:hover { color: #a91a2c; }
            QLabel#cardTitle { color: #363b45; font-size: 13px; font-weight: 700; }
            QLabel#mutedText { color: #717784; font-size: 12px; }
            QLabel#statValue { color: #20242c; font-size: 23px; font-weight: 750; }
            QFrame#pipelineStep { background: #f8f8fa; border: 1px solid #e9eaee; border-radius: 8px; }
            QLabel#stepNumber { color: #cf1f32; font-size: 20px; font-weight: 800; }
            QLabel#stepTitle { color: #292e37; font-size: 13px; font-weight: 700; }
            QLabel#codeLabel { color: #9e1b2b; background: #fff2f3; border-radius: 5px; padding: 6px; font-family: Consolas; }
            QLabel#consoleStatus { color: #505662; font-weight: 700; }
            QLabel#metricValue { color: #173f35; font-size: 20px; font-weight: 700; }
            QLineEdit, QSpinBox, QComboBox, QListWidget, QPlainTextEdit {
                background: #ffffff; border: 1px solid #d9dce2; border-radius: 6px; padding: 8px;
                selection-background-color: #d92337; }
            QPlainTextEdit { font-family: Consolas, monospace; }
            QListWidget::item { padding: 9px; border-radius: 5px; }
            QListWidget::item:selected { background: #fbe8eb; color: #9e1b2b; }
            QPushButton { background: #ffffff; border: 1px solid #d7dae0; border-radius: 6px;
                padding: 9px 13px; font-weight: 600; }
            QPushButton:hover { background: #f7e9eb; border-color: #d48a92; }
            QPushButton:disabled { color: #a4a8b0; background: #f1f2f4; }
            QPushButton#primaryButton { color: #ffffff; background: #c91d30; border-color: #c91d30; }
            QPushButton#primaryButton:hover { background: #ad1829; }
            QProgressBar { border: 0; background: #e4e5e9; border-radius: 4px; height: 8px; }
            QProgressBar::chunk { background: #d51f34; border-radius: 4px; }
            QStatusBar { background: #e8e9ed; color: #5c626d; }
            QCheckBox { spacing: 8px; }
        """


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName("Rubinnova Robotekin Training Studio")
    app.setFont(QFont("Segoe UI", 10))
    window = TrainingWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
