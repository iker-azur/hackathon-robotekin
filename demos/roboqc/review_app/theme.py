"""Light theme tokens, size constants and stylesheet.

Every page uses the same constants so panels line up: an 8 px grid, 20 px page margins,
16 px card padding, 32 px controls, a fixed 440 px control column and 380 px side panel.
"""

PALETTE = {
    'bg': '#F5F7FA', 'surface': '#FFFFFF', 'surface_alt': '#EEF2F7',
    'border': '#D8DEE6', 'border_strong': '#B8C2CF',
    'text': '#1F2933', 'muted': '#5B6776', 'on_primary': '#FFFFFF',
    'primary': '#2563EB', 'primary_hover': '#1D4ED8', 'primary_soft': '#E8F0FE',
    'success': '#166534', 'success_soft': '#E7F6EC',
    'warn': '#B45309', 'warn_soft': '#FEF3E2',
    'danger': '#B91C1C', 'danger_soft': '#FDECEC',
    'canvas': '#E6EAF0', 'tooltip': '#1F2933',
}

# Distinct on a light background; a class keeps its colour for the whole project.
CLASS_COLORS = ['#2563EB', '#E11D48', '#059669', '#D97706', '#7C3AED',
                '#0891B2', '#DB2777', '#65A30D', '#EA580C', '#4F46E5']

GRID = 8
MARGIN = 20
CARD_PAD = 16
GAP = 8
CONTROL_H = 32
BUTTON_W = 96
LABEL_W = 170
SIDE_W = 400
CONTROL_COL_W = 440
MIN_WINDOW = (1280, 720)
DEFAULT_WINDOW = (1440, 900)


def hex_to_rgb(value):
    value = value.lstrip('#')
    return tuple(int(value[i:i + 2], 16) for i in (0, 2, 4))


def luminance(color):
    channels = []
    for value in hex_to_rgb(color):
        value /= 255
        channels.append(value / 12.92 if value <= .03928 else ((value + .055) / 1.055) ** 2.4)
    return .2126 * channels[0] + .7152 * channels[1] + .0722 * channels[2]


def contrast(foreground, background):
    """WCAG contrast ratio between two '#rrggbb' colours."""
    a, b = sorted((luminance(foreground), luminance(background)), reverse=True)
    return (a + .05) / (b + .05)


STYLE = """
QToolTip { background: @tooltip; color: #FFFFFF; border: none; padding: 6px 8px; }
QMainWindow, QDialog { background: @bg; }
QWidget#page, QWidget#pageBody { background: @bg; }
QScrollArea { border: none; background: transparent; }
QScrollArea > QWidget > QWidget { background: transparent; }

QTabWidget::pane { border: none; border-top: 1px solid @border; background: @bg; }
QTabBar { background: @surface; }
QTabBar::tab { padding: 11px 22px; min-width: 96px; background: transparent; color: @muted;
               border: none; border-bottom: 3px solid transparent; font-weight: 600; }
QTabBar::tab:selected { color: @primary; border-bottom-color: @primary; }
QTabBar::tab:hover:!selected { color: @text; background: @surface_alt; }

QFrame#card { background: @surface; border: 1px solid @border; border-radius: 8px; }
QLabel#cardTitle { font-weight: 700; font-size: 10.5pt; background: transparent; }
QLabel#pageTitle { font-size: 16pt; font-weight: 700; background: transparent; }
QLabel#pageSubtitle { color: @muted; background: transparent; }
QLabel[muted="true"] { color: @muted; background: transparent; }
QLabel { background: transparent; }

QLabel[chip="true"] { border-radius: 10px; padding: 2px 10px; font-weight: 600; font-size: 9pt;
                      background: @surface_alt; color: @muted; }
QLabel[chip="true"][kind="ok"] { background: @success_soft; color: @success; }
QLabel[chip="true"][kind="warn"] { background: @warn_soft; color: @warn; }
QLabel[chip="true"][kind="danger"] { background: @danger_soft; color: @danger; }
QLabel[chip="true"][kind="info"] { background: @primary_soft; color: @primary; }

QLabel#keycap { background: @surface_alt; border: 1px solid @border_strong; border-radius: 4px;
                padding: 1px 5px; color: @muted; font-family: Consolas, monospace; font-size: 8.5pt; }
QLabel[chip="true"][wrap="true"] { border-radius: 6px; padding: 6px 10px; font-weight: 500; }

QPushButton { min-height: 30px; padding: 0px 14px; border: 1px solid @border_strong;
              border-radius: 6px; background: @surface; color: @text; }
QPushButton:hover { background: @surface_alt; }
QPushButton:pressed { background: @border; }
QPushButton:disabled { color: @border_strong; background: @bg; border-color: @border; }
QPushButton:checked { background: @primary_soft; border-color: @primary; color: @primary; }
QPushButton:focus { border-color: @primary; }
QPushButton[role="primary"] { background: @primary; border-color: @primary; color: @on_primary; font-weight: 600; }
QPushButton[role="primary"]:hover { background: @primary_hover; border-color: @primary_hover; }
QPushButton[role="primary"]:disabled { background: @border; border-color: @border; color: @surface; }
QPushButton[role="danger"] { color: @danger; border-color: @danger; }
QPushButton[role="danger"]:hover { background: @danger_soft; }
QPushButton[role="flat"] { border-color: transparent; background: transparent; color: @muted; }
QPushButton[role="flat"]:hover { background: @surface_alt; color: @text; }
QPushButton::menu-indicator { image: none; width: 0px; }

QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox { min-height: 30px; padding: 0px 8px;
    border: 1px solid @border_strong; border-radius: 6px; background: @surface;
    selection-background-color: @primary; selection-color: #FFFFFF; }
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus { border-color: @primary; }
QLineEdit:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled, QComboBox:disabled {
    background: @bg; color: @border_strong; }
QComboBox QAbstractItemView { background: @surface; border: 1px solid @border_strong;
    selection-background-color: @primary_soft; selection-color: @text; outline: none; }
QComboBox { padding-right: 28px; }
QComboBox::drop-down { subcontrol-origin: padding; subcontrol-position: center right; width: 26px; border: none; }
QComboBox::down-arrow { image: @icon_down; width: 10px; height: 6px; }
QSpinBox, QDoubleSpinBox { padding-right: 26px; }
QSpinBox::up-button, QDoubleSpinBox::up-button { subcontrol-origin: border; subcontrol-position: top right;
    width: 22px; border-left: 1px solid @border; border-top-right-radius: 5px; }
QSpinBox::down-button, QDoubleSpinBox::down-button { subcontrol-origin: border; subcontrol-position: bottom right;
    width: 22px; border-left: 1px solid @border; border-bottom-right-radius: 5px; }
QSpinBox::up-button:hover, QSpinBox::down-button:hover, QDoubleSpinBox::up-button:hover,
QDoubleSpinBox::down-button:hover { background: @surface_alt; }
QSpinBox::up-arrow, QDoubleSpinBox::up-arrow { image: @icon_up; width: 10px; height: 6px; }
QSpinBox::down-arrow, QDoubleSpinBox::down-arrow { image: @icon_down; width: 10px; height: 6px; }
QCheckBox::indicator, QListView::indicator { width: 16px; height: 16px; border: 1px solid @border_strong;
    border-radius: 4px; background: @surface; }
QCheckBox::indicator:hover, QListView::indicator:hover { border-color: @primary; }
QCheckBox::indicator:checked, QListView::indicator:checked { background: @primary; border-color: @primary;
    image: @icon_check; }
QCheckBox::indicator:disabled { background: @bg; border-color: @border; }

QPlainTextEdit, QListWidget, QTableWidget { background: @surface; border: 1px solid @border;
    border-radius: 6px; selection-background-color: @primary_soft; selection-color: @text; }
QListWidget { outline: none; padding: 2px; }
QListWidget::item { padding: 5px 6px; border-radius: 4px; }
QListWidget::item:hover { background: @surface_alt; }
QListWidget::item:selected { background: @primary_soft; color: @text; }
QTableWidget { gridline-color: transparent; alternate-background-color: #FAFBFD; outline: none; }
QTableWidget::item { padding: 4px 8px; }
QHeaderView::section { background: @surface_alt; color: @muted; border: none;
    border-bottom: 1px solid @border; padding: 8px 10px; font-weight: 600; }

QCheckBox, QRadioButton { spacing: 8px; background: transparent; }
QSlider::groove:horizontal { height: 4px; background: @border; border-radius: 2px; }
QSlider::sub-page:horizontal { background: @primary; border-radius: 2px; }
QSlider::handle:horizontal { width: 14px; height: 14px; margin: -6px 0; border-radius: 7px;
    background: @surface; border: 2px solid @primary; }

QProgressBar { border: 1px solid @border; border-radius: 6px; background: @surface_alt;
    text-align: center; min-height: 16px; max-height: 16px; color: @text; }
QProgressBar::chunk { background: @primary; border-radius: 5px; }

QScrollBar:vertical { width: 10px; background: transparent; margin: 2px; }
QScrollBar::handle:vertical { background: @border_strong; border-radius: 4px; min-height: 28px; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0px; }
QScrollBar:horizontal { height: 10px; background: transparent; margin: 2px; }
QScrollBar::handle:horizontal { background: @border_strong; border-radius: 4px; min-width: 28px; }
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal { width: 0px; }

QStatusBar { background: @surface; border-top: 1px solid @border; color: @muted; }
QStatusBar::item { border: none; }
QMenuBar { background: @surface; border-bottom: 1px solid @border; }
QMenuBar::item { padding: 6px 12px; background: transparent; }
QMenuBar::item:selected { background: @primary_soft; border-radius: 4px; }
QMenu { background: @surface; border: 1px solid @border_strong; padding: 4px; }
QMenu::item { padding: 6px 28px 6px 20px; border-radius: 4px; }
QMenu::item:selected { background: @primary_soft; color: @text; }
QMenu::item:disabled { color: @border_strong; }
QMenu::separator { height: 1px; background: @border; margin: 4px 8px; }
QFrame#canvasFrame { background: @canvas; border: 1px solid @border; border-radius: 8px; }
QGraphicsView { background: @canvas; border: none; border-radius: 8px; }
"""


def make_icons(directory=None):
    """Write the small check / arrow PNGs the stylesheet needs; returns {name: 'url(path)'}."""
    import tempfile
    from pathlib import Path
    from PySide6.QtCore import QPointF, Qt
    from PySide6.QtGui import QColor, QImage, QPainter, QPen, QPolygonF
    folder = Path(directory or Path(tempfile.gettempdir()) / 'robotekin_review_icons')
    folder.mkdir(parents=True, exist_ok=True)
    icons = {}
    def draw(name, size, painter_fn):
        image = QImage(size[0], size[1], QImage.Format.Format_ARGB32)
        image.fill(Qt.GlobalColor.transparent)
        painter = QPainter(image)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter_fn(painter)
        painter.end()
        path = folder / f'{name}.png'
        image.save(str(path))
        icons[name] = f'url({path.as_posix()})'
    def check(p):
        p.setPen(QPen(QColor('#FFFFFF'), 2.2, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
        p.drawPolyline(QPolygonF([QPointF(3.5, 8.5), QPointF(6.5, 11.5), QPointF(12.5, 4.8)]))
    def arrow(up):
        def paint(p):
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QColor(PALETTE['muted']))
            tip, base = (0.5, 5.5) if up else (5.5, 0.5)
            p.drawPolygon(QPolygonF([QPointF(1, base), QPointF(9, base), QPointF(5, tip)]))
        return paint
    draw('check', (16, 16), check)
    draw('up', (10, 6), arrow(True))
    draw('down', (10, 6), arrow(False))
    return icons


def stylesheet(icons=None):
    qss = STYLE
    tokens = {**PALETTE, **{f'icon_{name}': value for name, value in (icons or {}).items()}}
    for key in sorted(tokens, key=len, reverse=True):
        qss = qss.replace('@' + key, tokens[key])
    for name in ('check', 'up', 'down'):
        qss = qss.replace(f'@icon_{name}', 'none')
    return qss


def apply_theme(app):
    from PySide6.QtGui import QColor, QFont, QPalette
    app.setStyle('Fusion')
    palette = QPalette()
    for role, key in [(QPalette.ColorRole.Window, 'bg'), (QPalette.ColorRole.Base, 'surface'),
                      (QPalette.ColorRole.AlternateBase, 'surface_alt'), (QPalette.ColorRole.Button, 'surface'),
                      (QPalette.ColorRole.WindowText, 'text'), (QPalette.ColorRole.Text, 'text'),
                      (QPalette.ColorRole.ButtonText, 'text'), (QPalette.ColorRole.Highlight, 'primary'),
                      (QPalette.ColorRole.HighlightedText, 'on_primary'),
                      (QPalette.ColorRole.ToolTipBase, 'tooltip'), (QPalette.ColorRole.ToolTipText, 'on_primary'),
                      (QPalette.ColorRole.PlaceholderText, 'muted')]:
        palette.setColor(role, QColor(PALETTE[key]))
    app.setPalette(palette)
    app.setFont(QFont('Segoe UI', 10))
    app.setStyleSheet(stylesheet(make_icons()))
