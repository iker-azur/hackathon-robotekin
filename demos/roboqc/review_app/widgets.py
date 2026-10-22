"""Shared widgets: cards, page header, form grid, buttons that show their key."""

from PySide6.QtCore import QRectF, QSize, Qt, Signal
from PySide6.QtGui import QColor, QFont, QFontMetrics, QIcon, QPainter, QPalette, QPen, QPixmap
from PySide6.QtWidgets import (
    QFrame, QGridLayout, QHBoxLayout, QLabel, QLayout, QPushButton, QScrollArea, QSizePolicy, QStyle,
    QStyleOptionButton, QStylePainter, QVBoxLayout, QWidget,
)

from .shortcuts import SHORTCUTS, hint
from .theme import CARD_PAD, CONTROL_COL_W, CONTROL_H, GAP, LABEL_W, MARGIN, PALETTE


def set_property(widget, name, value):
    """Set a dynamic property and re-evaluate the stylesheet."""
    widget.setProperty(name, value)
    widget.style().unpolish(widget)
    widget.style().polish(widget)


def swatch_icon(color, size=14, ring=False):
    pixmap = QPixmap(size, size)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    painter.setBrush(QColor(color))
    painter.setPen(QColor('#1F2933') if ring else Qt.PenStyle.NoPen)
    painter.drawRoundedRect(1, 1, size - 2, size - 2, 3, 3)
    painter.end()
    return QIcon(pixmap)


class Chip(QLabel):
    """Small status pill; kind is one of ok / warn / danger / info / '' (neutral)."""

    def __init__(self, text='', kind='', wrap=False):
        super().__init__(text)
        self.setProperty('chip', True)
        if wrap:
            self.setProperty('wrap', True)
            self.setWordWrap(True)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter if not wrap else Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        self.setSizePolicy(QSizePolicy.Policy.Maximum if not wrap else QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed
                           if not wrap else QSizePolicy.Policy.Minimum)
        self.set(text, kind)

    def set(self, text, kind=''):
        self.setText(text)
        set_property(self, 'kind', kind)


class Keycap(QLabel):
    def __init__(self, text):
        super().__init__(text)
        self.setObjectName('keycap')
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)


class ElidedLabel(QLabel):
    """One-line label that ends in an ellipsis instead of forcing its parent wider."""

    def __init__(self, text=''):
        super().__init__(text)
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        self.setMinimumWidth(0)

    def paintEvent(self, event):
        painter = QPainter(self)
        text = self.fontMetrics().elidedText(self.text(), Qt.TextElideMode.ElideRight, self.width())
        self.style().drawItemText(painter, self.rect(), int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter),
                                  self.palette(), self.isEnabled(), text, QPalette.ColorRole.WindowText)


def _key_font():
    font = QFont()
    font.setFamilies(['Consolas', 'Cascadia Mono', 'Courier New'])
    font.setPointSizeF(8.0)
    return font


class ActionButton(QPushButton):
    """Button that draws its caption and the key cap from the shortcut table.

    It is drawn here (not with nested labels) so colours follow the button state exactly, and it
    can shrink: the key cap is dropped first, then the caption is elided.
    """

    PAD = 12
    SWATCH = 14

    def __init__(self, text, shortcut_id=None, role=None, checkable=False, color=None, align_left=False, parent=None):
        super().__init__(parent)
        self.caption = text
        self.shortcut_id = shortcut_id
        self.align_left = align_left
        self.color = color
        self.key = hint(shortcut_id) if shortcut_id else ''
        if role:
            self.setProperty('role', role)
        self.setCheckable(checkable)
        self.setMinimumHeight(CONTROL_H)
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        keys = ', '.join(SHORTCUTS[shortcut_id].keys) if shortcut_id in SHORTCUTS else ''
        self.setToolTip(f'{text}  ({keys})' if keys else text)
        self.setAccessibleName(text)

    def set_caption(self, text):
        self.caption = text
        self.updateGeometry()
        self.update()

    def _text_font(self):
        font = QFont(self.font())
        font.setBold(self.property('role') == 'primary' or self.isChecked())
        return font

    def _key_width(self):
        return QFontMetrics(_key_font()).horizontalAdvance(self.key) + 12 if self.key else 0

    def sizeHint(self):
        width = 2 * self.PAD + QFontMetrics(self._text_font()).horizontalAdvance(self.caption)
        if self.color:
            width += self.SWATCH + 8
        if self.key:
            width += 8 + self._key_width()
        return QSize(max(64, width), CONTROL_H)

    def minimumSizeHint(self):
        return QSize(44, CONTROL_H)

    def _colors(self, enabled):
        role = self.property('role')
        if not enabled:
            return PALETTE['border_strong'], PALETTE['border_strong'], PALETTE['border'], PALETTE['bg']
        if role == 'primary':
            return PALETTE['on_primary'], '#FFFFFF', '#9DB8F5', '#4F82EE'
        if role == 'danger':
            return PALETTE['danger'], PALETTE['danger'], '#E5A3A3', PALETTE['danger_soft']
        if self.isChecked():
            return PALETTE['primary'], PALETTE['primary'], PALETTE['primary'], PALETTE['surface']
        return PALETTE['text'], PALETTE['muted'], PALETTE['border_strong'], PALETTE['surface_alt']

    def paintEvent(self, event):
        painter = QStylePainter(self)
        option = QStyleOptionButton()
        self.initStyleOption(option)
        option.text = ''
        option.icon = QIcon()
        painter.drawControl(QStyle.ControlElement.CE_PushButton, option)
        enabled = bool(option.state & QStyle.StateFlag.State_Enabled)
        text_color, key_color, key_border, key_fill = self._colors(enabled)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        font = self._text_font()
        metrics = QFontMetrics(font)
        area = self.rect().adjusted(self.PAD, 0, -self.PAD, 0)
        swatch = self.SWATCH + 8 if self.color else 0
        key_width = self._key_width()
        show_key = bool(self.key) and area.width() >= swatch + metrics.horizontalAdvance(self.caption) + 8 + key_width
        reserved = swatch + (8 + key_width if show_key else 0)
        text = metrics.elidedText(self.caption, Qt.TextElideMode.ElideRight, max(0, area.width() - reserved))
        content = reserved + metrics.horizontalAdvance(text)
        x = area.left() if self.align_left else area.left() + max(0, (area.width() - content) // 2)
        middle = self.height() / 2
        if self.color:
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor(self.color))
            painter.drawRoundedRect(QRectF(x, middle - self.SWATCH / 2, self.SWATCH, self.SWATCH), 3, 3)
            x += swatch
        painter.setFont(font)
        painter.setPen(QColor(text_color))
        painter.drawText(QRectF(x, 0, metrics.horizontalAdvance(text) + 2, self.height()),
                         int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter), text)
        if show_key:
            x += metrics.horizontalAdvance(text) + 8
            box = QRectF(x, middle - 9, key_width, 18)
            painter.setBrush(QColor(key_fill))
            painter.setPen(QPen(QColor(key_border), 1))
            painter.drawRoundedRect(box, 4, 4)
            painter.setFont(_key_font())
            painter.setPen(QColor(key_color))
            painter.drawText(box, int(Qt.AlignmentFlag.AlignCenter), self.key)


class Card(QFrame):
    """White rounded panel with an optional title; add content through `body`."""

    def __init__(self, title=None, subtitle=None):
        super().__init__()
        self.setObjectName('card')
        outer = QVBoxLayout(self)
        outer.setContentsMargins(CARD_PAD, CARD_PAD, CARD_PAD, CARD_PAD)
        outer.setSpacing(12)
        self.title_label = None
        if title:
            self.title_label = QLabel(title)
            self.title_label.setObjectName('cardTitle')
            outer.addWidget(self.title_label)
        if subtitle:
            note = QLabel(subtitle)
            note.setProperty('muted', True)
            note.setWordWrap(True)
            outer.addWidget(note)
        self.body = outer

    def add(self, widget, stretch=0):
        self.body.addWidget(widget, stretch)
        return widget

    def add_layout(self, layout, stretch=0):
        self.body.addLayout(layout, stretch)
        return layout


class FormGrid(QGridLayout):
    """Label column of a fixed width so every form on every page lines up."""

    def __init__(self):
        super().__init__()
        self.setContentsMargins(0, 0, 0, 0)
        self.setHorizontalSpacing(12)
        self.setVerticalSpacing(GAP)
        self.setColumnMinimumWidth(0, LABEL_W)
        self.setColumnStretch(1, 1)

    def add_row(self, text, widget, hint_text=None):
        label = QLabel(text)
        label.setWordWrap(True)
        if hint_text:
            label.setToolTip(hint_text)
            if not isinstance(widget, QLayout):
                widget.setToolTip(hint_text)
        row = self.rowCount()
        self.addWidget(label, row, 0, Qt.AlignmentFlag.AlignVCenter)
        if isinstance(widget, QLayout):
            self.addLayout(widget, row, 1)
        else:
            self.addWidget(widget, row, 1)
        return label

    def add_wide(self, text, widget, hint_text=None):
        """Label on its own line, field below it across the full width (paths, long values)."""
        label = QLabel(text)
        if hint_text:
            label.setToolTip(hint_text)
        row = self.rowCount()
        self.addWidget(label, row, 0, 1, 2)
        if isinstance(widget, QLayout):
            self.addLayout(widget, row + 1, 0, 1, 2)
        else:
            self.addWidget(widget, row + 1, 0, 1, 2)
        return label


class PageHeader(QWidget):
    helpRequested = Signal()

    def __init__(self, title, subtitle=''):
        super().__init__()
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(GAP)
        texts = QVBoxLayout()
        texts.setSpacing(2)
        self.title = QLabel(title)
        self.title.setObjectName('pageTitle')
        self.subtitle = QLabel(subtitle)
        self.subtitle.setObjectName('pageSubtitle')
        self.subtitle.setWordWrap(True)
        texts.addWidget(self.title)
        texts.addWidget(self.subtitle)
        layout.addLayout(texts, 1)
        self.actions = QHBoxLayout()
        self.actions.setSpacing(GAP)
        layout.addLayout(self.actions)
        self.help = QPushButton('?')
        self.help.setFixedSize(CONTROL_H, CONTROL_H)
        self.help.setStyleSheet('QPushButton { padding: 0px; font-weight: 700; }')
        self.help.setToolTip('Keyboard shortcuts  (F1)')
        self.help.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.help.clicked.connect(self.helpRequested)
        layout.addWidget(self.help)

    def add_action(self, widget):
        self.actions.addWidget(widget)
        return widget


class Page(QWidget):
    """Common page frame: margins, header and a body area."""

    def __init__(self, title, subtitle=''):
        super().__init__()
        self.setObjectName('page')
        self.root = QVBoxLayout(self)
        self.root.setContentsMargins(MARGIN, MARGIN, MARGIN, MARGIN)
        self.root.setSpacing(16)
        self.header = PageHeader(title, subtitle)
        self.root.addWidget(self.header)


class ControlColumn(QScrollArea):
    """Fixed-width scrollable column of cards (Augmentation, Training, Validation, Review side panel)."""

    def __init__(self, width=CONTROL_COL_W):
        super().__init__()
        self.setWidgetResizable(True)
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setFixedWidth(width + 18)
        inner = QWidget()
        inner.setObjectName('pageBody')
        self.column = QVBoxLayout(inner)
        self.column.setContentsMargins(0, 0, 10, 0)
        self.column.setSpacing(16)
        self.column.addStretch(1)
        self.setWidget(inner)

    def add(self, widget):
        self.column.insertWidget(self.column.count() - 1, widget)
        return widget
