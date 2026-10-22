"""Dialogs: shortcuts, class manager, auto-group, export options, unsaved changes."""

from collections import Counter
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QColorDialog, QDialog, QDialogButtonBox, QFileDialog, QHBoxLayout, QInputDialog,
    QLabel, QLineEdit, QMessageBox, QPushButton, QScrollArea, QSpinBox, QTableWidget, QTableWidgetItem,
    QVBoxLayout, QWidget, QHeaderView, QAbstractItemView,
)

from .shortcuts import GROUP_ORDER, SHORTCUTS, Shortcut, pretty
from .suggest import auto_groups
from .theme import GAP, MARGIN
from .widgets import Card, FormGrid, Keycap, swatch_icon


class ShortcutsDialog(QDialog):
    """Read-only, searchable reference built from the same table that binds the keys."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Keyboard shortcuts')
        self.resize(760, 640)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(MARGIN, MARGIN, MARGIN, MARGIN)
        layout.setSpacing(12)
        intro = QLabel('Single keys work while the image is focused; they pause when you type in a text field. '
                       'Hover any button to see its key.')
        intro.setProperty('muted', True)
        intro.setWordWrap(True)
        layout.addWidget(intro)
        self.search = QLineEdit()
        self.search.setPlaceholderText('Filter shortcuts…')
        self.search.setClearButtonEnabled(True)
        self.search.textChanged.connect(self.filter)
        layout.addWidget(self.search)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        inner = QWidget()
        inner.setObjectName('pageBody')
        column = QVBoxLayout(inner)
        column.setContentsMargins(0, 0, 8, 0)
        column.setSpacing(12)
        self.rows = []
        for group in GROUP_ORDER:
            entries = [s for s in SHORTCUTS.values() if s.group == group]
            if not entries:
                continue
            card = Card(group)
            for shortcut in entries:
                row = self._row(shortcut)
                card.add(row)
                self.rows.append((shortcut, row))
            column.addWidget(card)
        column.addStretch(1)
        scroll.setWidget(inner)
        layout.addWidget(scroll, 1)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    @staticmethod
    def _row(shortcut: Shortcut):
        row = QWidget()
        line = QHBoxLayout(row)
        line.setContentsMargins(0, 0, 0, 0)
        line.setSpacing(GAP)
        line.addWidget(QLabel(shortcut.label), 1)
        for text in dict.fromkeys(pretty(key) for key in shortcut.keys):      # 'Ctrl+Return' and 'Ctrl+Enter' read the same
            line.addWidget(Keycap(text))
        return row

    def filter(self, text):
        text = text.casefold()
        for shortcut, row in self.rows:
            haystack = ' '.join((shortcut.label, shortcut.group, *shortcut.keys)).casefold()
            row.setVisible(text in haystack)


class ClassDialog(QDialog):
    """Add, rename, recolour, reorder, merge and delete classes. Keys 1-9 follow the row order."""

    def __init__(self, project, parent=None):
        super().__init__(parent)
        self.project = project
        self.changed = False
        self.setWindowTitle('Manage classes')
        self.resize(640, 460)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(MARGIN, MARGIN, MARGIN, MARGIN)
        layout.setSpacing(12)
        note = QLabel('Each class keeps its colour everywhere. The first nine get the keys 1-9. '
                      'Renaming or merging re-saves affected images as new revisions; originals are kept.')
        note.setProperty('muted', True)
        note.setWordWrap(True)
        layout.addWidget(note)
        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(['Key', 'Class', 'Accepted masks', 'Colour'])
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.verticalHeader().hide()
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.table.doubleClicked.connect(lambda *_: self.rename())
        layout.addWidget(self.table, 1)
        row = QHBoxLayout()
        row.setSpacing(GAP)
        for text, callback in [('Add…', self.add), ('Rename…', self.rename), ('Colour…', self.recolor),
                               ('Up', lambda: self.move(-1)), ('Down', lambda: self.move(1)),
                               ('Merge into…', self.merge), ('Delete', self.delete)]:
            button = QPushButton(text)
            button.clicked.connect(callback)
            row.addWidget(button)
        layout.addLayout(row)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self.reload()

    def reload(self, select=None):
        usage = self.project.class_usage()
        rows = self.project.classes()
        self.table.setRowCount(len(rows))
        for index, info in enumerate(rows):
            values = [str(index + 1) if index < 9 else '', info['name'], str(usage.get(info['name'], 0)), info['color']]
            for column, value in enumerate(values):
                item = QTableWidgetItem(value)
                if column == 3:
                    item.setIcon(swatch_icon(info['color'], 16))
                self.table.setItem(index, column, item)
        if select in self.project.class_names():
            self.table.selectRow(self.project.class_names().index(select))

    def current(self):
        row = self.table.currentRow()
        names = self.project.class_names()
        return names[row] if 0 <= row < len(names) else None

    def add(self):
        name, ok = QInputDialog.getText(self, 'Add class', 'Class name:')
        name = name.strip()
        if ok and name:
            self.project.ensure_class(name)
            self.changed = True
            self.reload(name)

    def rename(self):
        old = self.current()
        if old is None:
            return
        new, ok = QInputDialog.getText(self, 'Rename class', f'New name for "{old}":', text=old)
        new = new.strip()
        if not ok or not new or new == old:
            return
        self._run(lambda: self.project.rename_class(old, new), new)

    def merge(self):
        old = self.current()
        others = [n for n in self.project.class_names() if n != old]
        if old is None or not others:
            return
        target, ok = QInputDialog.getItem(self, 'Merge class', f'Merge "{old}" into:', others, 0, False)
        if ok:
            self._run(lambda: self.project.rename_class(old, target), target)

    def _run(self, action, select):
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            action()
        except Exception as exc:
            QMessageBox.warning(self, 'Manage classes', str(exc))
        finally:
            QApplication.restoreOverrideCursor()
        self.changed = True
        self.reload(select)

    def recolor(self):
        name = self.current()
        if name is None:
            return
        color = QColorDialog.getColor(QColor(self.project.color_of(name)), self, f'Colour for {name}')
        if color.isValid():
            self.project.set_class_color(name, color.name())
            self.changed = True
            self.reload(name)

    def move(self, offset):
        name = self.current()
        if name is not None:
            self.project.move_class(name, offset)
            self.changed = True
            self.reload(name)

    def delete(self):
        name = self.current()
        if name is None:
            return
        try:
            self.project.delete_class(name)
        except ValueError as exc:
            QMessageBox.warning(self, 'Manage classes', str(exc))
            return
        self.changed = True
        self.reload()


class AutoGroupDialog(QDialog):
    """Choose how unreviewed images get their default capture group."""

    def __init__(self, images, window_seconds, parent=None):
        super().__init__(parent)
        self.images = images
        self.setWindowTitle('Capture groups')
        self.setMinimumWidth(520)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(MARGIN, MARGIN, MARGIN, MARGIN)
        layout.setSpacing(12)
        text = QLabel('Frames of one burst are near-identical, so they must share a group or the same scene '
                      'lands in both training and validation. By default each folder is one group (safest). '
                      'Time windows give more, smaller groups for a finer split, with some leakage risk.')
        text.setWordWrap(True)
        layout.addWidget(text)
        form = FormGrid()
        self.seconds = QSpinBox()
        self.seconds.setRange(0, 3600)
        self.seconds.setSuffix(' s')
        self.seconds.setSpecialValueText('Folder (no splitting)')
        self.seconds.setValue(window_seconds)
        self.seconds.valueChanged.connect(self.update_preview)
        form.add_row('Maximum block length', self.seconds, 'Frames of one folder are cut into blocks of at most this many seconds.')
        layout.addLayout(form)
        self.preview = QLabel()
        self.preview.setWordWrap(True)
        layout.addWidget(self.preview)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self.update_preview()

    def update_preview(self):
        groups = auto_groups(self.images, self.seconds.value())
        counts = Counter(groups.values())
        biggest = max(counts.values()) if counts else 0
        self.preview.setText(f'{len(counts)} groups over {len(self.images)} images (largest: {biggest} images). '
                             + ('Applies to images you have not accepted yet.' if self.seconds.value() == 0 else
                                'Applies to images you have not accepted yet. Neighbouring blocks of one burst still look alike.'))


class ExportDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Export approved dataset')
        self.setMinimumWidth(560)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(MARGIN, MARGIN, MARGIN, MARGIN)
        layout.setSpacing(12)
        note = QLabel('Only accepted images and checked masks are copied into a fresh folder '
                      '(images, YOLO labels, exact reviewed masks, manifest).')
        note.setWordWrap(True)
        layout.addWidget(note)
        form = FormGrid()
        row = QHBoxLayout()
        self.parent_folder = QLineEdit()
        self.parent_folder.setPlaceholderText('Parent folder (a new subfolder is created)')
        row.addWidget(self.parent_folder, 1)
        browse = QPushButton('Browse…')
        browse.clicked.connect(self.browse)
        row.addWidget(browse)
        form.add_row('Output', row)
        layout.addLayout(form)
        self.zip_check = QCheckBox('Also create a ZIP next to the folder (for Colab)')
        self.zip_check.setChecked(True)
        layout.addWidget(self.zip_check)
        self.holes_check = QCheckBox('Allow hole filling in YOLO labels (lossy; exact PNG masks are kept)')
        layout.addWidget(self.holes_check)
        self.buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        self.buttons.button(QDialogButtonBox.StandardButton.Ok).setText('Export')
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        layout.addWidget(self.buttons)

    def browse(self):
        folder = QFileDialog.getExistingDirectory(self, 'Export parent folder')
        if folder:
            self.parent_folder.setText(folder)

    def accept(self):
        if not self.parent_folder.text().strip() or not Path(self.parent_folder.text()).is_dir():
            QMessageBox.warning(self, 'Export', 'Choose an existing parent folder.')
            return
        super().accept()


def ask_unsaved(parent):
    """'accept' (save then continue), 'discard' or 'cancel'."""
    box = QMessageBox(parent)
    box.setIcon(QMessageBox.Icon.Question)
    box.setWindowTitle('Unsaved mask edits')
    box.setText('This image has unsaved edits.')
    box.setInformativeText('Accept saves them as a new revision. Discard drops them.')
    accept = box.addButton('Accept && continue', QMessageBox.ButtonRole.AcceptRole)
    discard = box.addButton('Discard', QMessageBox.ButtonRole.DestructiveRole)
    box.addButton('Cancel', QMessageBox.ButtonRole.RejectRole)
    box.setDefaultButton(accept)
    box.exec()
    clicked = box.clickedButton()
    return 'accept' if clicked is accept else 'discard' if clicked is discard else 'cancel'
