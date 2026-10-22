"""One table of keyboard shortcuts.

The same table builds the QActions (so menus and keys never disagree), the key hints shown
on buttons and the F1 dialog. Plain single keys (letters, digits, arrows, Tab, Del) are
switched off while the user types in a text field, so they never steal characters.
"""

from dataclasses import dataclass

from PySide6.QtCore import QObject, Qt, Slot
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (
    QAbstractSpinBox, QApplication, QComboBox, QLineEdit, QPlainTextEdit, QTextEdit, QWidget,
)


@dataclass(frozen=True)
class Shortcut:
    id: str
    keys: tuple
    label: str
    group: str
    scope: str = 'global'     # 'review' shortcuts only fire on the Review page
    bound: bool = True        # False: handled by an event filter (Space = hold to pan)
    repeat: bool = False      # allow auto-repeat while the key is held


def _table():
    rows = [
        # Navigation
        Shortcut('prev_image', ('Left',), 'Previous image', 'Navigation', 'review'),
        Shortcut('next_image', ('Right',), 'Next image', 'Navigation', 'review'),
        Shortcut('next_unreviewed', ('N',), 'Next unreviewed image', 'Navigation', 'review'),
        Shortcut('prev_method', ('Up',), 'Previous method / prompt', 'Navigation', 'review'),
        Shortcut('next_method', ('Down',), 'Next method / prompt', 'Navigation', 'review'),
        Shortcut('merge_mode', ('Shift+M',), 'Toggle "add masks from other methods"', 'Navigation', 'review'),
    ]
    pages = ['Explorer', 'Review', 'Augmentation', 'Training', 'Validation']
    rows += [Shortcut(f'page_{i}', (f'Ctrl+{i}',), f'Go to {name}', 'Navigation') for i, name in enumerate(pages, 1)]
    rows += [
        # Decision
        Shortcut('accept_next', ('Ctrl+Return', 'Ctrl+Enter'), 'Accept and go to next unreviewed', 'Decision', 'review'),
        Shortcut('accept', ('Ctrl+S',), 'Accept (save revision)', 'Decision', 'review'),
        Shortcut('exclude', ('Ctrl+E',), 'Exclude image from training', 'Decision', 'review'),
        Shortcut('undo', ('Ctrl+Z',), 'Undo', 'Decision', 'review', repeat=True),
        Shortcut('redo', ('Ctrl+Y', 'Ctrl+Shift+Z'), 'Redo', 'Decision', 'review', repeat=True),
        # Tools
        Shortcut('tool_select', ('V',), 'Select tool (click a mask)', 'Tools', 'review'),
        Shortcut('tool_pan', ('P',), 'Pan tool', 'Tools', 'review'),
        Shortcut('tool_brush', ('B',), 'Brush: paint into the active mask', 'Tools', 'review'),
        Shortcut('tool_erase', ('E',), 'Eraser', 'Tools', 'review'),
        Shortcut('tool_lasso', ('L',), 'Lasso polygon (Enter closes, Shift+Enter subtracts)', 'Tools', 'review'),
        Shortcut('pan_hold', ('Space',), 'Hold to pan with any tool', 'Tools', 'review', bound=False),
        Shortcut('brush_smaller', ('[',), 'Smaller brush', 'Tools', 'review', repeat=True),
        Shortcut('brush_larger', (']',), 'Larger brush', 'Tools', 'review', repeat=True),
        Shortcut('toggle_masks', ('H',), 'Show / hide masks', 'Tools', 'review'),
        Shortcut('cycle_opacity', ('O',), 'Cycle mask opacity', 'Tools', 'review'),
        # View
        Shortcut('zoom_fit', ('F',), 'Fit image to window', 'View', 'review'),
        Shortcut('zoom_100', ('0',), 'Zoom to 100 %', 'View', 'review'),
        Shortcut('zoom_in', ('+', '='), 'Zoom in', 'View', 'review', repeat=True),
        Shortcut('zoom_out', ('-',), 'Zoom out', 'View', 'review', repeat=True),
        # Instances
        Shortcut('next_instance', ('Tab',), 'Next mask', 'Masks', 'review'),
        Shortcut('prev_instance', ('Shift+Tab', 'Shift+Backtab'), 'Previous mask', 'Masks', 'review'),
        Shortcut('toggle_keep', ('K',), 'Keep / drop the selected mask', 'Masks', 'review'),
        Shortcut('keep_only', ('Shift+K',), 'Keep only the selected mask', 'Masks', 'review'),
        Shortcut('delete_instance', ('Del',), 'Delete selected mask(s)', 'Masks', 'review'),
        Shortcut('new_instance', ('Ctrl+N',), 'New empty mask (then paint it)', 'Masks', 'review'),
        Shortcut('duplicate_instance', ('Ctrl+D',), 'Duplicate selected mask', 'Masks', 'review'),
        Shortcut('merge_instances', ('M',), 'Merge selected masks into one', 'Masks', 'review'),
        Shortcut('split_instance', ('S',), 'Split mask into connected pieces', 'Masks', 'review'),
        Shortcut('check_all', ('A',), 'Keep all masks', 'Masks', 'review'),
        Shortcut('check_none', ('Shift+A',), 'Drop all masks', 'Masks', 'review'),
        Shortcut('check_invert', ('I',), 'Invert kept / dropped', 'Masks', 'review'),
    ]
    rows += [Shortcut(f'class_{i}', (str(i),), f'Assign class {i} to selected mask(s)', 'Classes', 'review')
             for i in range(1, 10)]
    rows += [
        Shortcut('manage_classes', ('Ctrl+Shift+C',), 'Manage classes', 'Classes', 'review'),
        Shortcut('reroll_preview', ('R',), 'Re-roll the augmentation preview', 'Augmentation', 'augment'),
        # Files / help
        Shortcut('open_zip', ('Ctrl+O',), 'Open 04e results ZIP', 'File'),
        Shortcut('open_folder', ('Ctrl+Shift+O',), 'Open extracted bundle folder', 'File'),
        Shortcut('export_dataset', ('Ctrl+Shift+E',), 'Export approved dataset', 'File'),
        Shortcut('show_shortcuts', ('F1', '?'), 'Show keyboard shortcuts', 'Help'),
    ]
    return {row.id: row for row in rows}


SHORTCUTS = _table()
GROUP_ORDER = ['Navigation', 'Decision', 'Tools', 'View', 'Masks', 'Classes', 'Augmentation', 'File', 'Help']

_PRETTY = {'Return': 'Enter', 'Left': '←', 'Right': '→', 'Up': '↑', 'Down': '↓',
           'Backtab': 'Tab', 'Delete': 'Del', 'Escape': 'Esc'}


def pretty(sequence):
    """'Ctrl+Return' -> 'Ctrl+↵' for key caps and tooltips."""
    if sequence == '+':
        return '+'
    tokens = sequence[:-2].split('+') + ['+'] if sequence.endswith('++') else sequence.split('+')
    return '+'.join(_PRETTY.get(token, token) for token in tokens)


def hint(shortcut_id):
    """Key cap text for a shortcut, or '' if the id has no key."""
    shortcut = SHORTCUTS.get(shortcut_id)
    return pretty(shortcut.keys[0]) if shortcut else ''


def modifier_free(key):
    """True for keys that a text field also uses (letters, digits, arrows, Tab, Del…)."""
    return not any(modifier in key for modifier in ('Ctrl', 'Alt', 'Meta')) and not (
        key.startswith('F') and key[1:].isdigit())


def is_text_input(widget):
    if isinstance(widget, QComboBox):
        return widget.isEditable()
    return isinstance(widget, (QLineEdit, QPlainTextEdit, QTextEdit, QAbstractSpinBox))


class ShortcutManager(QObject):
    """Owns one QAction per shortcut id and enables it only where it makes sense."""

    def __init__(self, window):
        super().__init__(window)
        self.window = window
        self.scope = 'other'
        self.busy = False
        self.typing = False
        self.actions = {}
        for shortcut in SHORTCUTS.values():
            if not shortcut.bound:
                continue
            action = QAction(shortcut.label, window)
            action.setShortcuts([QKeySequence(key) for key in shortcut.keys])
            action.setShortcutContext(Qt.ShortcutContext.WindowShortcut)
            action.setAutoRepeat(shortcut.repeat)
            window.addAction(action)
            self.actions[shortcut.id] = action
        application = QApplication.instance()
        if application is not None:
            application.focusChanged.connect(self.on_focus_changed)
        self.refresh()

    @Slot(QWidget, QWidget)
    def on_focus_changed(self, old, now):
        self.refresh(now)

    def bind(self, shortcut_id, callback):
        self.actions[shortcut_id].triggered.connect(lambda checked=False: callback())

    def allowed(self, shortcut, typing=None, scope=None, busy=None):
        typing = self.typing if typing is None else typing
        scope = self.scope if scope is None else scope
        busy = self.busy if busy is None else busy
        if busy or (shortcut.scope != 'global' and shortcut.scope != scope):
            return False
        return not (typing and all(modifier_free(key) for key in shortcut.keys))

    def set_scope(self, scope):
        self.scope = scope
        self.refresh()

    def set_busy(self, busy):
        self.busy = busy
        self.refresh()

    def refresh(self, focus=None):
        self.typing = is_text_input(focus if focus is not None else QApplication.focusWidget())
        for shortcut_id, action in self.actions.items():
            action.setEnabled(self.allowed(SHORTCUTS[shortcut_id]))
