"""Qt UI tests: offscreen, handmade masks, no model import, inference or training."""

import json
import os
from pathlib import Path
import sys
import threading
import time

import numpy as np
import pytest

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from review_app.core import Project, export_dataset
from test_review_app import approve, bundle  # noqa: F401  (bundle is a fixture)

from PySide6.QtCore import QByteArray, QPoint, QPointF, Qt
from PySide6.QtGui import QFontDatabase, QKeySequence
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QInputDialog, QMessageBox

from review_app import page_review
from review_app.canvas import Canvas
from review_app.document import ReviewDocument
from review_app.shortcuts import GROUP_ORDER, SHORTCUTS, hint, modifier_free, pretty
from review_app.theme import CONTROL_COL_W, CONTROL_H, MARGIN, MIN_WINDOW, SIDE_W
from review_app.widgets import ActionButton, ControlColumn

CTRL = Qt.KeyboardModifier.ControlModifier


@pytest.fixture(scope='session')
def qapp():
    app = QApplication.instance() or QApplication([])
    for name in ('segoeui.ttf', 'consola.ttf'):
        font = Path('C:/Windows/Fonts') / name
        if font.exists():
            QFontDatabase.addApplicationFont(str(font))
    return app


@pytest.fixture
def window(qapp, bundle, tmp_path):
    from review_app.app import Window
    win = Window()
    win.errors = []
    win.error = win.errors.append
    win.infos = []
    win.info = lambda title, text: win.infos.append((title, text))
    win.bundle_loaded((bundle, Project(tmp_path / 'project', bundle)))
    win.show()
    win.activateWindow()
    qapp.processEvents()
    yield win
    win.review.discard_edits()
    win.close()
    qapp.processEvents()


def settle(app, win, timeout=10):
    """Process events until background work and queued follow-ups are done."""
    deadline, quiet = time.monotonic() + timeout, 0
    while time.monotonic() < deadline:
        app.processEvents()
        quiet = quiet + 1 if win.task_thread is None and not win._after_task else 0
        if quiet > 6:
            return
        time.sleep(.003)
    pytest.fail('background work did not finish')


def need_active(app, win):
    app.processEvents()
    if QApplication.activeWindow() is not win:
        pytest.skip('this platform does not activate the offscreen window, so key shortcuts cannot fire')


def open_image(app, win, image_id='image_0'):
    win.open_image(image_id)
    settle(app, win)
    assert win.review.current_id == image_id


def press(win, key, modifier=Qt.KeyboardModifier.NoModifier):
    QTest.keyClick(win, key, modifier)


# -- shortcut table and manager -------------------------------------------------------------

def test_shortcut_table_has_no_conflicts_and_pretty_labels():
    seen = {}
    for shortcut in SHORTCUTS.values():
        assert shortcut.group in GROUP_ORDER, shortcut.id
        for key in shortcut.keys:
            normalized = QKeySequence(key).toString()
            assert normalized, f'{shortcut.id}: {key} is not a valid key'
            for other_scope in seen.get(normalized, set()):
                # the same key may be reused only by two different non-global pages
                assert 'global' not in (other_scope, shortcut.scope) and other_scope != shortcut.scope, (shortcut.id, key)
            seen.setdefault(normalized, set()).add(shortcut.scope)
    assert pretty('Ctrl+Return') == 'Ctrl+Enter' and pretty('Left') == '←' and pretty('+') == '+'
    assert pretty('Ctrl+Shift+A') == 'Ctrl+Shift+A' and hint('missing') == ''
    assert {f'class_{n}' for n in range(1, 10)} <= set(SHORTCUTS) and not SHORTCUTS['pan_hold'].bound


def test_plain_keys_pause_while_typing_and_pages_scope_the_rest(qapp, window):
    manager = window.manager
    brush, accept, class_one = SHORTCUTS['tool_brush'], SHORTCUTS['accept'], SHORTCUTS['class_1']
    assert manager.allowed(brush, typing=False, scope='review', busy=False)
    assert not manager.allowed(brush, typing=True, scope='review', busy=False)
    assert not manager.allowed(class_one, typing=True, scope='review', busy=False)
    assert manager.allowed(accept, typing=True, scope='review', busy=False)          # Ctrl+S still works in a text field
    assert manager.allowed(SHORTCUTS['show_shortcuts'], typing=True, scope='other', busy=False)
    assert not manager.allowed(brush, typing=False, scope='other', busy=False)       # Review keys are off elsewhere
    assert not manager.allowed(accept, typing=False, scope='review', busy=True)      # and everything pauses while busy
    assert modifier_free('Shift+A') and not modifier_free('Ctrl+Z') and not modifier_free('F1')
    window.go(1)
    qapp.processEvents()
    assert manager.actions['tool_brush'].isEnabled()
    window.review.group.setFocus()
    qapp.processEvents()
    assert manager.typing and not manager.actions['tool_brush'].isEnabled() and manager.actions['accept'].isEnabled()
    window.review.canvas.setFocus()
    qapp.processEvents()
    assert not manager.typing and manager.actions['tool_brush'].isEnabled()
    window.go(0)
    assert not manager.actions['tool_brush'].isEnabled() and manager.actions['page_2'].isEnabled()


def test_every_button_shows_the_key_from_the_table(qapp, window):
    buttons = [b for b in window.findChildren(ActionButton) if b.shortcut_id]
    assert len(buttons) == 29          # 3 explorer + 21 review + 1 augmentation + 1 training + 3 decision (accept/exclude/next)
    for button in buttons:
        shortcut = SHORTCUTS[button.shortcut_id]
        assert button.key == pretty(shortcut.keys[0]), button.caption
        assert shortcut.keys[0] in button.toolTip()
    labels = {b.caption for b in buttons}
    assert {'Previous', 'Next', 'Undo', 'Accept', 'Exclude', 'Open ZIP', 'Re-roll'} <= labels
    for shortcut_id, action in window.manager.actions.items():
        assert [s.toString() for s in action.shortcuts()] == [QKeySequence(k).toString() for k in SHORTCUTS[shortcut_id].keys]
    menu_titles = [a.text() for a in window.menuBar().actions()]
    assert menu_titles == ['&File', '&Edit', '&Masks', '&Tools', '&View', '&Go', '&Help']


def test_shortcuts_dialog_lists_every_shortcut_and_filters(qapp, window):
    from review_app.dialogs import ShortcutsDialog
    dialog = ShortcutsDialog(window)
    assert len(dialog.rows) == len(SHORTCUTS)
    dialog.search.setText('undo')
    assert [s.id for s, row in dialog.rows if not row.isHidden()] == ['undo']
    dialog.search.setText('')
    assert all(not row.isHidden() for _, row in dialog.rows)
    dialog.close()


# -- window structure and size standards -----------------------------------------------------

@pytest.mark.parametrize('size', [(1280, 720), (1440, 900)])
def test_pages_share_the_same_dimensions_and_never_overflow(qapp, window, size):
    window.resize(*size)
    for index in range(window.tabs.count()):
        window.go(index)
        qapp.processEvents()
    assert window.tabs.count() == 5
    assert window.minimumSizeHint().width() <= MIN_WINDOW[0] and window.minimumSizeHint().height() <= MIN_WINDOW[1]
    for page in (window.explorer, window.review, window.augment, window.training, window.validation):
        margins = page.root.contentsMargins()
        assert (margins.left(), margins.top(), margins.right(), margins.bottom()) == (MARGIN,) * 4
    columns = [window.augment.column, window.training.column, window.validation.column]
    assert {c.width() for c in columns} == {CONTROL_COL_W + 18}
    side = window.review.findChild(ControlColumn)
    assert side.width() == SIDE_W + 18
    for column in columns + [side]:
        assert column.widget().minimumSizeHint().width() <= column.viewport().width()   # cards never clip sideways
    for button in window.findChildren(ActionButton):
        assert button.sizeHint().height() == CONTROL_H and button.minimumSizeHint().width() <= button.sizeHint().width()


def test_screenshots_of_every_page_are_rendered(qapp, window):
    folder = Path(__file__).resolve().parents[2] / 'review_app/verification'
    folder.mkdir(exist_ok=True)
    window.resize(1440, 900)
    open_image(qapp, window)
    for index, name in enumerate(['explorer', 'review', 'augmentation', 'training', 'validation']):
        window.go(index)
        qapp.processEvents()
        image = window.grab()
        assert image.save(str(folder / f'{name}_fixture.png'))
        picture = image.toImage()
        colors = {picture.pixel(x, y) for x in range(0, picture.width(), 37) for y in range(0, picture.height(), 41)}
        assert len(colors) >= 5                      # not a blank page


# -- background work ----------------------------------------------------------------------------

def test_background_callbacks_run_on_gui_thread(qapp, window):
    """Regression: closures connected to worker-thread signals once ran in the worker and built
    widgets off the GUI thread ('Timers cannot be started from another thread')."""
    seen = []
    window.background(lambda progress: threading.current_thread(),
                      lambda worker: seen.append((worker, threading.current_thread())))
    settle(qapp, window)
    window.background(lambda progress: 1 / 0, lambda result: None,
                      lambda message: seen.append((message, threading.current_thread())))
    settle(qapp, window)
    assert len(seen) == 2
    assert all(handler is threading.main_thread() for _, handler in seen)
    assert seen[0][0] is not threading.main_thread() and 'ZeroDivisionError' in seen[1][0]


def test_busy_window_refuses_second_task_quietly_and_pauses_shortcuts(qapp, window):
    release = threading.Event()
    assert window.background(lambda progress: release.wait(5), lambda result: None)
    assert window.manager.busy and not window.tabs.isEnabled()
    assert window.background(lambda progress: 1, lambda result: None) is False       # no modal error pile-up
    assert not window.errors
    release.set()
    settle(qapp, window)
    assert not window.manager.busy and window.tabs.isEnabled()


# -- review flow with real key events --------------------------------------------------------

def test_review_flow_with_keyboard(qapp, window):
    need_active(qapp, window)
    review = window.review
    open_image(qapp, window)
    assert window.tabs.currentIndex() == 1 and review.methods.currentIndex() == 1    # first usable method auto-loaded
    assert len(review.doc.instances) == 1 and review.doc.instances[0]['label'] == 'part'
    assert not window.dirty and review.position.text() == 'Image 1 of 8'
    press(window, Qt.Key.Key_Right)
    settle(qapp, window)
    assert review.current_id == 'image_1'
    press(window, Qt.Key.Key_Left)
    settle(qapp, window)
    assert review.current_id == 'image_0'
    press(window, Qt.Key.Key_Down)                          # next method: wraps to "Manual / empty"
    settle(qapp, window)
    assert review.methods.currentIndex() == 0 and not review.doc.instances and not window.dirty
    press(window, Qt.Key.Key_Up)
    settle(qapp, window)
    assert review.methods.currentIndex() == 1 and len(review.doc.instances) == 1
    assert window.project.pref('method') == ['sam3_text', None]                     # sticky for the next image
    for key, tool in [(Qt.Key.Key_B, 'brush'), (Qt.Key.Key_E, 'erase'), (Qt.Key.Key_L, 'lasso'),
                      (Qt.Key.Key_P, 'pan'), (Qt.Key.Key_V, 'select')]:
        press(window, key)
        assert review.canvas.tool == tool and review.tool_buttons[tool].isChecked()
    press(window, Qt.Key.Key_Tab)
    assert review.canvas.active is review.doc.instances[0]
    press(window, Qt.Key.Key_K)                             # drop the selected mask
    assert review.doc.instances[0]['keep'] is False and window.dirty and window.windowTitle().startswith('* ')
    press(window, Qt.Key.Key_Z, CTRL)
    assert review.doc.instances[0]['keep'] is True and not window.dirty               # undo returns to the saved state
    press(window, Qt.Key.Key_Delete)
    assert not review.doc.instances
    press(window, Qt.Key.Key_Z, CTRL)
    assert len(review.doc.instances) == 1
    press(window, Qt.Key.Key_S, CTRL)
    settle(qapp, window)
    decision = window.project.decision('image_0')
    assert decision['status'] == 'accepted' and len(decision['instances']) == 1 and decision['group'] == 'session_0/nested'
    assert review.status_chip.text() == 'Accepted' and not window.dirty
    assert 'Accepted 1' in window.stats_label.text()
    assert any(window.explorer.table.item(r, 6).text() == 'Accepted' for r in range(8))
    press(window, Qt.Key.Key_E, CTRL)                       # exclude the same image
    settle(qapp, window)
    assert window.project.decision('image_0')['status'] == 'excluded' and review.status_chip.text() == 'Excluded'
    press(window, Qt.Key.Key_Return, CTRL)                  # accept and go to the next unreviewed image
    settle(qapp, window)
    assert window.project.decision('image_0')['status'] == 'accepted' and review.current_id == 'image_1'
    press(window, Qt.Key.Key_N)
    settle(qapp, window)
    assert review.current_id == 'image_2'


def test_typing_in_the_group_field_never_triggers_shortcuts(qapp, window):
    need_active(qapp, window)
    open_image(qapp, window)
    review = window.review
    review.group.setFocus()
    qapp.processEvents()
    for key in (Qt.Key.Key_B, Qt.Key.Key_E, Qt.Key.Key_1, Qt.Key.Key_K):
        QTest.keyClick(review.group, key)
    assert review.canvas.tool == 'select' and review.group.text().endswith('be1k')
    assert review.meta_dirty and window.dirty                  # edited text counts as unsaved


def test_unsaved_edits_prompt_with_three_choices(qapp, window, monkeypatch):
    need_active(qapp, window)
    open_image(qapp, window)
    review = window.review
    review.doc.set_keep([(review.doc.instances[0], False)])
    answers = iter(['cancel', 'discard'])
    monkeypatch.setattr(page_review, 'ask_unsaved', lambda parent: next(answers))
    press(window, Qt.Key.Key_Right)
    settle(qapp, window)
    assert review.current_id == 'image_0' and window.dirty                            # cancel: nothing happens
    press(window, Qt.Key.Key_Right)
    settle(qapp, window)
    assert review.current_id == 'image_1' and not window.dirty                         # discard: edits dropped
    assert window.project.decision('image_0') is None
    review.doc.set_keep([(review.doc.instances[0], False)])
    monkeypatch.setattr(page_review, 'ask_unsaved', lambda parent: 'accept')
    review.doc.set_keep([(review.doc.instances[0], True)])
    review.group.setText('burst_a')
    review.mark_meta_dirty()
    press(window, Qt.Key.Key_Left)
    settle(qapp, window)
    assert window.project.decision('image_1')['group'] == 'burst_a' and review.current_id == 'image_0'


def test_class_keys_label_selected_or_all_kept_masks_and_undo(qapp, window):
    need_active(qapp, window)
    window.project.ensure_class('bracket')
    window.project.ensure_class('clip')
    open_image(qapp, window)
    review = window.review
    review.rebuild_classes()
    assert list(review.class_buttons) == ['bracket', 'clip'] and review.class_buttons['bracket'].key == '1'
    assert [i['label'] for i in review.doc.instances] == ['bracket']                  # first class is the default
    press(window, Qt.Key.Key_2)                             # nothing selected: all kept masks
    assert [i['label'] for i in review.doc.instances] == ['clip'] and review.last_class == 'clip'
    press(window, Qt.Key.Key_Z, CTRL)
    assert [i['label'] for i in review.doc.instances] == ['bracket'] and not window.dirty
    review.canvas.set_selection([review.doc.instances[0]])
    press(window, Qt.Key.Key_1)
    assert review.doc.instances[0]['label'] == 'bracket'
    press(window, Qt.Key.Key_9)                             # no ninth class: ignored
    assert review.doc.instances[0]['label'] == 'bracket'


def test_brush_with_nothing_selected_starts_a_new_mask_and_one_undo_removes_it(qapp, window):
    open_image(qapp, window)
    review = window.review
    canvas = review.canvas
    review.set_tool('brush')
    canvas.set_selection([])
    start = canvas.mapFromScene(80, 50)
    viewport = canvas.viewport()
    QTest.mousePress(viewport, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, start)
    for step in range(1, 5):
        QTest.mouseMove(viewport, start + QPoint(step * 6, 0))
    QTest.mouseRelease(viewport, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, start + QPoint(24, 0))
    assert len(review.doc.instances) == 2 and review.doc.instances[1]['mask'].any() and window.dirty
    assert review.canvas.active is review.doc.instances[1]
    review.doc.undo()
    assert len(review.doc.instances) == 1 and not window.dirty
    review.set_tool('erase')
    canvas.set_selection([])
    QTest.mousePress(viewport, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, start)
    QTest.mouseRelease(viewport, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, start)
    assert review.doc.stack.count() == 1 + 1 - 1 or not window.dirty                  # erasing nothing records nothing


def test_merge_mode_adds_masks_without_discarding_edits(qapp, window, bundle):
    open_image(qapp, window)
    review = window.review
    window.bundle.by_proposal['image_0:sam3_text:0']['kept_masks'] = 1
    review.merge_check.setChecked(True)
    review.methods.setCurrentIndex(0)                       # "Manual / empty" replaces: merge only applies to real methods
    settle(qapp, window)
    assert not review.doc.instances
    review.methods.setCurrentIndex(1)
    settle(qapp, window)
    assert len(review.doc.instances) == 1
    review.doc.set_keep([(review.doc.instances[0], False)])
    before = len(review.doc.instances)
    review.methods.blockSignals(True)
    review.methods.setCurrentIndex(0)
    review.methods.setCurrentIndex(1)
    review.methods.blockSignals(False)
    review.change_proposal(user=True)
    settle(qapp, window)
    assert len(review.doc.instances) == before + 1 and window.dirty                    # appended, edits kept
    assert review.doc.instances[0]['keep'] is False
    review.doc.undo()
    assert len(review.doc.instances) == before


# -- explorer ------------------------------------------------------------------------------------

def test_explorer_filters_sorting_and_navigation_order(qapp, window):
    project = window.project
    approve(project, 3)
    project.save('image_3', None, [], 'excluded')
    window.explorer.refresh_all()
    assert window.explorer.visible_ids() == [f'image_{n}' for n in range(8)]
    window.explorer.status_filter.setCurrentText('Accepted')
    assert window.explorer.visible_ids() == ['image_0', 'image_1', 'image_2']
    window.explorer.status_filter.setCurrentText('Unreviewed')
    assert window.explorer.visible_ids() == ['image_4', 'image_5', 'image_6', 'image_7']
    window.explorer.status_filter.setCurrentText('All')
    window.explorer.search.setText('session_5')
    assert window.explorer.visible_ids() == ['image_5']
    window.explorer.search.setText('')
    window.explorer.table.sortByColumn(0, Qt.SortOrder.DescendingOrder)
    assert window.explorer.visible_ids()[0] == 'image_7'
    assert window.explorer.bar.value() == 4 and window.explorer.unreviewed_chip.text() == '4 to review'
    open_image(qapp, window, 'image_7')
    window.review.go(1)                                      # Previous/Next follow the explorer's current order
    settle(qapp, window)
    assert window.review.current_id == 'image_6'


# -- document, canvas ---------------------------------------------------------------------------

def make_document(shape=(64, 96)):
    first, second = np.zeros(shape, bool), np.zeros(shape, bool)
    first[10:40, 20:60] = True
    second[44:60, 10:30] = True
    document = ReviewDocument()
    document.load(shape, [{'id': 0, 'label': 'a', 'mask': first, 'keep': True},
                          {'id': 1, 'label': 'a', 'mask': second, 'keep': True}])
    return document


def test_document_operations_undo_redo_and_clean_state(qapp):
    doc = make_document()
    first, second = doc.instances
    original = [i['mask'].copy() for i in doc.instances]
    events = []
    doc.modifiedChanged.connect(events.append)
    doc.set_label([first], 'b')
    doc.set_keep([(second, False)])
    doc.delete([first])
    assert doc.instances == [second] and doc.modified and events[-1] is True
    doc.undo(), doc.undo(), doc.undo()
    assert doc.instances[0] is first and first['label'] == 'a' and second['keep'] and not doc.modified and events[-1] is False
    merged = doc.merge([first, second])
    assert merged['mask'].sum() == original[0].sum() + original[1].sum() and doc.instances == [merged]
    doc.undo()
    assert doc.instances[0] is first and doc.instances[1] is second                    # the very same objects come back
    pieces = doc.split(first)
    assert pieces == 1                                                                 # already one connected piece
    first['mask'][50:52, 80:82] = True
    assert doc.split(first) == 2 and len(doc.instances) == 3
    doc.undo()
    assert len(doc.instances) == 2 and doc.instances[0] is first
    copy = doc.duplicate(first)
    assert np.array_equal(copy['mask'], first['mask']) and copy['mask'] is not first['mask'] and doc.instances[1] is copy
    doc.undo()
    ring = np.zeros((64, 96), bool)
    ring[10:30, 10:30] = True
    ring[15:25, 15:25] = False
    first['mask'][:] = ring
    assert doc.fill_holes(first) and first['mask'][20, 20] and not first['mask'][5, 5]
    doc.undo()
    assert not first['mask'][20, 20]
    first['mask'][2:3, 2:3] = True                                                     # a one-pixel speck
    assert doc.remove_specks(first, 5) and not first['mask'][2, 2] and first['mask'][12, 12]
    doc.undo(), doc.redo()
    assert not first['mask'][2, 2]


def test_stack_clean_index_clears_dirty_only_when_back_at_saved_state(qapp):
    doc = make_document()
    doc.set_label([doc.instances[0]], 'x')
    doc.mark_clean()
    assert not doc.modified
    doc.set_label([doc.instances[0]], 'y')
    assert doc.modified
    doc.undo()
    assert not doc.modified
    doc.undo()
    assert doc.modified                                       # one step before the saved state is different again
    doc.redo()
    assert not doc.modified


def make_canvas(qapp, doc=None):
    doc = doc or make_document()
    canvas = Canvas()
    canvas.resize(900, 600)
    canvas.show()
    canvas.display(np.full((64, 96, 3), 120, np.uint8), doc, lambda label: '#E11D48')
    qapp.processEvents()
    return canvas, doc


def drag(canvas, points):
    viewport = canvas.viewport()
    first = canvas.mapFromScene(*points[0])
    QTest.mousePress(viewport, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, first)
    for x, y in points[1:]:
        QTest.mouseMove(viewport, canvas.mapFromScene(x, y))
    QTest.mouseRelease(viewport, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, canvas.mapFromScene(*points[-1]))


def test_canvas_brush_and_eraser_are_undoable_strokes(qapp):
    canvas, doc = make_canvas(qapp)
    first = doc.instances[0]
    before = first['mask'].copy()
    canvas.set_selection([first])
    canvas.set_tool('brush')
    canvas.set_radius(3)
    drag(canvas, [(70, 20), (75, 22), (80, 24)])
    assert first['mask'][22, 75] and first['mask'].sum() > before.sum() and doc.stack.undoText() == 'Brush stroke'
    doc.undo()
    assert np.array_equal(first['mask'], before)
    doc.redo()
    assert first['mask'][22, 75]
    canvas.set_tool('erase')
    canvas.set_radius(4)
    drag(canvas, [(30, 25), (40, 25), (50, 25)])
    assert not first['mask'][25, 40] and first['mask'][12, 22]
    doc.undo()
    assert first['mask'][25, 40]
    count = doc.stack.count()
    canvas.set_selection([])
    drag(canvas, [(5, 5), (6, 5)])                          # erasing with nothing selected changes nothing
    assert doc.stack.count() == count


def test_canvas_overlay_recomposes_only_the_touched_region(qapp):
    canvas, doc = make_canvas(qapp)
    first = doc.instances[0]
    canvas.set_selection([first])
    canvas.set_tool('brush')
    canvas.set_radius(2)
    before = canvas.overlay.copy()
    drag(canvas, [(80, 50), (82, 50)])
    changed = np.argwhere((canvas.overlay != before).any(axis=2))
    assert len(changed) > 0
    assert changed[:, 1].min() >= 60 and changed[:, 0].min() >= 35        # nothing near the other masks moved
    assert canvas.overlay[..., 3].max() > 0 and canvas.overlay[20, 30, 3] > 0   # filled mask is visible


def test_canvas_lasso_adds_and_subtracts(qapp):
    canvas, doc = make_canvas(qapp)
    first = doc.instances[0]
    canvas.set_selection([first])
    canvas.set_tool('lasso')
    area = first['mask'].sum()
    for point in [(65, 5), (90, 5), (90, 25), (65, 25)]:
        QTest.mouseClick(canvas.viewport(), Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier,
                         canvas.mapFromScene(*point))
    QTest.keyClick(canvas, Qt.Key.Key_Return)
    assert first['mask'].sum() > area + 300 and first['mask'][15, 75]
    grown = first['mask'].sum()
    for point in [(66, 6), (80, 6), (80, 20), (66, 20)]:
        QTest.mouseClick(canvas.viewport(), Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier,
                         canvas.mapFromScene(*point))
    QTest.keyClick(canvas, Qt.Key.Key_Return, Qt.KeyboardModifier.ShiftModifier)
    assert first['mask'].sum() < grown and not first['mask'][12, 72]
    doc.undo(), doc.undo()
    assert first['mask'].sum() == area
    for point in [(1, 1), (5, 1), (5, 5)]:                    # Escape cancels a half-drawn polygon
        QTest.mouseClick(canvas.viewport(), Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier,
                         canvas.mapFromScene(*point))
    QTest.keyClick(canvas, Qt.Key.Key_Escape)
    assert first['mask'].sum() == area and not canvas._lasso


def test_holding_space_pans_with_any_tool_and_never_sticks(qapp):
    canvas, doc = make_canvas(qapp)
    canvas.set_tool('brush')
    canvas.setFocus()
    assert canvas.dragMode() == Canvas.DragMode.NoDrag
    QTest.keyPress(canvas, Qt.Key.Key_Space)
    assert canvas._panning and canvas.dragMode() == Canvas.DragMode.ScrollHandDrag
    QTest.keyRelease(canvas, Qt.Key.Key_Space)
    assert not canvas._panning and canvas.dragMode() == Canvas.DragMode.NoDrag
    QTest.keyPress(canvas, Qt.Key.Key_Space)
    canvas.clearFocus()                                        # focus lost while Space is held: pan mode must end
    qapp.processEvents()
    assert not canvas._panning and canvas.tool == 'brush'


def test_canvas_selection_hover_zoom_and_layers_survive_new_images(qapp):
    canvas, doc = make_canvas(qapp)
    first, second = doc.instances
    assert canvas.pick(QPointF(30, 20)) is first and canvas.pick(QPointF(20, 50)) is second and canvas.pick(QPointF(0, 0)) is None
    changes = []
    canvas.selectionChanged.connect(lambda: changes.append(1))
    QTest.mouseClick(canvas.viewport(), Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, canvas.mapFromScene(30, 20))
    QTest.mouseClick(canvas.viewport(), Qt.MouseButton.LeftButton, Qt.KeyboardModifier.ControlModifier, canvas.mapFromScene(20, 50))
    assert canvas.selection == [first, second] and canvas.active is second and len(changes) == 2
    doc.set_keep([(first, False)])
    assert canvas.pick(QPointF(30, 20)) is first and canvas.pick(QPointF(20, 50)) is second   # dropped masks stay clickable
    scale = abs(canvas.transform().m11())
    canvas.zoom_by(2)
    assert abs(abs(canvas.transform().m11()) / scale - 2) < 1e-6 and not canvas.fit_mode
    canvas.fit()
    assert canvas.fit_mode
    canvas.cycle_opacity()
    assert canvas.toggle_masks() is False and canvas.toggle_masks() is True
    for _ in range(3):                                         # opening further images must not leave dangling layers
        canvas.display(np.full((64, 96, 3), 90, np.uint8), make_document(), lambda label: '#2563EB')
        canvas.set_tool('brush')
        canvas.set_tool('lasso')
        qapp.processEvents()
    assert canvas.scene().items().__len__() == 4               # base, overlay, brush ring, lasso preview


# -- augmentation page, classes, jobs -----------------------------------------------------------

def test_augmentation_page_presets_preview_and_project_storage(qapp, window):
    from review_app import augment
    need_active(qapp, window)
    open_image(qapp, window)
    page = window.augment
    window.go(2)
    qapp.processEvents()
    assert page.source is not None and all(cell.source is not None for cell in page.cells)
    assert page.presets.currentText() == 'Metal parts (default)' and 'Metal parts' in page.summary()
    first_seed = page.seed
    press(window, Qt.Key.Key_R)                              # R re-rolls only on this page
    assert page.seed == first_seed + 1
    page.widgets['degrees'].setValue(30)
    assert page.presets.currentText() == 'Custom' and window.project.pref('augmentation')['degrees'] == 30
    page.select_preset('Heavy')
    assert page.widgets['mosaic'].value() == .8 and page.presets.currentText() == 'Heavy'
    page.widgets['copy_paste_mode'].setCurrentText('mixup')
    page.widgets['degrees'].setValue(12)
    page.set_config(augment.defaults())                      # widgets reset, but the project still holds the last choice
    assert page.widgets['degrees'].value() == 180
    page.load_project()                                      # the project remembers the last configuration
    assert page.widgets['copy_paste_mode'].currentText() == 'mixup' and page.widgets['degrees'].value() == 12
    window.go(1)
    press(window, Qt.Key.Key_R)
    assert page.seed == first_seed + 1                       # not on other pages


def test_training_config_carries_every_augmentation_key_and_never_starts_unconfirmed(qapp, window, monkeypatch, tmp_path):
    from review_app import augment
    training = window.training
    window.augment.select_preset('Conservative')
    config = training.config()
    assert {k: config[k] for k in augment.TRAIN_KEYS} == augment.train_kwargs(window.augment.config())
    assert config['degrees'] == 15 and config['epochs'] == 80 and config['family'] == 'YOLOE'
    assert 'Conservative' in training.aug_summary.text()
    training.start()                                         # no dataset chosen
    assert window.errors and 'reviewed dataset' in window.errors[-1] and not window.job.running
    project = window.project
    approve(project)
    dataset = export_dataset(project, tmp_path / 'ds')
    training.dataset.setText(str(dataset))
    monkeypatch.setattr(QMessageBox, 'question', staticmethod(lambda *args, **kwargs: QMessageBox.StandardButton.No))
    training.start()
    assert not window.job.running


def test_job_output_drives_progress_plot_log_and_validation_weights(qapp, window, tmp_path):
    runner, training, validation = window.job, window.training, window.validation
    runner.output_dir, runner.action = tmp_path / 'run', 'train'
    (runner.output_dir / 'train').mkdir(parents=True)
    (runner.output_dir / 'train/results.csv').write_text('epoch,metrics/mAP50-95(M),metrics/mAP50-95(B)\n1,0.4,0.6\n2,0.5,0.7\n')
    training.poll_metrics()
    assert [r['metrics/mAP50-95(M)'] for r in training.plot.rows] == ['0.4', '0.5']

    class Saved:
        def readAllStandardOutput(self):
            return QByteArray(b'hello\nROBOQC_EVENT {"status":"epoch","epoch":3,"epochs":8}\n'
                              b'ROBOQC_EVENT {"status":"finished","weights":"runs/best.pt"}\n')
    runner.process = Saved()
    runner.read()
    assert training.progress.value() == 3 and 'hello' in training.log.toPlainText()
    assert validation.weights.text() == 'runs/best.pt'
    assert (runner.output_dir / 'console.log').is_file()
    runner.process = None


def test_class_dialog_adds_renames_merges_and_protects_used_classes(qapp, window, monkeypatch):
    from review_app.dialogs import ClassDialog
    project = window.project
    approve(project)
    dialog = ClassDialog(project, window)
    assert dialog.table.rowCount() == 1 and dialog.table.item(0, 2).text() == '6' and dialog.table.item(0, 0).text() == '1'
    monkeypatch.setattr(QInputDialog, 'getText', staticmethod(lambda *a, **k: ('clip', True)))
    dialog.add()
    assert project.class_names() == ['part', 'clip'] and dialog.changed
    dialog.table.selectRow(0)
    monkeypatch.setattr(QInputDialog, 'getText', staticmethod(lambda *a, **k: ('bracket', True)))
    dialog.rename()
    assert project.class_names() == ['bracket', 'clip'] and project.class_usage()['bracket'] == 6
    warnings = []
    monkeypatch.setattr(QMessageBox, 'warning', staticmethod(lambda parent, title, text: warnings.append(text)))
    dialog.delete()                                           # used by accepted masks: refused with an explanation
    assert project.class_names() == ['bracket', 'clip'] and 'used by' in warnings[-1]
    dialog.table.selectRow(1)
    dialog.delete()
    assert project.class_names() == ['bracket']
    dialog.close()


def test_auto_group_dialog_previews_groups(qapp, window):
    from review_app.dialogs import AutoGroupDialog
    dialog = AutoGroupDialog(window.bundle.images, 0, window)
    assert '8 groups over 8 images' in dialog.preview.text()
    dialog.seconds.setValue(30)
    assert 'near' in dialog.preview.text() or 'Applies' in dialog.preview.text()
    dialog.close()


def test_offline_export_from_the_page_writes_dataset_and_zip(qapp, window, monkeypatch, tmp_path):
    from PySide6.QtWidgets import QFileDialog
    project = window.project
    approve(project)
    dataset = export_dataset(project, tmp_path / 'ds')
    page = window.augment
    page.export_dataset.setText(str(dataset))
    page.copies.setValue(2)
    out_parent = tmp_path / 'out'
    out_parent.mkdir()
    monkeypatch.setattr(QFileDialog, 'getExistingDirectory', staticmethod(lambda *args, **kwargs: str(out_parent)))
    page.export()
    settle(qapp, window, 30)
    created = [p for p in out_parent.iterdir() if p.is_dir()]
    assert len(created) == 1 and (created[0] / 'manifest.json').is_file() and created[0].with_suffix('.zip').is_file()
    manifest = json.loads((created[0] / 'manifest.json').read_text())
    assert manifest['augmentation']['copies'] == 2 and manifest['schema'] == 'roboqc-augmented-dataset-1'
    assert window.infos and 'augmented images added' in window.infos[-1][1]
    assert window.training.dataset.text() == str(created[0])
    assert not window.errors


def test_old_project_without_new_keys_opens_in_the_new_window(qapp, bundle, tmp_path):
    from review_app.app import Window
    project = Project(tmp_path / 'old', bundle)
    approve(project)
    state = json.loads((project.folder / 'project.json').read_text())
    state.pop('classes'), state.pop('prefs', None)
    (project.folder / 'project.json').write_text(json.dumps(state))
    win = Window()
    win.error = lambda message: pytest.fail(message)
    win.bundle_loaded((bundle, Project(project.folder, bundle)))
    assert win.explorer.table.rowCount() == 8 and win.project.stats()['accepted'] == 6
    assert win.explorer.accepted_chip.text() == '6 accepted'
    win.review.discard_edits()
    win.close()
