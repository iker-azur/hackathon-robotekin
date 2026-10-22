"""Review page: toolbar, canvas, mask list, classes, scene and decision panels."""

from collections import OrderedDict

import numpy as np
from PySide6.QtCore import QPoint, Qt
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (
    QAbstractItemView, QButtonGroup, QCheckBox, QComboBox, QFrame, QGridLayout, QHBoxLayout, QLabel, QLineEdit,
    QListWidget, QListWidgetItem, QMenu, QPushButton, QSpinBox, QVBoxLayout, QWidget,
)

from .canvas import Canvas
from .dialogs import ClassDialog, ask_unsaved
from .document import ReviewDocument
from .shortcuts import hint
from .suggest import suggest_class
from .theme import GAP, SIDE_W
from .widgets import ActionButton, Card, Chip, ControlColumn, ElidedLabel, FormGrid, Keycap, Page, swatch_icon

CACHE_BYTES = 128 * 1024**2
SPECK_PIXELS = 50
STATUS_CHIPS = {'unreviewed': ('Unreviewed', ''), 'accepted': ('Accepted', 'ok'), 'excluded': ('Excluded', 'warn')}


class ReviewPage(Page):
    def __init__(self, win):
        super().__init__('Review', 'Double-click an image in the Explorer, or press N for the next unreviewed one.')
        self.win = win
        self.doc = ReviewDocument()
        self.cache = OrderedDict()
        self.current_id = self.current_proposal = None
        self.rgb = None
        self.last_class = None
        self.suggested = None
        self.displayed_index = 0
        self.meta_dirty = False
        self._loading = False
        self._syncing = False
        self.canvas = Canvas()
        self.canvas.color_for = self.color_for
        self.canvas.new_label = lambda: self.current_class()
        self._build_header()
        body = QHBoxLayout()
        body.setSpacing(16)
        body.addLayout(self._build_work_area(), 1)
        body.addWidget(self._build_side_panel())
        self.root.addLayout(body, 1)
        self.canvas.selectionChanged.connect(self.on_canvas_selection)
        self.canvas.keepToggled.connect(lambda inst: self.doc.set_keep([(inst, not inst.get('keep', True))]))
        self.canvas.strokeFinished.connect(lambda inst: self.refresh_list())
        self.canvas.hoverChanged.connect(lambda text: self.info.setText(text or self.default_info()))
        self.canvas.zoomChanged.connect(lambda scale: self.zoom_label.setText(f'{scale * 100:.0f} %'))
        self.doc.changed.connect(lambda region: self.refresh_list())
        self.doc.modifiedChanged.connect(lambda modified: self.win.update_title())
        self.doc.stack.canUndoChanged.connect(self.undo_button.setEnabled)
        self.doc.stack.canRedoChanged.connect(self.redo_button.setEnabled)
        self.undo_button.setEnabled(False)
        self.redo_button.setEnabled(False)
        self.set_tool('select')

    # -- construction -------------------------------------------------------------------

    @staticmethod
    def _no_focus(*buttons):
        for button in buttons:
            button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        return buttons[0] if len(buttons) == 1 else buttons

    def _build_header(self):
        self.position = QLabel('')
        self.position.setProperty('muted', True)
        self.status_chip = Chip('', '')
        self.status_chip.hide()
        self.header.add_action(self.position)
        self.header.add_action(self.status_chip)

    def _build_work_area(self):
        column = QVBoxLayout()
        column.setSpacing(12)
        toolbar = Card()
        toolbar.body.setSpacing(GAP)
        toolbar.body.setContentsMargins(12, 12, 12, 12)
        row = QHBoxLayout()
        row.setSpacing(GAP)
        self.prev_button = self._no_focus(ActionButton('Previous', 'prev_image'))
        self.next_button = self._no_focus(ActionButton('Next', 'next_image'))
        self.unreviewed_button = self._no_focus(ActionButton('Next unreviewed', 'next_unreviewed'))
        for button, callback in [(self.prev_button, lambda: self.go(-1)), (self.next_button, lambda: self.go(1)),
                                 (self.unreviewed_button, self.next_unreviewed)]:
            button.clicked.connect(callback)
            row.addWidget(button)
        row.addStretch(1)
        row.addWidget(QLabel('Method'))
        self.methods = QComboBox()
        self.methods.setMinimumWidth(260)
        self.methods.setFocusPolicy(Qt.FocusPolicy.ClickFocus)
        self.methods.currentIndexChanged.connect(self.on_method_changed)
        row.addWidget(self.methods, 1)
        row.addWidget(Keycap(hint('prev_method')))
        row.addWidget(Keycap(hint('next_method')))
        toolbar.add_layout(row)
        self.merge_check = QCheckBox(f'Add masks from other methods, keep my edits  ({hint("merge_mode")})')
        self.merge_check.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.merge_check.setToolTip('Off: choosing another method replaces the masks. '
                                    'On: its masks are appended so you can combine the best of several methods.')
        row2 = QHBoxLayout()
        row2.setSpacing(GAP)
        self.tool_buttons = {}
        group = QButtonGroup(self)
        group.setExclusive(True)
        for name, label in [('select', 'Select'), ('pan', 'Pan'), ('brush', 'Brush'), ('erase', 'Erase'), ('lasso', 'Lasso')]:
            button = self._no_focus(ActionButton(label, 'tool_' + name, checkable=True))
            button.clicked.connect(lambda checked=False, n=name: self.set_tool(n))
            group.addButton(button)
            self.tool_buttons[name] = button
            row2.addWidget(button)
        self.brush = QSpinBox()
        self.brush.setRange(1, 200)
        self.brush.setValue(12)
        self.brush.setSuffix(' px')
        self.brush.setFixedWidth(96)
        self.brush.setToolTip(f'Brush / eraser radius  ({hint("brush_smaller")} smaller, {hint("brush_larger")} larger)')
        self.brush.valueChanged.connect(self.canvas.set_radius)
        row2.addWidget(self.brush)
        row2.addStretch(1)
        toolbar.add_layout(row2)
        toolbar.add(self.merge_check)
        column.addWidget(toolbar)
        frame = QFrame()
        frame.setObjectName('canvasFrame')
        inner = QVBoxLayout(frame)
        inner.setContentsMargins(1, 1, 1, 1)
        inner.addWidget(self.canvas)
        column.addWidget(frame, 1)
        strip = QHBoxLayout()
        strip.setSpacing(GAP)
        self.info = ElidedLabel(self.default_info())
        self.info.setProperty('muted', True)
        strip.addWidget(self.info, 1)
        self.undo_button = self._no_focus(ActionButton('Undo', 'undo'))
        self.redo_button = self._no_focus(ActionButton('Redo', 'redo'))
        self.undo_button.clicked.connect(self.doc.undo)
        self.redo_button.clicked.connect(self.doc.redo)
        strip.addWidget(self.undo_button)
        strip.addWidget(self.redo_button)
        self.masks_button = self._no_focus(ActionButton('Masks', 'toggle_masks', checkable=True))
        self.masks_button.setChecked(True)
        self.masks_button.clicked.connect(self.toggle_masks)
        self.opacity_button = self._no_focus(ActionButton('Opacity', 'cycle_opacity'))
        self.opacity_button.clicked.connect(self.canvas.cycle_opacity)
        self.fit_button = self._no_focus(ActionButton('Fit', 'zoom_fit'))
        self.fit_button.clicked.connect(self.canvas.fit)
        self.zoom100_button = self._no_focus(ActionButton('100 %', 'zoom_100'))
        self.zoom100_button.clicked.connect(self.canvas.zoom_100)
        for button in (self.masks_button, self.opacity_button, self.fit_button, self.zoom100_button):
            strip.addWidget(button)
        self.zoom_label = QLabel('')
        self.zoom_label.setMinimumWidth(48)
        self.zoom_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        strip.addWidget(self.zoom_label)
        column.addLayout(strip)
        return column

    def _build_side_panel(self):
        panel = QWidget()
        panel.setFixedWidth(SIDE_W + 18)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(16)
        scroll = ControlColumn(SIDE_W)
        layout.addWidget(scroll, 1)

        self.masks_card = Card()
        self.masks_card.body.setContentsMargins(14, 14, 14, 14)
        self.masks_card.body.setSpacing(10)
        head = QHBoxLayout()
        head.setSpacing(GAP)
        self.masks_title = QLabel('Masks')
        self.masks_title.setObjectName('cardTitle')
        head.addWidget(self.masks_title, 1)
        more = self._no_focus(ActionButton('More ▾'))
        more.setMenu(self.build_more_menu())
        head.addWidget(more)
        self.masks_card.add_layout(head)
        self.mask_list = QListWidget()
        self.mask_list.setMinimumHeight(120)
        self.mask_list.setToolTip('Tick = kept in the dataset. Click a mask on the image to select it; '
                                  'double-click toggles keep. Brush with nothing selected starts a new mask.')
        self.mask_list.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.mask_list.setFocusPolicy(Qt.FocusPolicy.ClickFocus)
        self.mask_list.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.mask_list.customContextMenuRequested.connect(self.list_menu)
        self.mask_list.itemChanged.connect(self.on_item_changed)
        self.mask_list.itemSelectionChanged.connect(self.on_list_selection)
        self.masks_card.add(self.mask_list, 1)
        triage = QHBoxLayout()
        triage.setSpacing(GAP)
        for text, sid, callback in [('All', 'check_all', self.check_all), ('None', 'check_none', self.check_none),
                                    ('Invert', 'check_invert', self.check_invert)]:
            button = self._no_focus(ActionButton(text, sid))
            button.clicked.connect(callback)
            triage.addWidget(button)
        self.masks_card.add_layout(triage)
        actions = QHBoxLayout()
        actions.setSpacing(GAP)
        for text, sid, callback in [('New', 'new_instance', self.new_instance), ('Delete', 'delete_instance', self.delete_selected),
                                    ('Merge', 'merge_instances', self.merge_selected)]:
            button = self._no_focus(ActionButton(text, sid))
            button.clicked.connect(callback)
            actions.addWidget(button)
        self.masks_card.add_layout(actions)
        scroll.add(self.masks_card)

        self.class_card = Card()
        self.class_card.body.setContentsMargins(14, 14, 14, 14)
        self.class_card.body.setSpacing(10)
        head = QHBoxLayout()
        head.setSpacing(GAP)
        title = QLabel('Class')
        title.setObjectName('cardTitle')
        title.setToolTip('Click a class to label the selected masks (all kept masks when none is selected). '
                         'New masks use the highlighted class.')
        head.addWidget(title, 1)
        manage = self._no_focus(ActionButton('Manage…', 'manage_classes'))
        manage.clicked.connect(self.manage_classes)
        head.addWidget(manage)
        self.class_card.add_layout(head)
        self.suggest_chip = Chip('', 'info', wrap=True)
        self.suggest_chip.hide()
        self.class_card.add(self.suggest_chip)
        self.class_box = QGridLayout()
        self.class_box.setSpacing(GAP)
        self.class_card.add_layout(self.class_box)
        scroll.add(self.class_card)

        scene = Card('Scene')
        form = FormGrid()
        group_row = QHBoxLayout()
        group_row.setSpacing(GAP)
        self.group = QLineEdit()
        self.group.setPlaceholderText('Folder or burst name')
        self.group.textEdited.connect(self.mark_meta_dirty)
        group_row.addWidget(self.group, 1)
        auto = self._no_focus(QPushButton('Auto…'))
        auto.setToolTip('Choose how unreviewed images get their default group')
        auto.clicked.connect(self.auto_group)
        group_row.addWidget(auto)
        form.add_wide('Capture group', group_row, 'Near-identical frames of one scene MUST share a group so they stay in one split.')
        scene.add_layout(form)
        self.negative = QCheckBox('Confirmed empty scene (no target parts)')
        self.negative.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.negative.toggled.connect(self.mark_meta_dirty)
        scene.add(self.negative)
        scroll.add(scene)

        decision = Card()
        decision.body.setContentsMargins(12, 12, 12, 12)
        self.accept_next_button = self._no_focus(ActionButton('Accept & next', 'accept_next', role='primary'))
        self.accept_next_button.clicked.connect(self.accept_next)
        decision.add(self.accept_next_button)
        row = QHBoxLayout()
        row.setSpacing(GAP)
        self.accept_button = self._no_focus(ActionButton('Accept', 'accept'))
        self.accept_button.clicked.connect(self.accept)
        self.exclude_button = self._no_focus(ActionButton('Exclude', 'exclude', role='danger'))
        self.exclude_button.clicked.connect(self.exclude)
        row.addWidget(self.accept_button)
        row.addWidget(self.exclude_button)
        decision.add_layout(row)
        wrapper = QHBoxLayout()
        wrapper.setContentsMargins(0, 0, 10, 0)
        wrapper.addWidget(decision)
        layout.addLayout(wrapper)
        return panel

    def build_more_menu(self):
        menu = QMenu(self)
        manager = self.win.manager
        for sid in ('toggle_keep', 'keep_only', 'duplicate_instance', 'split_instance'):
            menu.addAction(manager.actions[sid])
        menu.addSeparator()
        fill = QAction('Fill holes in selected mask', menu)
        fill.triggered.connect(self.fill_holes)
        menu.addAction(fill)
        specks = QAction(f'Remove specks under {SPECK_PIXELS} px', menu)
        specks.triggered.connect(self.remove_specks)
        menu.addAction(specks)
        return menu

    def default_info(self):
        return f'Click a mask to select · wheel zooms · hold Space to pan · {hint("show_shortcuts")} lists all shortcuts'

    # -- state ---------------------------------------------------------------------------

    @property
    def dirty(self):
        return self.doc.modified or self.meta_dirty

    def mark_meta_dirty(self, *args):
        if not self._loading:
            self.meta_dirty = True
            self.win.update_title()

    @property
    def project(self):
        return self.win.project

    @property
    def instances(self):
        return self.doc.instances

    def color_for(self, label):
        return self.project.color_of(label) if self.project else '#2563EB'

    def current_class(self):
        names = self.project.class_names() if self.project else []
        return self.suggested or (self.last_class if self.last_class in names else None) or (names[0] if names else 'part')

    def clear(self):
        """A different bundle was opened."""
        self.current_id = self.current_proposal = self.rgb = None
        self.suggested = None
        self.meta_dirty = False
        self.cache.clear()
        self.doc.load((0, 0), [])
        self.canvas.clear()
        self.methods.blockSignals(True)
        self.methods.clear()
        self.methods.blockSignals(False)
        self.refresh_list()
        self.rebuild_classes()
        self.header.subtitle.setText('Double-click an image in the Explorer, or press N for the next unreviewed one.')
        self.position.setText('')
        self.status_chip.hide()

    # -- opening images ---------------------------------------------------------------

    def guard(self, action, on_cancel=None):
        """Run `action` once unsaved edits are saved or discarded; offer the choice."""
        if not self.dirty:
            action()
            return
        choice = ask_unsaved(self.win)
        if choice == 'cancel':
            if on_cancel:
                on_cancel()
        elif choice == 'discard':
            self.discard_edits()
            action()
        else:
            self.save_review('accepted', then=action)

    def discard_edits(self):
        self.doc.mark_clean()
        self.meta_dirty = False
        self.win.update_title()

    def open_image(self, image_id):
        self.guard(lambda: self._open(image_id))

    def _open(self, image_id):
        bundle, project = self.win.bundle, self.win.project
        if bundle.by_image[image_id]['status'] != 'ok':
            self.win.error('Invalid image in source manifest')
            return
        def load(progress):
            if image_id in self.cache:
                rgb = self.cache.pop(image_id)
            else:
                rgb = bundle.rgb(image_id)
            self.cache[image_id] = rgb
            while sum(value.nbytes for value in self.cache.values()) > CACHE_BYTES and len(self.cache) > 1:
                self.cache.popitem(last=False)
            return image_id, rgb, project.load_masks(image_id)
        self.win.background(load, self.image_loaded)

    def image_loaded(self, result):
        image_id, self.rgb, saved = result
        self.current_id = image_id
        bundle, project = self.win.bundle, self.win.project
        image = bundle.by_image[image_id]
        decision = project.decision(image_id) or {}
        self._loading = True
        self.header.subtitle.setText(image['source_relative_path'])
        self.suggested = None if decision.get('status') == 'accepted' else suggest_class(
            image['source_relative_path'], project.class_names())
        if self.suggested:
            project.ensure_class(self.suggested)
        self.group.setText(decision.get('group') or self.win.group_for(image))
        self.negative.setChecked(decision.get('negative', False))
        self.methods.blockSignals(True)
        self.methods.clear()
        self.methods.addItem('Saved reviewed masks' if saved or decision.get('negative') else 'Manual / empty', None)
        wanted = tuple(project.pref('method', ()) or ())
        selected = 0
        first_ok = 0
        for row in bundle.alternatives[image_id]:
            text = ' · '.join(part for part in (row['method'], row.get('prompt'), f"{row.get('kept_masks', '?')} masks") if part)
            if row['status'] != 'ok':
                text += ' · failed'
            self.methods.addItem(text, row['proposal_id'])
            index = self.methods.count() - 1
            if row['status'] == 'ok':
                first_ok = first_ok or index
                if wanted == (row['method'], row.get('prompt')) and not selected:
                    selected = index
        load_proposal = decision.get('status') != 'accepted'
        if load_proposal:
            selected = selected or first_ok
        else:
            selected = 0
        self.methods.setCurrentIndex(selected)
        self.methods.blockSignals(False)
        self.current_proposal = decision.get('proposal_id')
        self.displayed_index = 0
        self.meta_dirty = False
        self._loading = False
        self.win.go(1)
        for inst in saved:
            inst['keep'] = True
        self.doc.load(self.rgb.shape[:2], saved)
        self.canvas.display(self.rgb, self.doc, self.color_for)
        self.set_status(decision.get('status', 'unreviewed'))
        self.update_position()
        self.rebuild_classes()
        self.refresh_list()
        self.canvas.setFocus()
        if selected:
            self.win.after_task(lambda: self.change_proposal(user=False))

    def set_status(self, status):
        text, kind = STATUS_CHIPS.get(status, STATUS_CHIPS['unreviewed'])
        self.status_chip.set(text, kind)
        self.status_chip.show()

    def update_position(self):
        ids = self.win.explorer.visible_ids()
        if self.current_id in ids:
            self.position.setText(f'Image {ids.index(self.current_id) + 1} of {len(ids)}')
        else:
            self.position.setText('')

    # -- methods (proposals) ------------------------------------------------------------

    def on_method_changed(self, index):
        if self._loading or self.current_id is None:
            return
        self.change_proposal(user=True)

    def change_proposal(self, user=True):
        if self.current_id is None:
            return
        proposal = self.methods.currentData()
        requested = self.methods.currentIndex()
        image_id = self.current_id
        merge = self.merge_check.isChecked() and proposal is not None and user
        project, bundle = self.win.project, self.win.bundle
        def revert():
            self.methods.blockSignals(True)
            self.methods.setCurrentIndex(self.displayed_index)
            self.methods.blockSignals(False)
        def go():
            def load(progress):
                return bundle.candidates(proposal) if proposal else project.load_masks(image_id)
            def loaded(instances):
                if image_id != self.current_id:
                    return
                label = self.current_class()
                if proposal:
                    for inst in instances:
                        inst['label'] = label
                if merge:
                    self.doc.add(instances, 'Add masks from method')
                    self.win.status(f'Added {len(instances)} masks. Undo with {hint("undo")}.', 6000)
                else:
                    for inst in instances:
                        inst.setdefault('keep', True)
                    self.doc.load(self.rgb.shape[:2], instances)
                    self.current_proposal = proposal if proposal else (project.decision(image_id) or {}).get('proposal_id')
                    self.displayed_index = requested
                if user and proposal:
                    row = bundle.by_proposal[proposal]
                    project.set_pref('method', [row['method'], row.get('prompt')])
            def failed(message):
                revert()
                self.win.error(message)
            self.win.background(load, loaded, failed)
        if merge:
            go()
        else:
            self.guard(go, on_cancel=revert)

    def step_method(self, step):
        if self.methods.count() < 2:
            return
        self.methods.setCurrentIndex((self.methods.currentIndex() + step) % self.methods.count())

    def toggle_merge_mode(self):
        self.merge_check.toggle()

    # -- navigation ----------------------------------------------------------------------

    def go(self, step):
        ids = self.win.explorer.visible_ids()
        if self.current_id not in ids:
            return
        index = ids.index(self.current_id) + step
        if 0 <= index < len(ids):
            self.open_image(ids[index])

    def next_unreviewed(self, after=None):
        ids = self.win.explorer.visible_ids()
        if not ids:
            return False
        start = ids.index(after or self.current_id) + 1 if (after or self.current_id) in ids else 0
        project = self.win.project
        for image_id in ids[start:] + ids[:start]:
            if (project.decision(image_id) or {}).get('status') not in ('accepted', 'excluded'):
                if image_id != self.current_id:
                    self.open_image(image_id)
                    return True
        self.win.status('Every image in this view has been reviewed.', 6000)
        return False

    # -- tools ---------------------------------------------------------------------------

    def set_tool(self, name):
        self.canvas.set_tool(name)
        for key, button in self.tool_buttons.items():
            button.setChecked(key == name)

    def step_brush(self, direction):
        self.brush.setValue(max(1, self.brush.value() + direction * max(1, self.brush.value() // 6)))

    def toggle_masks(self):
        visible = self.canvas.toggle_masks()
        self.masks_button.setChecked(visible)

    # -- mask list ---------------------------------------------------------------------

    def refresh_list(self):
        self._syncing = True
        selected = {id(i) for i in self.canvas.selection}
        self.mask_list.clear()
        kept = 0
        for number, inst in enumerate(self.doc.instances, 1):
            keep = bool(inst.get('keep', True))
            kept += keep
            area = int(np.count_nonzero(inst['mask']))
            item = QListWidgetItem(f"{number}   {inst['label']}  ·  {area:,} px")
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Checked if keep else Qt.CheckState.Unchecked)
            item.setIcon(swatch_icon(self.color_for(inst['label'])))
            if not keep:
                item.setForeground(self.mask_list.palette().placeholderText())
            self.mask_list.addItem(item)
            if id(inst) in selected:
                item.setSelected(True)
            if inst is self.canvas.active:
                self.mask_list.setCurrentItem(item, self.mask_list.selectionModel().SelectionFlag.NoUpdate)
        self.masks_title.setText(f'Masks · {kept} kept of {len(self.doc.instances)}')
        self._syncing = False

    def on_item_changed(self, item):
        if self._syncing:
            return
        inst = self.doc.instances[self.mask_list.row(item)]
        self.doc.set_keep([(inst, item.checkState() == Qt.CheckState.Checked)])
        self.canvas.setFocus()                  # Space / Enter keep working on the image

    def on_list_selection(self):
        if self._syncing:
            return
        rows = sorted(index.row() for index in self.mask_list.selectedIndexes())
        insts = [self.doc.instances[r] for r in rows]
        current = self.mask_list.currentRow()
        active = self.doc.instances[current] if current in rows else (insts[-1] if insts else None)
        self.canvas.set_selection(insts, active, notify=False)
        self.update_class_buttons()
        self.canvas.setFocus()

    def on_canvas_selection(self):
        self._syncing = True
        self.mask_list.clearSelection()
        for row, inst in enumerate(self.doc.instances):
            if any(inst is s for s in self.canvas.selection):
                self.mask_list.item(row).setSelected(True)
            if inst is self.canvas.active:
                self.mask_list.setCurrentRow(row, self.mask_list.selectionModel().SelectionFlag.NoUpdate)
                self.mask_list.scrollToItem(self.mask_list.item(row))
        self._syncing = False
        self.update_class_buttons()

    def targets(self, fallback_kept=False):
        if self.canvas.selection:
            return list(self.canvas.selection)
        return [i for i in self.doc.instances if i.get('keep', True)] if fallback_kept else []

    def list_menu(self, position: QPoint):
        menu = QMenu(self)
        names = self.project.class_names() if self.project else []
        assign = menu.addMenu('Assign class')
        for name in names:
            action = assign.addAction(swatch_icon(self.color_for(name)), name)
            action.triggered.connect(lambda checked=False, n=name: self.assign_class(n))
        manager = self.win.manager
        for sid in ('toggle_keep', 'keep_only', 'delete_instance', 'duplicate_instance', 'merge_instances', 'split_instance'):
            menu.addAction(manager.actions[sid])
        menu.exec(self.mask_list.viewport().mapToGlobal(position))

    # -- mask operations -----------------------------------------------------------------

    def check_all(self):
        self.doc.set_keep([(i, True) for i in self.doc.instances])

    def check_none(self):
        self.doc.set_keep([(i, False) for i in self.doc.instances])

    def check_invert(self):
        self.doc.set_keep([(i, not i.get('keep', True)) for i in self.doc.instances])

    def toggle_keep(self):
        targets = self.targets()
        if targets:
            keep = not targets[-1].get('keep', True)
            self.doc.set_keep([(i, keep) for i in targets])

    def keep_only(self):
        targets = self.targets()
        if targets:
            self.doc.set_keep([(i, any(i is t for t in targets)) for i in self.doc.instances])

    def new_instance(self):
        if self.current_id is None:
            return
        inst = self.doc.new_empty(self.current_class())
        self.canvas.set_selection([inst], inst)
        self.set_tool('brush')

    def delete_selected(self):
        targets = self.targets()
        if targets:
            self.doc.delete(targets)

    def duplicate_selected(self):
        if self.canvas.active is not None:
            copy = self.doc.duplicate(self.canvas.active)
            self.canvas.set_selection([copy], copy)

    def merge_selected(self):
        merged = self.doc.merge(self.targets())
        if merged is None:
            self.win.status('Select two or more masks to merge (Ctrl-click on the image or in the list).', 5000)
        else:
            self.canvas.set_selection([merged], merged)

    def split_selected(self):
        inst = self.canvas.active
        if inst is None:
            return
        pieces = self.doc.split(inst)
        self.win.status(f'Split into {pieces} masks.' if pieces > 1 else 'That mask is a single piece.', 5000)

    def fill_holes(self):
        for inst in self.targets():
            self.doc.fill_holes(inst)

    def remove_specks(self):
        for inst in self.targets(fallback_kept=True):
            self.doc.remove_specks(inst, SPECK_PIXELS)

    def cycle_instance(self, step):
        insts = self.doc.instances
        if not insts:
            return
        index = next((i for i, inst in enumerate(insts) if inst is self.canvas.active), -1)
        chosen = insts[(index + step) % len(insts)]
        self.canvas.set_selection([chosen], chosen)

    # -- classes -------------------------------------------------------------------------

    def rebuild_classes(self):
        while self.class_box.count():
            item = self.class_box.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self.class_buttons = {}
        project = self.project
        if project is None:
            return
        for index, name in enumerate(project.class_names()):
            button = self._no_focus(ActionButton(name, f'class_{index + 1}' if index < 9 else None, checkable=True,
                                                 color=project.color_of(name), align_left=True))
            button.clicked.connect(lambda checked=False, n=name: self.assign_class(n))
            self.class_box.addWidget(button, index // 2, index % 2)
            self.class_box.setColumnStretch(index % 2, 1)
            self.class_buttons[name] = button
        if self.suggested and self.current_id:
            self.suggest_chip.set(f'Suggested from folder, saved when you accept', 'info')
            self.suggest_chip.show()
        else:
            self.suggest_chip.hide()
        self.update_class_buttons()

    def update_class_buttons(self):
        shown = {i['label'] for i in self.targets()} or ({self.canvas.active['label']} if self.canvas.active else set())
        for name, button in getattr(self, 'class_buttons', {}).items():
            button.setChecked(name in shown or (not shown and name == self.current_class()))

    def assign_class(self, name):
        if self.current_id is None or name not in self.project.class_names():
            return
        targets = self.targets(fallback_kept=True)
        self.last_class = name
        self.suggested = None                 # an explicit choice overrides the folder suggestion
        self.suggest_chip.hide()
        if targets:
            self.doc.set_label(targets, name)
        self.update_class_buttons()

    def assign_class_number(self, number):
        names = self.project.class_names() if self.project else []
        if 0 < number <= len(names):
            self.assign_class(names[number - 1])

    def manage_classes(self):
        if self.project is None:
            return
        def run():
            dialog = ClassDialog(self.project, self.win)
            dialog.exec()
            if dialog.changed:
                self.win.refresh_all()
                if self.current_id:
                    self._open(self.current_id)
                else:
                    self.rebuild_classes()
        self.guard(run)

    def auto_group(self):
        self.win.auto_group()

    # -- decisions -----------------------------------------------------------------------

    def accept(self):
        self.save_review('accepted')

    def accept_next(self):
        current = self.current_id
        self.save_review('accepted', then=lambda: self.next_unreviewed(after=current))

    def exclude(self):
        self.save_review('excluded')

    def save_review(self, status, then=None):
        if not self.current_id:
            return
        image_id, proposal = self.current_id, self.current_proposal
        instances = list(self.doc.instances)
        group, negative = self.group.text(), self.negative.isChecked()
        project = self.win.project
        def save(progress):
            return project.save(image_id, proposal, instances, status, group, negative)
        def saved(result):
            self.doc.mark_clean()
            self.meta_dirty = False
            self.set_status(status)
            self.win.status(f"Saved {status} revision {result['revision'][:8]}", 8000)
            self.win.after_save(image_id)
            self.rebuild_classes()
            self.win.update_title()
            if then:
                # the worker thread is still finishing here, so a follow-up background task must wait for it
                self.win.after_task(then)
        self.win.background(save, saved)
