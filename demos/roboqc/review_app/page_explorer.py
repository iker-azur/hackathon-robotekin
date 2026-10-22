"""Explorer page: open a bundle, see review progress, filter, sort and pick an image."""

from PySide6.QtCore import Qt
from PySide6.QtGui import QBrush, QColor
from PySide6.QtWidgets import (
    QAbstractItemView, QComboBox, QHBoxLayout, QHeaderView, QLabel, QLineEdit, QProgressBar, QTableWidget,
    QTableWidgetItem,
)

from .suggest import suggest_class
from .theme import GAP, PALETTE
from .widgets import ActionButton, Card, Chip, Page

COLUMNS = ['Source path', 'Size', 'Methods OK', 'Masks per method', 'Class', 'Group', 'Status']
STATUS_COLOR = {'accepted': PALETTE['success'], 'excluded': PALETTE['warn'], 'unreviewed': PALETTE['muted']}
SORT_ROLE = Qt.ItemDataRole.UserRole + 1


class SortItem(QTableWidgetItem):
    """Sorts by a hidden key (numbers sort numerically, not as text)."""

    def __lt__(self, other):
        a, b = self.data(SORT_ROLE), other.data(SORT_ROLE)
        try:
            return a < b
        except TypeError:
            return str(a) < str(b)


class ExplorerPage(Page):
    def __init__(self, win):
        super().__init__('Explorer', 'Open the 04e results ZIP (Drive → Robotekin2026/exports), then double-click an image to review it.')
        self.win = win
        self.header.add_action(ActionButton('Open ZIP', 'open_zip', parent=self)).clicked.connect(win.open_zip)
        self.header.add_action(ActionButton('Open folder', 'open_folder', parent=self)).clicked.connect(win.open_folder)
        groups = self.header.add_action(ActionButton('Groups…', None, parent=self))
        groups.setToolTip('Choose how unreviewed images get their default capture group (folder, or time blocks)')
        groups.clicked.connect(win.auto_group)
        self.export_button = self.header.add_action(ActionButton('Export dataset', 'export_dataset', role='primary', parent=self))
        self.export_button.clicked.connect(win.export_dataset)

        progress = Card('Review progress')
        row = QHBoxLayout()
        row.setSpacing(GAP)
        self.bar = QProgressBar()
        self.bar.setFormat('%v of %m reviewed')
        row.addWidget(self.bar, 1)
        self.accepted_chip = Chip('', 'ok')
        self.excluded_chip = Chip('', 'warn')
        self.unreviewed_chip = Chip('', '')
        for chip in (self.accepted_chip, self.excluded_chip, self.unreviewed_chip):
            row.addWidget(chip)
        progress.add_layout(row)
        self.source = QLabel('No bundle open.')
        self.source.setProperty('muted', True)
        self.source.setWordWrap(True)
        progress.add(self.source)
        self.root.addWidget(progress)

        filters = QHBoxLayout()
        filters.setSpacing(GAP)
        self.status_filter = QComboBox()
        self.status_filter.addItems(['All', 'Unreviewed', 'Accepted', 'Excluded'])
        self.status_filter.setMinimumWidth(150)
        self.status_filter.currentIndexChanged.connect(self.apply_filter)
        filters.addWidget(QLabel('Show'))
        filters.addWidget(self.status_filter)
        self.search = QLineEdit()
        self.search.setPlaceholderText('Filter by path, class or group…')
        self.search.setClearButtonEnabled(True)
        self.search.textChanged.connect(self.apply_filter)
        filters.addWidget(self.search, 1)
        self.shown = QLabel('')
        self.shown.setProperty('muted', True)
        filters.addWidget(self.shown)
        self.next_button = ActionButton('Review next unreviewed', None, role='primary', parent=self)
        self.next_button.clicked.connect(lambda: win.review.next_unreviewed())
        filters.addWidget(self.next_button)
        self.root.addLayout(filters)

        self.table = QTableWidget(0, len(COLUMNS))
        self.table.setHorizontalHeaderLabels(COLUMNS)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setAlternatingRowColors(True)
        self.table.setTextElideMode(Qt.TextElideMode.ElideMiddle)      # keep the file name visible in long paths
        self.table.setSortingEnabled(True)
        self.table.verticalHeader().hide()
        header = self.table.horizontalHeader()
        header.setDefaultAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        header.setSortIndicator(0, Qt.SortOrder.AscendingOrder)
        header.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.table.cellDoubleClicked.connect(lambda row, column: self.open_row(row))
        self.table.itemActivated.connect(lambda item: self.open_row(item.row()))
        self.root.addWidget(self.table, 1)
        self.clear()

    # -- data ---------------------------------------------------------------------------

    def clear(self):
        self.table.setRowCount(0)
        self.bar.setRange(0, 1)
        self.bar.setValue(0)
        for chip in (self.accepted_chip, self.excluded_chip, self.unreviewed_chip):
            chip.set('—', chip.property('kind') or '')
        self.shown.setText('')

    def row_values(self, image):
        bundle, project = self.win.bundle, self.win.project
        image_id = image['image_id']
        alternatives = bundle.alternatives[image_id]
        good = [r for r in alternatives if r['status'] == 'ok']
        decision = project.decision(image_id) or {}
        status = decision.get('status', 'unreviewed')
        saved = sorted({row['label'] for row in decision.get('instances', [])})
        suggestion = suggest_class(image['source_relative_path'], project.class_names())
        klass = ', '.join(saved) if saved else (f'{suggestion} (suggested)' if suggestion and status != 'accepted' else '')
        group = decision.get('group') or self.win.group_for(image)
        return [
            (image['source_relative_path'], image['source_relative_path'].casefold()),
            (f"{image.get('width', '?')} × {image.get('height', '?')}", image.get('width', 0) * image.get('height', 0)),
            (f'{len(good)}/{len(alternatives)}', len(good)),
            (', '.join(str(r.get('kept_masks', '?')) for r in good), sum(r.get('kept_masks', 0) for r in good)),
            (klass, klass.casefold()),
            (group, group.casefold()),
            (status.capitalize(), status),
        ], status

    def fill_row(self, row, image):
        values, status = self.row_values(image)
        for column, (text, key) in enumerate(values):
            item = self.table.item(row, column) or SortItem()
            item.setText(text)
            item.setData(Qt.ItemDataRole.UserRole, image['image_id'])
            item.setData(SORT_ROLE, key)
            if column == 6:
                item.setForeground(QBrush(QColor(STATUS_COLOR[status])))
            elif column == 4 and text.endswith('(suggested)'):
                item.setForeground(QBrush(QColor(PALETTE['muted'])))
            self.table.setItem(row, column, item)

    def refresh_all(self):
        bundle = self.win.bundle
        self.table.setSortingEnabled(False)
        self.table.setRowCount(len(bundle.images))
        for row, image in enumerate(bundle.images):
            self.fill_row(row, image)
        self.table.setSortingEnabled(True)
        self.source.setText(f'Source (read-only): {bundle.source}    ·    Review project: {self.win.project.folder}')
        self.update_stats()
        self.apply_filter()

    def refresh_row(self, image_id):
        for row in range(self.table.rowCount()):
            if self.table.item(row, 0).data(Qt.ItemDataRole.UserRole) == image_id:
                self.table.setSortingEnabled(False)
                self.fill_row(row, self.win.bundle.by_image[image_id])
                self.table.setSortingEnabled(True)
                break
        self.update_stats()
        self.apply_filter()

    def update_stats(self):
        stats = self.win.project.stats()
        reviewed = stats['accepted'] + stats['excluded']
        self.bar.setRange(0, max(1, stats['total']))
        self.bar.setValue(reviewed)
        self.accepted_chip.set(f"{stats['accepted']} accepted", 'ok')
        self.excluded_chip.set(f"{stats['excluded']} excluded", 'warn')
        self.unreviewed_chip.set(f"{stats['unreviewed']} to review", '')
        self.win.show_stats(stats)

    def apply_filter(self, *args):
        if self.win.bundle is None:
            return
        wanted = self.status_filter.currentText().casefold()
        text = self.search.text().casefold()
        visible = 0
        for row in range(self.table.rowCount()):
            status = self.table.item(row, 6).data(SORT_ROLE)
            haystack = ' '.join(self.table.item(row, c).text() for c in (0, 4, 5)).casefold()
            hide = (wanted != 'all' and status != wanted) or text not in haystack
            self.table.setRowHidden(row, hide)
            visible += not hide
        self.shown.setText(f'{visible} of {self.table.rowCount()} shown')
        if getattr(self.win, 'review', None) is not None:
            self.win.review.update_position()

    def visible_ids(self):
        """Image ids in the order the table currently shows them (filter and sort applied)."""
        return [self.table.item(row, 0).data(Qt.ItemDataRole.UserRole) for row in range(self.table.rowCount())
                if not self.table.isRowHidden(row)]

    def open_row(self, row):
        self.win.open_image(self.table.item(row, 0).data(Qt.ItemDataRole.UserRole))
