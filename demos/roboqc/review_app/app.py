"""Five-page Qt workstation: Explorer, Review, Augmentation, Training, Validation.

Bundle code and model runtime stay isolated: the 04e archive is only read, never executed,
and models run in a separate process that the user starts explicitly.
"""

from pathlib import Path
import sys
import uuid

from PySide6.QtCore import QObject, QSettings, QThread, QTimer, Signal, Slot
from PySide6.QtWidgets import (
    QApplication, QFileDialog, QLabel, QMainWindow, QMessageBox, QProgressBar, QTabWidget,
)

from .core import Bundle, Project, export_dataset, zip_dataset
from .dialogs import AutoGroupDialog, ExportDialog, ShortcutsDialog, ask_unsaved
from .page_augment import AugmentPage
from .page_explorer import ExplorerPage
from .page_jobs import JobRunner, TrainingPage, ValidationPage
from .page_review import ReviewPage
from .shortcuts import ShortcutManager
from .suggest import auto_groups, folder_of, seed_classes
from .theme import DEFAULT_WINDOW, MIN_WINDOW, apply_theme

TABS = ['1  Explorer', '2  Review', '3  Augmentation', '4  Training', '5  Validation']
SCOPES = ['other', 'review', 'augment', 'other', 'other']
TITLE = 'Robotekin | Segmentation Review & Training'


class Task(QObject):
    done = Signal(object)
    failed = Signal(str)
    progress = Signal(int, int)

    def __init__(self, function):
        super().__init__()
        self.function = function

    def run(self):
        try:
            self.done.emit(self.function(self.progress.emit))
        except Exception as exc:
            self.failed.emit(f'{type(exc).__name__}: {exc}')


class Window(QMainWindow):
    def __init__(self):
        super().__init__()
        application = QApplication.instance()
        if not application.styleSheet():
            apply_theme(application)
        self.setWindowTitle(TITLE)
        self.resize(*DEFAULT_WINDOW)
        self.setMinimumSize(*MIN_WINDOW)
        self.bundle = self.project = None
        self.group_map = {}
        self.task_thread = self.task = None
        self.on_done = self.on_failed = None
        self._after_task = []
        self.manager = ShortcutManager(self)
        self.job = JobRunner(self)
        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)
        self.setCentralWidget(self.tabs)
        self.explorer = ExplorerPage(self)
        self.review = ReviewPage(self)
        self.augment = AugmentPage(self)
        self.training = TrainingPage(self)
        self.validation = ValidationPage(self)
        for page, title in zip((self.explorer, self.review, self.augment, self.training, self.validation), TABS):
            self.tabs.addTab(page, title)
            page.header.helpRequested.connect(self.show_shortcuts)
        self.tabs.currentChanged.connect(self.on_tab_changed)
        self.busy = QProgressBar()
        self.busy.setMaximumWidth(260)
        self.busy.hide()
        self.stats_label = QLabel('')
        self.statusBar().addWidget(self.stats_label)
        self.statusBar().addPermanentWidget(self.busy)
        self.bind_shortcuts()
        self.build_menus()
        self.on_tab_changed(0)

    # -- shortcuts and menus ------------------------------------------------------------

    def bind_shortcuts(self):
        review, bind = self.review, self.manager.bind
        canvas = review.canvas
        for sid, callback in {
            'prev_image': lambda: review.go(-1), 'next_image': lambda: review.go(1),
            'next_unreviewed': review.next_unreviewed, 'prev_method': lambda: review.step_method(-1),
            'next_method': lambda: review.step_method(1), 'merge_mode': review.toggle_merge_mode,
            'accept_next': review.accept_next, 'accept': review.accept, 'exclude': review.exclude,
            'undo': review.doc.undo, 'redo': review.doc.redo,
            'tool_select': lambda: review.set_tool('select'), 'tool_pan': lambda: review.set_tool('pan'),
            'tool_brush': lambda: review.set_tool('brush'), 'tool_erase': lambda: review.set_tool('erase'),
            'tool_lasso': lambda: review.set_tool('lasso'),
            'brush_smaller': lambda: review.step_brush(-1), 'brush_larger': lambda: review.step_brush(1),
            'toggle_masks': review.toggle_masks, 'cycle_opacity': canvas.cycle_opacity,
            'zoom_fit': canvas.fit, 'zoom_100': canvas.zoom_100,
            'zoom_in': lambda: canvas.zoom_by(1.25), 'zoom_out': lambda: canvas.zoom_by(1 / 1.25),
            'next_instance': lambda: review.cycle_instance(1), 'prev_instance': lambda: review.cycle_instance(-1),
            'toggle_keep': review.toggle_keep, 'keep_only': review.keep_only,
            'delete_instance': review.delete_selected, 'new_instance': review.new_instance,
            'duplicate_instance': review.duplicate_selected, 'merge_instances': review.merge_selected,
            'split_instance': review.split_selected, 'check_all': review.check_all,
            'check_none': review.check_none, 'check_invert': review.check_invert,
            'manage_classes': review.manage_classes, 'reroll_preview': self.augment.reroll,
            'open_zip': self.open_zip, 'open_folder': self.open_folder, 'export_dataset': self.export_dataset,
            'show_shortcuts': self.show_shortcuts,
        }.items():
            bind(sid, callback)
        for number in range(1, 10):
            bind(f'class_{number}', lambda n=number: review.assign_class_number(n))
        for number in range(1, 6):
            bind(f'page_{number}', lambda n=number: self.go(n - 1))

    def build_menus(self):
        actions = self.manager.actions
        bar = self.menuBar()
        for title, ids in [
            ('&File', ['open_zip', 'open_folder', None, 'export_dataset']),
            ('&Edit', ['undo', 'redo', None, 'new_instance', 'duplicate_instance', 'merge_instances', 'split_instance',
                       'delete_instance', None, 'manage_classes']),
            ('&Masks', ['check_all', 'check_none', 'check_invert', None, 'toggle_keep', 'keep_only', None,
                        'next_instance', 'prev_instance']),
            ('&Tools', ['tool_select', 'tool_pan', 'tool_brush', 'tool_erase', 'tool_lasso', None,
                        'brush_smaller', 'brush_larger']),
            ('&View', ['zoom_fit', 'zoom_100', 'zoom_in', 'zoom_out', None, 'toggle_masks', 'cycle_opacity']),
            ('&Go', ['prev_image', 'next_image', 'next_unreviewed', None, 'prev_method', 'next_method', 'merge_mode',
                     None, 'page_1', 'page_2', 'page_3', 'page_4', 'page_5']),
            ('&Help', ['show_shortcuts']),
        ]:
            menu = bar.addMenu(title)
            for sid in ids:
                menu.addSeparator() if sid is None else menu.addAction(actions[sid])

    def show_shortcuts(self):
        ShortcutsDialog(self).exec()

    def on_tab_changed(self, index):
        self.manager.set_scope(SCOPES[index])
        if index == 2 and self.project is not None:
            self.augment.refresh_sources()

    # -- background work ----------------------------------------------------------------

    def background(self, function, finished, failed=None):
        if self.task_thread is not None:
            self.statusBar().showMessage('Busy: wait for the current operation to finish.', 4000)
            return False
        thread = QThread(self)
        task = Task(function)
        task.moveToThread(thread)
        # Plain callables connected to a worker-thread signal run in that thread, so
        # deliver results through Window slots, which Qt queues onto the GUI thread.
        self.on_done, self.on_failed = finished, failed or self.error
        thread.started.connect(task.run)
        task.progress.connect(self.progress)
        task.done.connect(self.task_done)
        task.failed.connect(self.task_failed)
        task.done.connect(thread.quit)
        task.failed.connect(thread.quit)
        task.done.connect(task.deleteLater)
        task.failed.connect(task.deleteLater)
        thread.finished.connect(self.task_finished)
        thread.finished.connect(thread.deleteLater)
        self.task_thread, self.task = thread, task
        self.tabs.setEnabled(False)
        self.manager.set_busy(True)
        self.busy.setRange(0, 0)
        self.busy.show()
        thread.start()
        return True

    @Slot(object)
    def task_done(self, result):
        self.on_done(result)

    @Slot(str)
    def task_failed(self, message):
        self.on_failed(message)

    def task_finished(self):
        self.task_thread = self.task = None
        self.tabs.setEnabled(True)
        self.manager.set_busy(False)
        self.busy.hide()
        while self._after_task:
            QTimer.singleShot(0, self._after_task.pop(0))

    def after_task(self, callback):
        """Run `callback` once the current background task has fully finished."""
        self._after_task.append(callback)

    def progress(self, done, total):
        self.busy.setRange(0, total)
        self.busy.setValue(done)

    def error(self, message):
        QMessageBox.warning(self, 'Review workstation', str(message))

    def info(self, title, message):
        QMessageBox.information(self, title, message)

    def status(self, text, milliseconds=8000):
        self.statusBar().showMessage(str(text), milliseconds)

    # -- project state ------------------------------------------------------------------

    @property
    def dirty(self):
        return self.review.dirty

    @dirty.setter
    def dirty(self, value):
        if not value:
            self.review.discard_edits()

    def update_title(self):
        name = f' — {self.project.folder.name}' if self.project else ''
        self.setWindowTitle(f"{'* ' if self.dirty else ''}{TITLE}{name}")

    def show_stats(self, stats):
        reviewed = stats['accepted'] + stats['excluded']
        self.stats_label.setText(f"Reviewed {reviewed}/{stats['total']}  ·  Accepted {stats['accepted']}  ·  "
                                 f"Excluded {stats['excluded']}")

    def go(self, index):
        self.tabs.setCurrentIndex(index)

    def open_image(self, image_id):
        self.review.open_image(image_id)

    def after_save(self, image_id):
        self.explorer.refresh_row(image_id)

    def refresh_all(self):
        if self.bundle:
            self.explorer.refresh_all()

    def group_for(self, image):
        return self.group_map.get(image['image_id'], folder_of(image['source_relative_path']))

    def open_zip(self):
        self.review.guard(lambda: self.choose_and_load(False))

    def open_folder(self):
        self.review.guard(lambda: self.choose_and_load(True))

    def choose_and_load(self, folder):
        settings = QSettings('Robotekin', 'RoboQC')
        if folder:
            source = QFileDialog.getExistingDirectory(self, '04e bundle folder', settings.value('source', ''))
        else:
            source, _ = QFileDialog.getOpenFileName(self, '04e results ZIP', settings.value('source', ''), 'ZIP (*.zip)')
        if not source:
            return
        project_folder = QFileDialog.getExistingDirectory(
            self, 'Choose a separate review project folder (empty, or an existing project)', settings.value('project', ''))
        if not project_folder:
            return
        settings.setValue('source', str(Path(source).parent if not folder else source))
        settings.setValue('project', project_folder)
        self.load_bundle(source, project_folder)

    def load_bundle(self, source, project_folder):
        def load(progress):
            bundle = Bundle(source)
            try:
                project = Project(project_folder, bundle)
            except Exception:
                bundle.close()
                raise
            return bundle, project
        self.background(load, self.bundle_loaded)

    def bundle_loaded(self, result):
        if self.bundle:
            self.bundle.close()
        self.bundle, self.project = result
        for name in seed_classes(self.bundle.images):
            self.project.ensure_class(name)
        self.group_map = auto_groups(self.bundle.images, self.project.pref('group_window_s', 0))
        self.review.clear()
        self.explorer.refresh_all()
        self.augment.load_project()
        self.training.refresh_summary()
        self.update_title()
        self.go(0)

    def auto_group(self):
        if self.bundle is None:
            return
        dialog = AutoGroupDialog(self.bundle.images, self.project.pref('group_window_s', 0), self)
        if dialog.exec():
            seconds = dialog.seconds.value()
            self.project.set_pref('group_window_s', seconds)
            self.group_map = auto_groups(self.bundle.images, seconds)
            self.explorer.refresh_all()
            self.status('Default groups updated. Accepted images keep the group they were saved with.')

    def export_dataset(self):
        if not self.project:
            self.error('Open a bundle first')
            return
        if self.dirty:
            self.error('Accept or discard the current mask edits before exporting')
            return
        dialog = ExportDialog(self)
        if not dialog.exec():
            return
        output = Path(dialog.parent_folder.text()) / ('reviewed_dataset_' + uuid.uuid4().hex[:12])
        allow_holes, make_zip = dialog.holes_check.isChecked(), dialog.zip_check.isChecked()
        project = self.project
        def save(progress):
            export_dataset(project, output, allow_holes=allow_holes, progress=progress)
            if make_zip:
                zip_dataset(output, output.with_suffix('.zip'), progress)
            return output
        def done(path):
            self.training.dataset.setText(str(path))
            self.validation.dataset.setText(str(path))
            self.augment.export_dataset.setText(str(path))
            self.info('Dataset exported', f'{path}\n' + ('ZIP alongside the folder.' if make_zip else 'Ready for training.'))
        self.background(save, done)

    def closeEvent(self, event):
        if self.task_thread or self.job.running:
            self.error('Wait for the file operation to finish or stop the model job before closing')
            event.ignore()
            return
        if self.dirty:
            choice = ask_unsaved(self)
            if choice != 'discard':
                if choice == 'accept':
                    self.review.save_review('accepted', then=self.close)
                event.ignore()
                return
        if self.bundle:
            self.bundle.close()
        event.accept()


def main():
    application = QApplication(sys.argv)
    apply_theme(application)
    window = Window()
    window.show()
    sys.exit(application.exec())
