"""The working set of masks for one image, with undo/redo.

Instances are dicts ``{'id', 'label', 'mask' (bool HxW), 'keep'}`` as in ``core.Project``.
Commands hold references to those dicts (not indexes), so undo is exact after reordering.
Stroke edits store only the changed bounding box, which keeps 12 MP images cheap.
"""

import cv2
import numpy as np
from PySide6.QtCore import QObject, Signal, Slot
from PySide6.QtGui import QUndoCommand, QUndoStack


def mask_bbox(mask):
    """(x, y, w, h) of the non-zero area; w == 0 for an empty mask."""
    return cv2.boundingRect(np.ascontiguousarray(mask, np.uint8))


def diff_bbox(before, after):
    changed = before != after
    return mask_bbox(changed) if changed.any() else None


class _Command(QUndoCommand):
    def __init__(self, document, text, do, undo, region=None, applied=False):
        super().__init__(text)
        self.document, self._do, self._undo, self.region = document, do, undo, region
        self._skip_first = applied

    def redo(self):
        if self._skip_first:
            self._skip_first = False
        else:
            self._do()
        self.document.notify(self.region)

    def undo(self):
        self._undo()
        self.document.notify(self.region)


class ReviewDocument(QObject):
    changed = Signal(object)          # None = structure changed; (x0, y0, x1, y1) = pixel region only
    modifiedChanged = Signal(bool)

    def __init__(self):
        super().__init__()
        self.instances = []
        self.shape = (0, 0)
        self.stack = QUndoStack(self)
        self.stack.setUndoLimit(200)
        self.stack.cleanChanged.connect(self._clean_changed)

    @Slot(bool)
    def _clean_changed(self, clean):
        self.modifiedChanged.emit(not clean)

    @property
    def modified(self):
        return not self.stack.isClean()

    def load(self, shape, instances):
        self.shape = tuple(shape)
        self.instances = list(instances)
        self.stack.clear()
        self.stack.setClean()
        self.notify(None)

    def mark_clean(self):
        self.stack.setClean()

    def notify(self, region=None):
        self.changed.emit(region)

    def _push(self, text, do, undo, region=None, applied=False):
        self.stack.push(_Command(self, text, do, undo, region, applied))

    def _remove(self, insts):
        ids = {id(inst) for inst in insts}
        self.instances[:] = [inst for inst in self.instances if id(inst) not in ids]

    def index_of(self, inst):
        """Position by identity (dict equality would compare numpy arrays)."""
        return next(i for i, other in enumerate(self.instances) if other is inst)

    def _region(self, insts):
        """Union pixel box of `insts`, so repaint can stay local; None if all are empty."""
        region = None
        for inst in insts:
            x, y, w, h = mask_bbox(inst['mask'])
            if w:
                box = (x, y, x + w, y + h)
                region = box if region is None else (min(region[0], x), min(region[1], y),
                                                     max(region[2], x + w), max(region[3], y + h))
        return region

    # -- keep / class ------------------------------------------------------------

    def set_keep(self, changes):
        """changes: list of (instance, new_keep)."""
        changes = [(inst, keep) for inst, keep in changes if bool(inst.get('keep', True)) != keep]
        if not changes:
            return
        old = [(inst, bool(inst.get('keep', True))) for inst, _ in changes]
        def apply(pairs):
            for inst, keep in pairs:
                inst['keep'] = keep
        self._push('Keep / drop masks', lambda: apply(changes), lambda: apply(old),
                   self._region([inst for inst, _ in changes]))

    def set_label(self, insts, name):
        insts = [inst for inst in insts if inst['label'] != name]
        if not insts:
            return
        old = [(inst, inst['label']) for inst in insts]
        def apply(pairs):
            for inst, label in pairs:
                inst['label'] = label
        new = [(inst, name) for inst in insts]
        self._push('Assign class', lambda: apply(new), lambda: apply(old), self._region(insts))

    # -- structure ---------------------------------------------------------------

    def add(self, insts, text='Add masks'):
        insts = list(insts)
        if not insts:
            return
        def do():
            self.instances.extend(insts)
        self._push(text, do, lambda: self._remove(insts))

    def new_empty(self, label):
        inst = {'id': None, 'label': label, 'mask': np.zeros(self.shape, bool), 'keep': True}
        self.add([inst], 'New empty mask')
        return inst

    def delete(self, insts):
        ids = {id(inst) for inst in insts}
        positions = [(i, inst) for i, inst in enumerate(self.instances) if id(inst) in ids]
        if not positions:
            return
        def undo():
            for index, inst in positions:
                self.instances.insert(index, inst)
        self._push('Delete masks', lambda: self._remove([inst for _, inst in positions]), undo)

    def duplicate(self, inst):
        copy = {**inst, 'id': None, 'mask': inst['mask'].copy(), 'keep': True}
        index = self.index_of(inst) + 1
        self._push('Duplicate mask', lambda: self.instances.insert(index, copy), lambda: self._remove([copy]))
        return copy

    def merge(self, insts):
        """OR the masks of `insts` into one instance at the first one's position."""
        order = [inst for inst in self.instances if any(inst is other for other in insts)]
        if len(order) < 2:
            return None
        merged_mask = np.zeros(self.shape, bool)
        for inst in order:
            merged_mask |= inst['mask']
        merged = {'id': None, 'label': order[0]['label'], 'mask': merged_mask, 'keep': True}
        positions = [(self.index_of(inst), inst) for inst in order]
        def do():
            self._remove(order)
            self.instances.insert(positions[0][0], merged)
        def undo():
            self._remove([merged])
            for index, inst in positions:
                self.instances.insert(index, inst)
        self._push('Merge masks', do, undo)
        return merged

    def split(self, inst):
        """Replace a mask by one mask per connected piece. Returns the number of pieces."""
        count, labels = cv2.connectedComponents(np.ascontiguousarray(inst['mask'], np.uint8), connectivity=8)
        if count <= 2:
            return count - 1
        pieces = [{'id': None, 'label': inst['label'], 'mask': labels == n, 'keep': inst.get('keep', True)}
                  for n in range(1, count)]
        index = self.index_of(inst)
        def do():
            self._remove([inst])
            self.instances[index:index] = pieces
        def undo():
            self._remove(pieces)
            self.instances.insert(index, inst)
        self._push('Split mask', do, undo)
        return len(pieces)

    # -- pixel edits ---------------------------------------------------------------

    def edit_region(self, inst, bbox, before, after, text='Edit mask', applied=True):
        """Record a change of inst['mask'][y:y+h, x:x+w] from `before` to `after`."""
        x, y, w, h = bbox
        mask = inst['mask']
        def do():
            mask[y:y + h, x:x + w] = after
        def undo():
            mask[y:y + h, x:x + w] = before
        self._push(text, do, undo, (x, y, x + w, y + h), applied)

    def _edit_with(self, inst, text, transform):
        x, y, w, h = mask_bbox(inst['mask'])
        if not w:
            return False
        # one pixel of margin so morphology at the border stays inside the crop
        x0, y0 = max(0, x - 1), max(0, y - 1)
        x1, y1 = min(self.shape[1], x + w + 1), min(self.shape[0], y + h + 1)
        before = inst['mask'][y0:y1, x0:x1].copy()
        after = transform(before)
        if np.array_equal(before, after):
            return False
        self.edit_region(inst, (x0, y0, x1 - x0, y1 - y0), before, after, text, applied=False)
        return True

    def fill_holes(self, inst):
        def transform(crop):
            contours, _ = cv2.findContours(crop.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            filled = np.zeros(crop.shape, np.uint8)
            cv2.drawContours(filled, contours, -1, 1, cv2.FILLED)
            return filled > 0
        return self._edit_with(inst, 'Fill holes', transform)

    def remove_specks(self, inst, min_pixels):
        def transform(crop):
            count, labels, stats, _ = cv2.connectedComponentsWithStats(crop.astype(np.uint8), connectivity=8)
            keep = np.zeros(count, bool)
            keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= min_pixels
            return keep[labels]
        return self._edit_with(inst, 'Remove specks', transform)

    # -- undo / redo -----------------------------------------------------------------

    def undo(self):
        self.stack.undo()

    def redo(self):
        self.stack.redo()
