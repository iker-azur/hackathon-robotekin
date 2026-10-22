"""Zoomable image canvas with a class-coloured mask overlay and edit tools.

The base image and the mask overlay are two layers that draw straight from numpy buffers,
so a brush stroke recomposes only the rectangle it touched (cheap even on 12 MP photos).
Edits go through ``ReviewDocument`` so every stroke is undoable.
"""

import math

import cv2
import numpy as np
from PySide6.QtCore import QRectF, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QCursor, QImage, QPainter, QPen, QPolygonF
from PySide6.QtWidgets import (
    QGraphicsEllipseItem, QGraphicsItem, QGraphicsPolygonItem, QGraphicsScene, QGraphicsView, QFrame,
)

from .document import mask_bbox
from .theme import hex_to_rgb

TOOLS = ('select', 'pan', 'brush', 'erase', 'lasso')
OPACITIES = (.22, .38, .60)
_K3 = np.ones((3, 3), np.uint8)
MIN_SCALE, MAX_SCALE = .02, 64.


class _Layer(QGraphicsItem):
    """Draws only the exposed part of a QImage (no QPixmap copy of the whole frame)."""

    def __init__(self, image, z, buffer):
        super().__init__()
        self.image = image
        self.buffer = buffer          # a QImage does not keep its numpy memory alive
        self.setZValue(z)

    def boundingRect(self):
        return QRectF(0, 0, self.image.width(), self.image.height())

    def paint(self, painter, option, widget=None):
        rect = option.exposedRect.toAlignedRect().intersected(self.image.rect())
        if rect.isEmpty():
            return
        scale = abs(painter.worldTransform().m11())
        if scale < .7:
            # Zoomed out: shrink just the exposed part with an area filter. Qt's smooth scaling of a whole
            # 12 MP layer costs about a second per repaint; this costs a few tens of milliseconds.
            x, y, w, h = rect.x(), rect.y(), rect.width(), rect.height()
            size = (max(1, round(w * scale)), max(1, round(h * scale)))
            small = cv2.resize(self.buffer[y:y + h, x:x + w], size, interpolation=cv2.INTER_AREA)
            small = np.ascontiguousarray(small)
            painter.drawImage(QRectF(x, y, w, h), QImage(small.data, size[0], size[1], small.strides[0],
                                                         self.image.format()))
        else:
            painter.drawImage(rect, self.image, rect)


def _union(a, b):
    if a is None:
        return b
    return (min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3]))


class Canvas(QGraphicsView):
    selectionChanged = Signal()
    keepToggled = Signal(object)          # double-click on a mask
    strokeFinished = Signal(object)       # instance whose pixels changed
    hoverChanged = Signal(str)
    zoomChanged = Signal(float)

    def __init__(self):
        super().__init__()
        self.setScene(QGraphicsScene(self))
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setRenderHints(QPainter.RenderHint.SmoothPixmapTransform | QPainter.RenderHint.Antialiasing)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self.setViewportUpdateMode(QGraphicsView.ViewportUpdateMode.MinimalViewportUpdate)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.ClickFocus)
        self.setMinimumSize(400, 240)
        self.doc = None
        self.rgb = None
        self.overlay = None
        self.base_item = self.overlay_item = None
        self.color_for = lambda label: '#2563EB'
        self.new_label = lambda: 'part'
        self.tool = 'select'
        self.radius = 12
        self.opacity = OPACITIES[1]
        self.show_masks = True
        self.selection = []
        self.active = None
        self.hover = None
        self.fit_mode = True
        self._ow = 1
        self._bboxes = {}
        self._panning = False
        self._middle = None
        self._stroke = None
        self._lasso = []
        self._ring = QGraphicsEllipseItem()
        self._ring.setPen(self._cosmetic_pen('#111827', 1.5))
        self._ring.setBrush(QColor(255, 255, 255, 40))
        self._ring.setZValue(20)
        self._ring.hide()
        self._lasso_item = QGraphicsPolygonItem()
        pen = self._cosmetic_pen('#2563EB', 2)
        pen.setStyle(Qt.PenStyle.DashLine)
        self._lasso_item.setPen(pen)
        self._lasso_item.setBrush(QColor(37, 99, 235, 40))
        self._lasso_item.setZValue(21)
        self._lasso_item.hide()
        self.scene().addItem(self._ring)
        self.scene().addItem(self._lasso_item)
        self._rescale_timer = QTimer(self)
        self._rescale_timer.setSingleShot(True)
        self._rescale_timer.timeout.connect(self._rescale_outline)

    @staticmethod
    def _cosmetic_pen(color, width):
        pen = QPen(QColor(color), width)
        pen.setCosmetic(True)
        return pen

    # -- document -------------------------------------------------------------------

    def display(self, rgb, doc, color_for=None):
        if self.doc is not None:
            try:
                self.doc.changed.disconnect(self._on_doc_changed)
            except (RuntimeError, TypeError):
                pass
        self.doc = doc
        if color_for:
            self.color_for = color_for
        self.rgb = np.ascontiguousarray(rgb, np.uint8)
        h, w = self.rgb.shape[:2]
        self.overlay = np.zeros((h, w, 4), np.uint8)
        self._drop_layers()
        self._ring.hide()
        self._lasso_item.hide()
        base = QImage(self.rgb.data, w, h, w * 3, QImage.Format.Format_RGB888)
        over = QImage(self.overlay.data, w, h, w * 4, QImage.Format.Format_RGBA8888_Premultiplied)
        self.base_item, self.overlay_item = _Layer(base, 0, self.rgb), _Layer(over, 1, self.overlay)
        for item in (self.base_item, self.overlay_item):
            self.scene().addItem(item)
        self.scene().setSceneRect(0, 0, w, h)
        self._bboxes.clear()
        self.selection, self.active, self.hover = [], None, None
        self._lasso = []
        doc.changed.connect(self._on_doc_changed)
        self.fit()
        self._ow = self._outline_width()
        self._compose()

    def clear(self):
        if self.doc is not None:
            try:
                self.doc.changed.disconnect(self._on_doc_changed)
            except (RuntimeError, TypeError):
                pass
        self.doc = self.rgb = self.overlay = None
        self._drop_layers()
        self._ring.hide()
        self._lasso_item.hide()
        self.selection, self.active, self.hover = [], None, None

    def _drop_layers(self):
        """Remove the per-image layers; the brush ring and lasso preview live as long as the canvas."""
        for item in (self.base_item, self.overlay_item):
            if item is not None:
                self.scene().removeItem(item)
        self.base_item = self.overlay_item = None

    def _on_doc_changed(self, region):
        if region is None:
            self._bboxes.clear()
            live = {id(i) for i in self.doc.instances}
            self.selection = [i for i in self.selection if id(i) in live]
            if self.active is not None and id(self.active) not in live:
                self.active = self.selection[-1] if self.selection else None
            if self.hover is not None and id(self.hover) not in live:
                self.hover = None
            self._compose()
            self.selectionChanged.emit()
        else:
            # undo / redo of a pixel edit: grow the cached boxes of masks that now touch the region
            x0, y0, x1, y1 = region
            for inst in self.doc.instances:
                cached = self._bboxes.get(id(inst))
                if cached is not None and inst['mask'][y0:y1, x0:x1].any():
                    self._bboxes[id(inst)] = _union(cached, region) if cached[2] > cached[0] else tuple(region)
            self._compose(self._expand(region))

    # -- geometry helpers -------------------------------------------------------------

    def _outline_width(self):
        scale = abs(self.transform().m11()) or 1.
        return int(max(1, min(10, round(2. / scale))))

    def _expand(self, region, pad=None):
        pad = (2 * self._ow + 3) if pad is None else pad
        h, w = self.rgb.shape[:2]
        return (max(0, region[0] - pad), max(0, region[1] - pad), min(w, region[2] + pad), min(h, region[3] + pad))

    def _bbox_of(self, inst):
        """(x0, y0, x1, y1) of the mask; cached, and may be a superset after edits."""
        key = id(inst)
        cached = self._bboxes.get(key)
        if cached is None:
            x, y, w, h = mask_bbox(inst['mask'])
            cached = self._bboxes[key] = (x, y, x + w, y + h) if w else (0, 0, 0, 0)
        return cached

    def _region_of(self, insts):
        region = None
        for inst in insts:
            x0, y0, x1, y1 = self._bbox_of(inst)
            if x1 > x0:
                region = _union(region, (x0, y0, x1, y1))
        return self._expand(region) if region else None

    # -- compositing ------------------------------------------------------------------

    @staticmethod
    def _blend(tile, mask, color, alpha):
        """Premultiplied 'over' of one translucent colour onto tile[mask]."""
        if not mask.any():
            return
        # tile*(1-alpha) + colour*alpha, applied only where `mask`, as whole-array OpenCV operations;
        # boolean-index gather/scatter is several times slower on masks that cover millions of pixels
        base = np.ascontiguousarray(tile)
        blended = cv2.add(cv2.convertScaleAbs(base, alpha=1. - alpha),
                          (color[0] * alpha, color[1] * alpha, color[2] * alpha, 255 * alpha))
        tile[...] = cv2.copyTo(blended, np.ascontiguousarray(mask, np.uint8), base)

    def _edge(self, inst, rect, thickness):
        """Boolean outline band (inside the mask) for `rect`, computed on a padded crop."""
        h, w = inst['mask'].shape
        pad = thickness + 2
        ex0, ey0 = max(0, rect[0] - pad), max(0, rect[1] - pad)
        ex1, ey1 = min(w, rect[2] + pad), min(h, rect[3] + pad)
        crop = inst['mask'][ey0:ey1, ex0:ex1].astype(np.uint8)
        # one erosion with a square kernel (separable, so its cost does not grow with the thickness)
        eroded = cv2.erode(crop, np.ones((2 * thickness + 1,) * 2, np.uint8),
                           borderType=cv2.BORDER_CONSTANT, borderValue=0)
        edge = (crop > 0) & (eroded == 0)
        return edge[rect[1] - ey0:rect[3] - ey0, rect[0] - ex0:rect[2] - ex0]

    def _compose(self, region=None):
        if self.rgb is None or self.doc is None:
            return
        h, w = self.rgb.shape[:2]
        if region is None:
            x0, y0, x1, y1 = 0, 0, w, h
        else:
            x0, y0, x1, y1 = max(0, region[0]), max(0, region[1]), min(w, region[2]), min(h, region[3])
        if x1 <= x0 or y1 <= y0:
            return
        self.overlay[y0:y1, x0:x1] = 0
        if self.show_masks:
            selected_ids = {id(i) for i in self.selection}
            visible = []
            for inst in self.doc.instances:
                bx0, by0, bx1, by1 = self._bbox_of(inst)
                rect = (max(x0, bx0), max(y0, by0), min(x1, bx1), min(y1, by1))
                if rect[2] <= rect[0] or rect[3] <= rect[1]:
                    continue
                color = hex_to_rgb(self.color_for(inst['label']))
                keep = inst.get('keep', True)
                chosen = id(inst) in selected_ids
                if keep:
                    alpha = min(.85, self.opacity + (.18 if chosen or inst is self.hover else 0.))
                    self._blend(self.overlay[rect[1]:rect[3], rect[0]:rect[2]],
                                inst['mask'][rect[1]:rect[3], rect[0]:rect[2]], color, alpha)
                visible.append((inst, rect, color, keep, chosen))
            for inst, rect, color, keep, chosen in visible:
                tile = self.overlay[rect[1]:rect[3], rect[0]:rect[2]]
                if not keep:
                    tile[self._edge(inst, rect, self._ow)] = (107 * .6, 114 * .6, 128 * .6, 255 * .6)
                    continue
                if chosen:
                    tile[self._edge(inst, rect, 2 * self._ow)] = (255, 255, 255, 255)
                tile[self._edge(inst, rect, self._ow)] = (*color, 255)
        self.overlay_item.update(QRectF(x0, y0, x1 - x0, y1 - y0))

    def refresh(self, insts=None):
        """Recompose after a selection / hover change (only the affected area)."""
        if insts is None:
            self._compose()
            return
        region = self._region_of(insts)
        if region is not None:
            self._compose(region)

    # -- state setters ----------------------------------------------------------------

    def set_tool(self, tool):
        self._finish_lasso(cancel=True)
        self.tool = tool
        self._apply_cursor()

    def set_radius(self, radius):
        self.radius = int(max(1, min(200, radius)))
        self._ring_update()

    def cycle_opacity(self):
        index = (OPACITIES.index(self.opacity) + 1) % len(OPACITIES) if self.opacity in OPACITIES else 0
        self.opacity = OPACITIES[index]
        self._compose()

    def toggle_masks(self):
        self.show_masks = not self.show_masks
        self._compose()
        return self.show_masks

    def set_selection(self, insts, active=None, notify=True):
        old = list(self.selection)
        self.selection = list(insts)
        self.active = active if active is not None else (self.selection[-1] if self.selection else None)
        self.refresh(old + self.selection)
        if notify:
            self.selectionChanged.emit()

    def set_temp_pan(self, on):
        self._panning = on
        self._apply_cursor()

    def _is_pan(self):
        return self.tool == 'pan' or self._panning

    def _apply_cursor(self):
        if self._is_pan():
            self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
            self.viewport().setCursor(Qt.CursorShape.OpenHandCursor)
        else:
            self.setDragMode(QGraphicsView.DragMode.NoDrag)
            shape = {'select': Qt.CursorShape.ArrowCursor, 'lasso': Qt.CursorShape.CrossCursor}.get(
                self.tool, Qt.CursorShape.BlankCursor if self.tool in ('brush', 'erase') else Qt.CursorShape.ArrowCursor)
            self.viewport().setCursor(QCursor(shape))
        if self.tool not in ('brush', 'erase') or self._is_pan():
            self._ring.hide()

    # -- zoom -------------------------------------------------------------------------

    def fit(self):
        if self.rgb is None:
            return
        self.fit_mode = True
        if self.viewport().width() < 32 or self.viewport().height() < 32:
            return                      # not laid out yet; resizeEvent fits once it has a size
        self.fitInView(self.scene().sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)
        self._after_zoom()

    def zoom_100(self):
        self.resetTransform()
        self.fit_mode = False
        self._after_zoom()

    def zoom_by(self, factor):
        scale = abs(self.transform().m11())
        factor = max(MIN_SCALE / scale, min(MAX_SCALE / scale, factor))
        self.scale(factor, factor)
        self.fit_mode = False
        self._after_zoom()

    def _after_zoom(self):
        self.zoomChanged.emit(abs(self.transform().m11()))
        self._rescale_timer.start(120)
        self._ring_update()

    def _rescale_outline(self):
        width = self._outline_width()
        if width != self._ow:
            self._ow = width
            self._compose()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.fit_mode and self.rgb is not None and self.viewport().width() >= 32 and self.viewport().height() >= 32:
            self.fitInView(self.scene().sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)
            self._after_zoom()

    def wheelEvent(self, event):
        self.zoom_by(1.15 ** (event.angleDelta().y() / 120.))

    # -- picking / hover --------------------------------------------------------------

    def _scene_point(self, event):
        return self.mapToScene(event.position().toPoint())

    def pick(self, point):
        """Topmost mask under `point`: kept masks first, then dropped ones."""
        if self.doc is None or self.rgb is None:
            return None
        x, y = math.floor(point.x()), math.floor(point.y())
        h, w = self.rgb.shape[:2]
        if not (0 <= x < w and 0 <= y < h):
            return None
        for want_keep in (True, False):
            for inst in reversed(self.doc.instances):
                if bool(inst.get('keep', True)) != want_keep:
                    continue
                x0, y0, x1, y1 = self._bbox_of(inst)
                if x0 <= x < x1 and y0 <= y < y1 and inst['mask'][y, x]:
                    return inst
        return None

    def _set_hover(self, inst):
        if inst is self.hover:
            return
        old, self.hover = self.hover, inst
        self.refresh([i for i in (old, inst) if i is not None and any(i is j for j in self.doc.instances)])
        if inst is None:
            self.hoverChanged.emit('')
        else:
            index = self.doc.index_of(inst) + 1
            self.hoverChanged.emit(f"Mask {index}: {inst['label']} · {int(inst['mask'].sum()):,} px"
                                   + ('' if inst.get('keep', True) else ' (dropped, double-click to keep)'))

    # -- mouse ------------------------------------------------------------------------

    def _ring_update(self, point=None):
        if self.tool not in ('brush', 'erase') or self._is_pan() or self.rgb is None:
            self._ring.hide()
            return
        if point is not None:
            self._ring.setPos(point)
        r = self.radius
        self._ring.setRect(-r, -r, 2 * r, 2 * r)
        self._ring.show()

    def mousePressEvent(self, event):
        if self.rgb is None:
            return super().mousePressEvent(event)
        self.setFocus()
        if event.button() == Qt.MouseButton.MiddleButton:
            self._middle = event.position()
            self.viewport().setCursor(Qt.CursorShape.ClosedHandCursor)
            return
        if self._is_pan() or event.button() != Qt.MouseButton.LeftButton:
            self.fit_mode = False
            return super().mousePressEvent(event)
        point = self._scene_point(event)
        if self.tool == 'select':
            inst = self.pick(point)
            ctrl = bool(event.modifiers() & Qt.KeyboardModifier.ControlModifier)
            if inst is None:
                if not ctrl:
                    self.set_selection([])
            elif ctrl:
                chosen = [i for i in self.selection if i is not inst]
                self.set_selection(chosen if len(chosen) != len(self.selection) else chosen + [inst], inst)
            else:
                self.set_selection([inst], inst)
        elif self.tool in ('brush', 'erase'):
            self._begin_stroke(point)
        elif self.tool == 'lasso':
            self._lasso.append(point)
            self._lasso_update()

    def mouseMoveEvent(self, event):
        if self.rgb is None:
            return super().mouseMoveEvent(event)
        if self._middle is not None:
            delta = event.position() - self._middle
            self._middle = event.position()
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - int(delta.x()))
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - int(delta.y()))
            self.fit_mode = False
            return
        point = self._scene_point(event)
        if self.tool in ('brush', 'erase') and not self._is_pan():
            self._ring_update(point)
            if self._stroke is not None and event.buttons() & Qt.MouseButton.LeftButton:
                self._paint_to(point)
        elif self.tool == 'lasso' and self._lasso:
            self._lasso_update(point)
        elif self.tool == 'select':
            self._set_hover(self.pick(point))
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.MiddleButton:
            self._middle = None
            self._apply_cursor()
            return
        if self._stroke is not None:
            self._end_stroke()
        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event):
        if self.rgb is None or self._is_pan():
            return super().mouseDoubleClickEvent(event)
        if self.tool == 'select':
            inst = self.pick(self._scene_point(event))
            if inst is not None:
                self.keepToggled.emit(inst)
        elif self.tool == 'lasso':
            if self._lasso:
                self._lasso.pop()        # the second click of the double click
            self._finish_lasso(subtract=bool(event.modifiers() & Qt.KeyboardModifier.ShiftModifier))

    def leaveEvent(self, event):
        self._ring.hide()
        if self.tool == 'select' and self.rgb is not None:
            self._set_hover(None)
        super().leaveEvent(event)

    def keyReleaseEvent(self, event):
        if event.key() == Qt.Key.Key_Space and not event.isAutoRepeat():
            self.set_temp_pan(False)
            return
        super().keyReleaseEvent(event)

    def focusOutEvent(self, event):
        self.set_temp_pan(False)               # never leave the view stuck in pan mode
        super().focusOutEvent(event)

    def keyPressEvent(self, event):
        key = event.key()
        if key == Qt.Key.Key_Space:            # hold to pan with any tool (a shortcut cannot see the release)
            if not event.isAutoRepeat():
                self.set_temp_pan(True)
            return
        if self.tool == 'lasso' and self._lasso:
            if key in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
                self._finish_lasso(subtract=bool(event.modifiers() & Qt.KeyboardModifier.ShiftModifier))
                return
            if key == Qt.Key.Key_Escape:
                self._finish_lasso(cancel=True)
                return
            if key == Qt.Key.Key_Backspace:
                self._lasso.pop()
                self._lasso_update()
                return
        super().keyPressEvent(event)

    # -- brush / eraser ---------------------------------------------------------------

    def _target_instance(self):
        """Active mask, or a new empty one (class from `new_label`) when none is selected."""
        if self.active is not None and any(self.active is i for i in self.doc.instances):
            return self.active, False
        inst = self.doc.new_empty(self.new_label())
        self.set_selection([inst], inst)
        return inst, True

    def _begin_stroke(self, point):
        if self.tool == 'erase' and self.active is None:
            return
        self.doc.stack.beginMacro('Brush stroke' if self.tool == 'brush' else 'Erase stroke')
        inst, created = self._target_instance()
        if not inst.get('keep', True):
            self.doc.set_keep([(inst, True)])
        self._stroke = {'inst': inst, 'before': inst['mask'].copy(), 'prev': None, 'rect': None, 'created': created}
        self._paint_to(point)

    def _paint_to(self, point):
        stroke = self._stroke
        inst = stroke['inst']
        mask = inst['mask']
        h, w = mask.shape
        x, y = int(round(point.x())), int(round(point.y()))
        prev = stroke['prev'] or (x, y)
        r = self.radius
        x0, x1 = max(0, min(prev[0], x) - r - 1), min(w, max(prev[0], x) + r + 2)
        y0, y1 = max(0, min(prev[1], y) - r - 1), min(h, max(prev[1], y) + r + 2)
        stroke['prev'] = (x, y)
        if x1 <= x0 or y1 <= y0:
            return
        value = 0 if self.tool == 'erase' else 1
        tile = mask[y0:y1, x0:x1].astype(np.uint8)
        cv2.line(tile, (prev[0] - x0, prev[1] - y0), (x - x0, y - y0), value, 2 * r)
        cv2.circle(tile, (x - x0, y - y0), r, value, -1)
        mask[y0:y1, x0:x1] = tile > 0
        rect = (x0, y0, x1, y1)
        stroke['rect'] = _union(stroke['rect'], rect)
        cached = self._bboxes.get(id(inst))
        self._bboxes[id(inst)] = _union(cached, rect) if cached and cached[2] > cached[0] else rect
        self._compose(self._expand(rect))

    def _end_stroke(self):
        stroke, self._stroke = self._stroke, None
        inst, rect = stroke['inst'], stroke['rect']
        if rect is not None:
            x0, y0, x1, y1 = rect
            before = stroke['before'][y0:y1, x0:x1]
            after = inst['mask'][y0:y1, x0:x1]
            if not np.array_equal(before, after):
                self.doc.edit_region(inst, (x0, y0, x1 - x0, y1 - y0), before.copy(), after.copy(),
                                     'Brush stroke' if self.tool == 'brush' else 'Erase stroke', applied=True)
        self.doc.stack.endMacro()
        if stroke['created'] and not inst['mask'].any():
            self.doc.undo()                    # an empty click must not leave an empty mask behind
        else:
            self.strokeFinished.emit(inst)

    # -- lasso ------------------------------------------------------------------------

    def _lasso_update(self, cursor=None):
        points = list(self._lasso) + ([cursor] if cursor is not None else [])
        self._lasso_item.setPolygon(QPolygonF(points))
        self._lasso_item.setVisible(bool(points))

    def _finish_lasso(self, subtract=False, cancel=False):
        points, self._lasso = self._lasso, []
        self._lasso_item.hide()
        if cancel or len(points) < 3 or self.doc is None or self.rgb is None:
            return
        if subtract and self.active is None:
            return
        h, w = self.rgb.shape[:2]
        polygon = np.array([[int(round(p.x())), int(round(p.y()))] for p in points], np.int32)
        x0, y0 = max(0, polygon[:, 0].min()), max(0, polygon[:, 1].min())
        x1, y1 = min(w, polygon[:, 0].max() + 1), min(h, polygon[:, 1].max() + 1)
        if x1 <= x0 or y1 <= y0:
            return
        self.doc.stack.beginMacro('Lasso')
        inst, created = self._target_instance()
        if not inst.get('keep', True):
            self.doc.set_keep([(inst, True)])
        filled = np.zeros((y1 - y0, x1 - x0), np.uint8)
        cv2.fillPoly(filled, [polygon - np.array([x0, y0], np.int32)], 1)
        before = inst['mask'][y0:y1, x0:x1].copy()
        after = (before & ~(filled > 0)) if subtract else (before | (filled > 0))
        if not np.array_equal(before, after):
            self.doc.edit_region(inst, (x0, y0, x1 - x0, y1 - y0), before, after,
                                 'Lasso subtract' if subtract else 'Lasso add', applied=False)
        self.doc.stack.endMacro()
        if created and not inst['mask'].any():
            self.doc.undo()
        else:
            self.strokeFinished.emit(inst)
