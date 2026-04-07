"""Qt implementation of IImageAnnotationViewer.

This module is the *only* place in the codebase where Qt rendering APIs
(QGraphicsView, QGraphicsScene, QPixmap, etc.) are used for image display.

Swapping Qt for another rendering backend (e.g. OpenGL) requires only
implementing the interface in a new file; no other module needs to change.
"""
from __future__ import annotations

import os
import logging
from typing import Callable, Optional

import cv2
import numpy as np

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QCursor, QImage, QMouseEvent, QPixmap, QColor
from PyQt6.QtWidgets import (
    QGraphicsPixmapItem,
    QGraphicsScene,
    QGraphicsView,
    QVBoxLayout,
    QWidget,
)

from viewer.interface import IImageAnnotationViewer

logger = logging.getLogger(__name__)

# Z-order constants (higher = on top)
_Z_IMAGE       = 0
_Z_ANNOTATIONS = 1
_Z_MISSING     = 2
_Z_SELECTION   = 3
_Z_TOOL        = 4
_Z_GRID        = 5

_RESOURCES_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "resources")
_TOOL_CURSOR_FILES = {"pen": "pen.png", "selector": "wand.png", "fill": "fill.png"}


class QtImageAnnotationViewer(QWidget):
    """Qt-based implementation of the viewer interface.

    Wraps ``QGraphicsView`` / ``QGraphicsScene`` to display the base image,
    annotation overlays, interaction feedback masks, and a pixel grid.

    All public API methods accept NumPy arrays; Qt internals are hidden.
    The caller (``AnnotatorController``) depends only on the abstract
    ``IImageAnnotationViewer`` contract.
    """

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._zoom: int = 5

        self._scene = QGraphicsScene()
        self._view = QGraphicsView(self._scene)
        self._view.setAlignment(
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop
        )
        self._view.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self._view.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self._view.setMouseTracking(True)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._view)

        # Graphics items — all start as None; created in set_base_image.
        self._q_image:       Optional[QGraphicsPixmapItem] = None
        self._q_annotations: Optional[QGraphicsPixmapItem] = None
        self._q_missing:     Optional[QGraphicsPixmapItem] = None
        self._q_selection:   Optional[QGraphicsPixmapItem] = None
        self._q_tool:        Optional[QGraphicsPixmapItem] = None
        self._q_grid:        Optional[QGraphicsPixmapItem] = None

        # Cursor cache: {(tool_name, (r, g, b)): QCursor}
        self._cursor_cache: dict[tuple, QCursor] = {}

        # Registered callbacks
        self._cb_mouse_press:   list[Callable] = []
        self._cb_mouse_release: list[Callable] = []
        self._cb_mouse_move:    list[Callable] = []
        self._cb_scroll:        list[Callable] = []
        self._cb_key_press:     list[Callable] = []
        self._cb_key_release:   list[Callable] = []

        # Wire Qt input events to our handlers.
        self._view.mousePressEvent   = self._on_mouse_press
        self._view.mouseMoveEvent    = self._on_mouse_move
        self._view.mouseReleaseEvent = self._on_mouse_release
        self._view.wheelEvent        = self._on_wheel

        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    # ------------------------------------------------------------------
    # IImageAnnotationViewer — display API
    # ------------------------------------------------------------------

    def set_base_image(self, bgr: np.ndarray) -> None:
        self._scene.clear()
        self._q_image = self._q_annotations = self._q_missing = None
        self._q_selection = self._q_tool = self._q_grid = None

        pixmap = self._bgr_to_pixmap(bgr)
        self._q_image = QGraphicsPixmapItem(pixmap)
        self._q_image.setZValue(_Z_IMAGE)
        self._q_image.setScale(self._zoom)
        self._q_image.setPos(0, 0)
        self._scene.addItem(self._q_image)

        h, w = bgr.shape[:2]
        blank_rgba = np.zeros((h, w, 4), dtype=np.uint8)
        blank_gray = np.zeros((h, w), dtype=np.uint8)

        self._q_annotations = self._add_rgba_item(blank_rgba, _Z_ANNOTATIONS)
        self._q_missing      = self._add_gray_item(blank_gray, _Z_MISSING)
        self._q_selection    = self._add_rgba_item(blank_rgba, _Z_SELECTION)
        self._q_tool         = self._add_rgba_item(blank_rgba, _Z_TOOL)

        self._q_missing.setVisible(False)
        self._q_selection.setVisible(False)
        self._q_tool.setVisible(False)

        self._update_scene_rect()
        self._update_grid()

    def set_base_image_visible(self, visible: bool) -> None:
        if self._q_image:
            self._q_image.setVisible(visible)

    def set_annotation_overlay(self, rgba: np.ndarray) -> None:
        if self._q_annotations is None:
            return
        self._q_annotations.setPixmap(self._rgba_to_pixmap(rgba))
        self._q_annotations.setScale(self._zoom)

    def set_annotations_visible(self, visible: bool) -> None:
        if self._q_annotations:
            self._q_annotations.setVisible(visible)

    def set_selection_mask(
        self,
        mask: Optional[np.ndarray],
        color_rgb: tuple[int, int, int],
    ) -> None:
        if self._q_selection is None:
            return
        if mask is None:
            self._q_selection.setVisible(False)
            return
        self._q_selection.setPixmap(self._rgba_to_pixmap(self._mask_to_rgba(mask, color_rgb, 64)))
        self._q_selection.setScale(self._zoom)
        self._q_selection.setPos(0, 0)
        self._q_selection.setVisible(True)

    def set_tool_preview(
        self,
        mask: Optional[np.ndarray],
        color_rgb: tuple[int, int, int],
    ) -> None:
        if self._q_tool is None:
            return
        if mask is None:
            self._q_tool.setVisible(False)
            return
        self._q_tool.setPixmap(self._rgba_to_pixmap(self._mask_to_rgba(mask, color_rgb, 64)))
        self._q_tool.setScale(self._zoom)
        self._q_tool.setPos(0, 0)
        self._q_tool.setVisible(True)

    def set_missing_pixels_visible(
        self,
        mask: Optional[np.ndarray],
        visible: bool,
    ) -> None:
        if self._q_missing is None:
            return
        if not visible or mask is None:
            self._q_missing.setVisible(False)
            return
        inv = cv2.bitwise_not(mask)
        h, w = inv.shape
        qimg = QImage(inv.data, w, h, w, QImage.Format.Format_Grayscale8)
        self._q_missing.setPixmap(QPixmap.fromImage(qimg))
        self._q_missing.setScale(self._zoom)
        self._q_missing.setVisible(True)

    def set_zoom(
        self,
        zoom: int,
        center: Optional[tuple[float, float]] = None,
    ) -> None:
        self._zoom = zoom
        for item in (
            self._q_image,
            self._q_annotations,
            self._q_selection,
            self._q_tool,
            self._q_missing,
        ):
            if item is not None:
                item.setScale(zoom)
        self._update_scene_rect()
        self._update_grid()
        if center is not None:
            self._scroll_to_center(center)

    def get_zoom(self) -> int:
        return self._zoom

    def get_view_center(self) -> tuple[float, float]:
        hbar = self._view.horizontalScrollBar()
        vbar = self._view.verticalScrollBar()
        vw = self._view.viewport().width()
        vh = self._view.viewport().height()
        cx = (hbar.value() + vw // 2) / max(self._zoom, 1)
        cy = (vbar.value() + vh // 2) / max(self._zoom, 1)
        return float(cx), float(cy)

    def update_cursor(
        self,
        tool: str,
        layer_color: tuple[int, int, int],
    ) -> None:
        key = (tool, layer_color)
        cursor = self._cursor_cache.get(key)
        if cursor is None:
            cursor = self._build_cursor(tool, layer_color)
            self._cursor_cache[key] = cursor
        if cursor:
            self._view.setCursor(cursor)
        else:
            self._view.setCursor(Qt.CursorShape.CrossCursor)

    @property
    def widget(self) -> QWidget:
        return self

    # ------------------------------------------------------------------
    # IImageAnnotationViewer — input registration
    # ------------------------------------------------------------------

    def register_mouse_press(self, cb: Callable[[int, int, str], None]) -> None:
        self._cb_mouse_press.append(cb)

    def register_mouse_release(self, cb: Callable[[int, int, str], None]) -> None:
        self._cb_mouse_release.append(cb)

    def register_mouse_move(self, cb: Callable[[int, int], None]) -> None:
        self._cb_mouse_move.append(cb)

    def register_scroll(self, cb: Callable) -> None:
        self._cb_scroll.append(cb)

    def register_key_press(self, cb: Callable[[str, frozenset], None]) -> None:
        self._cb_key_press.append(cb)

    def register_key_release(self, cb: Callable[[str, frozenset], None]) -> None:
        self._cb_key_release.append(cb)

    # ------------------------------------------------------------------
    # Qt event handlers — translate to semantic callbacks
    # ------------------------------------------------------------------

    def _on_mouse_press(self, event: QMouseEvent) -> None:
        px, py = self._to_pixel(event.pos())
        btn = _button_name(event.button())
        for cb in self._cb_mouse_press:
            cb(px, py, btn)

    def _on_mouse_release(self, event: QMouseEvent) -> None:
        px, py = self._to_pixel(event.pos())
        btn = _button_name(event.button())
        for cb in self._cb_mouse_release:
            cb(px, py, btn)

    def _on_mouse_move(self, event: QMouseEvent) -> None:
        px, py = self._to_pixel(event.pos())
        for cb in self._cb_mouse_move:
            cb(px, py)

    def _on_wheel(self, event) -> None:
        mods = _modifiers_frozenset(event.modifiers())
        dy = event.angleDelta().y()
        dx = event.angleDelta().x()

        if "shift" in mods:
            self._view.horizontalScrollBar().setValue(
                self._view.horizontalScrollBar().value() - dy
            )
        elif "ctrl" in mods:
            scene_pos = self._view.mapToScene(event.pos())
            px = int(scene_pos.x() / max(self._zoom, 1))
            py = int(scene_pos.y() / max(self._zoom, 1))
            for cb in self._cb_scroll:
                cb(dy, dx, px, py, mods)
        else:
            self._view.verticalScrollBar().setValue(
                self._view.verticalScrollBar().value() - dy
            )
            self._view.horizontalScrollBar().setValue(
                self._view.horizontalScrollBar().value() - dx
            )

    def keyPressEvent(self, event) -> None:  # noqa: N802
        key = _key_name(event.key())
        mods = _modifiers_frozenset(event.modifiers())
        for cb in self._cb_key_press:
            cb(key, mods)

    def keyReleaseEvent(self, event) -> None:  # noqa: N802
        key = _key_name(event.key())
        mods = _modifiers_frozenset(event.modifiers())
        for cb in self._cb_key_release:
            cb(key, mods)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _to_pixel(self, pos) -> tuple[int, int]:
        scene_pos = self._view.mapToScene(pos)
        zoom = max(self._zoom, 1)
        return int(scene_pos.x() / zoom), int(scene_pos.y() / zoom)

    def _scroll_to_center(self, center: tuple[float, float]) -> None:
        hbar = self._view.horizontalScrollBar()
        vbar = self._view.verticalScrollBar()
        cx, cy = center
        vw = self._view.viewport().width()
        vh = self._view.viewport().height()
        target_h = int(cx * self._zoom) - vw // 2
        target_v = int(cy * self._zoom) - vh // 2
        hbar.setValue(max(hbar.minimum(), min(hbar.maximum(), target_h)))
        vbar.setValue(max(vbar.minimum(), min(vbar.maximum(), target_v)))

    def _update_scene_rect(self) -> None:
        if self._q_image is None:
            return
        px = self._q_image.pixmap()
        w = int(px.width() * self._zoom)
        h = int(px.height() * self._zoom)
        self._scene.setSceneRect(0, 0, w, h)

    def _update_grid(self) -> None:
        if self._q_image is None:
            return
        px = self._q_image.pixmap()
        zoom = self._zoom
        w = int(px.width() * zoom)
        h = int(px.height() * zoom)
        if w <= 0 or h <= 0:
            return

        grid = np.zeros((h, w, 4), dtype=np.uint8)
        grid[::zoom, :, :3] = 128
        grid[:, ::zoom, :3] = 128
        grid[::zoom, :, 3]  = 255
        grid[:, ::zoom, 3]  = 255

        qimg = QImage(grid.data, w, h, QImage.Format.Format_RGBA8888)
        grid_pixmap = QPixmap.fromImage(qimg)

        if self._q_grid:
            self._scene.removeItem(self._q_grid)
        self._q_grid = QGraphicsPixmapItem(grid_pixmap)
        self._q_grid.setZValue(_Z_GRID)
        self._scene.addItem(self._q_grid)

    def _add_rgba_item(self, rgba: np.ndarray, z: int) -> QGraphicsPixmapItem:
        h, w = rgba.shape[:2]
        bpl = w * 4
        qimg = QImage(rgba.data, w, h, bpl, QImage.Format.Format_RGBA8888)
        item = QGraphicsPixmapItem(QPixmap.fromImage(qimg))
        item.setZValue(z)
        item.setPos(0, 0)
        item.setScale(self._zoom)
        self._scene.addItem(item)
        return item

    def _add_gray_item(self, gray: np.ndarray, z: int) -> QGraphicsPixmapItem:
        h, w = gray.shape
        qimg = QImage(gray.data, w, h, w, QImage.Format.Format_Grayscale8)
        item = QGraphicsPixmapItem(QPixmap.fromImage(qimg))
        item.setZValue(z)
        item.setPos(0, 0)
        item.setScale(self._zoom)
        self._scene.addItem(item)
        return item

    @staticmethod
    def _build_cursor(
        tool: str,
        layer_color: tuple[int, int, int],
    ) -> Optional[QCursor]:
        cursor_file = _TOOL_CURSOR_FILES.get(tool)
        if not cursor_file:
            return None
        cursor_path = os.path.join(_RESOURCES_DIR, cursor_file)
        if not os.path.exists(cursor_path):
            return None
        image = QImage(cursor_path)
        target = QColor(layer_color[0], layer_color[1], layer_color[2])
        for y in range(image.height()):
            for x in range(image.width()):
                if image.pixelColor(x, y) == Qt.GlobalColor.white:
                    image.setPixelColor(x, y, target)
        return QCursor(QPixmap.fromImage(image), 0, 0)

    @staticmethod
    def _bgr_to_pixmap(bgr: np.ndarray) -> QPixmap:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        bpl = w * 3
        qimg = QImage(rgb.data, w, h, bpl, QImage.Format.Format_RGB888)
        return QPixmap.fromImage(qimg)

    @staticmethod
    def _rgba_to_pixmap(rgba: np.ndarray) -> QPixmap:
        h, w = rgba.shape[:2]
        bpl = w * 4
        qimg = QImage(rgba.data, w, h, bpl, QImage.Format.Format_RGBA8888)
        return QPixmap.fromImage(qimg)

    @staticmethod
    def _mask_to_rgba(
        mask: np.ndarray,
        color_rgb: tuple[int, int, int],
        opacity: int,
    ) -> np.ndarray:
        h, w = mask.shape
        rgba = np.zeros((h, w, 4), dtype=np.uint8)
        rgba[mask > 0] = color_rgb + (opacity,)
        return rgba


# ------------------------------------------------------------------
# Qt → semantic string helpers (module-level, stateless)
# ------------------------------------------------------------------

def _button_name(button) -> str:
    if button == Qt.MouseButton.LeftButton:
        return "left"
    if button == Qt.MouseButton.RightButton:
        return "right"
    if button == Qt.MouseButton.MiddleButton:
        return "middle"
    return "unknown"


def _key_name(key) -> str:
    _map = {
        Qt.Key.Key_Z:      "Z",
        Qt.Key.Key_E:      "E",
        Qt.Key.Key_R:      "R",
        Qt.Key.Key_Plus:   "Plus",
        Qt.Key.Key_Minus:  "Minus",
        Qt.Key.Key_Space:  "Space",
        Qt.Key.Key_Escape: "Escape",
        Qt.Key.Key_Return: "Return",
        Qt.Key.Key_Enter:  "Return",
    }
    if key in _map:
        return _map[key]
    # Digit keys Key_1 … Key_9
    for i in range(1, 10):
        if key == getattr(Qt.Key, f"Key_{i}"):
            return str(i)
    return ""


def _modifiers_frozenset(mods) -> frozenset:
    result = set()
    if mods & Qt.KeyboardModifier.ControlModifier:
        result.add("ctrl")
    if mods & Qt.KeyboardModifier.ShiftModifier:
        result.add("shift")
    if mods & Qt.KeyboardModifier.AltModifier:
        result.add("alt")
    return frozenset(result)
