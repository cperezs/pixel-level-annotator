"""Gallery panel — thumbnail browser for available images.

Opens as a slide-out panel between the left sidebar and the canvas.
Shows image thumbnails from the repository; clicking one loads it
through the existing image-loading flow.
"""
from __future__ import annotations

import os
import logging
from typing import Callable, Optional

from PyQt6.QtCore import Qt, QSize
from PyQt6.QtGui import QPixmap, QImage
from PyQt6.QtWidgets import (
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from presentation.style import (
    PRIMARY,
    ON_SURFACE,
    ON_SURFACE_VARIANT,
    OUTLINE_VARIANT,
    SURFACE,
    SURFACE_CONTAINER,
    SURFACE_CONTAINER_HIGH,
    SURFACE_CONTAINER_HIGHEST,
    SURFACE_CONTAINER_LOW,
    FONT_SIZE_SM,
    FONT_SIZE_XS,
)

logger = logging.getLogger(__name__)

_THUMB_SIZE = 100  # px

# Module-level pixmap cache: maps (absolute_path, mtime_ns) -> QPixmap
# Survives gallery close/open cycles without re-reading from disk.
_thumb_cache: dict[tuple, "QPixmap"] = {}


class _ThumbnailItem(QFrame):
    """A single thumbnail card with image preview and filename."""

    def __init__(self, filename: str, images_dir: str, parent=None):
        super().__init__(parent)
        self._filename = filename
        self._selected = False
        self._cb_clicked: Optional[Callable[[str], None]] = None

        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFixedSize(_THUMB_SIZE + 16, _THUMB_SIZE + 32)
        self._apply_style()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(2)

        # Thumbnail image
        thumb_label = QLabel()
        thumb_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        thumb_label.setFixedSize(_THUMB_SIZE + 8, _THUMB_SIZE)
        thumb_label.setStyleSheet("background: transparent; border: none;")

        image_path = os.path.join(images_dir, filename)
        pixmap = self._load_thumbnail(image_path)
        if pixmap:
            thumb_label.setPixmap(pixmap)
        else:
            thumb_label.setText("?")
            thumb_label.setStyleSheet(
                f"color: {ON_SURFACE_VARIANT}; font-size: 24px; "
                f"background: {SURFACE}; border-radius: 4px;"
            )
        layout.addWidget(thumb_label)

        # Filename label
        name_label = QLabel(filename)
        name_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        name_label.setStyleSheet(
            f"color: {ON_SURFACE_VARIANT}; font-size: {FONT_SIZE_XS}px; "
            f"background: transparent; border: none;"
        )
        name_label.setWordWrap(True)
        layout.addWidget(name_label)

    def on_clicked(self, cb: Callable[[str], None]):
        self._cb_clicked = cb

    def set_selected(self, selected: bool):
        self._selected = selected
        self._apply_style()

    def mousePressEvent(self, event):
        if self._cb_clicked:
            self._cb_clicked(self._filename)

    def _apply_style(self):
        if self._selected:
            self.setStyleSheet(f"""
                _ThumbnailItem {{
                    background-color: {SURFACE_CONTAINER_HIGHEST};
                    border: 1px solid rgba(161, 250, 255, 0.3);
                    border-radius: 8px;
                }}
            """)
        else:
            self.setStyleSheet(f"""
                _ThumbnailItem {{
                    background-color: {SURFACE_CONTAINER_HIGHEST};
                    border: 1px solid transparent;
                    border-radius: 8px;
                }}
                _ThumbnailItem:hover {{
                    border: 1px solid rgba(72, 72, 72, 0.3);
                }}
            """)

    @staticmethod
    def _load_thumbnail(path: str) -> Optional[QPixmap]:
        """Load and scale an image to thumbnail size, using a module-level cache."""
        try:
            mtime = os.stat(path).st_mtime_ns
        except OSError:
            return None
        key = (path, mtime)
        if key in _thumb_cache:
            return _thumb_cache[key]
        try:
            pixmap = QPixmap(path)
            if pixmap.isNull():
                return None
            scaled = pixmap.scaled(
                _THUMB_SIZE, _THUMB_SIZE,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            _thumb_cache[key] = scaled
            return scaled
        except Exception:
            return None


# ------------------------------------------------------------------
# GalleryPanel
# ------------------------------------------------------------------

class GalleryPanel(QWidget):
    """Slide-out gallery showing thumbnails of available images."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setFixedWidth(280)
        self.setStyleSheet(f"background-color: {SURFACE_CONTAINER_HIGH};")

        self._images_dir = ""
        self._items: list[_ThumbnailItem] = []
        self._cb_image_selected: Optional[Callable[[str], None]] = None
        self._current_filename: Optional[str] = None
        self._populated_filenames: list[str] = []

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Header
        header = QWidget()
        header.setStyleSheet(f"background-color: {SURFACE_CONTAINER_HIGH};")
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(14, 12, 14, 12)

        title = QLabel("GALLERY")
        title.setStyleSheet(
            f"color: {PRIMARY}; font-size: {FONT_SIZE_SM}px; "
            f"font-weight: 700; letter-spacing: 1px;"
        )
        header_layout.addWidget(title)
        header_layout.addStretch()

        close_btn = QPushButton("✕")
        close_btn.setFixedSize(24, 24)
        close_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        close_btn.setStyleSheet(f"""
            QPushButton {{
                background: transparent; border: none;
                color: {ON_SURFACE_VARIANT}; font-size: 14px;
                border-radius: 4px;
            }}
            QPushButton:hover {{ color: {ON_SURFACE}; background: {SURFACE_CONTAINER_HIGHEST}; }}
        """)
        close_btn.clicked.connect(self.hide)
        header_layout.addWidget(close_btn)
        layout.addWidget(header)

        # Scroll area with grid
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setStyleSheet("QScrollArea { border: none; background: transparent; }")

        self._grid_container = QWidget()
        self._grid_layout = QGridLayout(self._grid_container)
        self._grid_layout.setContentsMargins(10, 8, 10, 8)
        self._grid_layout.setSpacing(8)
        self._grid_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        scroll.setWidget(self._grid_container)
        layout.addWidget(scroll, 1)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def on_image_selected(self, cb: Callable[[str], None]) -> None:
        self._cb_image_selected = cb

    def populate(self, filenames: list[str], images_dir: str) -> None:
        """Fill the gallery with thumbnail items."""
        # Skip expensive rebuild when nothing has changed
        if filenames == self._populated_filenames and images_dir == self._images_dir:
            return

        self._images_dir = images_dir
        self._populated_filenames = list(filenames)

        # Clear existing items
        for item in self._items:
            self._grid_layout.removeWidget(item)
            item.deleteLater()
        self._items.clear()

        cols = 2
        for idx, filename in enumerate(filenames):
            item = _ThumbnailItem(filename, images_dir)
            item.on_clicked(self._on_item_clicked)
            if filename == self._current_filename:
                item.set_selected(True)
            self._items.append(item)
            row, col = divmod(idx, cols)
            self._grid_layout.addWidget(item, row, col)

    def set_current_filename(self, filename: Optional[str]) -> None:
        """Highlight the currently loaded image."""
        self._current_filename = filename
        for item in self._items:
            item.set_selected(item._filename == filename)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _on_item_clicked(self, filename: str) -> None:
        self.set_current_filename(filename)
        if self._cb_image_selected:
            self._cb_image_selected(filename)
