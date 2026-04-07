"""AnnotatorController — the application orchestrator.

Responsibilities:
- Respond to viewer input events (mouse, keyboard, scroll).
- Execute domain operations (annotation, undo, autolabel).
- Drive persistence (saving annotation files and metadata).
- Keep viewer overlays in sync with domain state.
- Expose a clean Python API for the presentation layer.

The controller knows nothing about QPushButton, QLabel, or any other
widget.  It is the only component that holds direct references to both
the viewer and the domain repository.
"""
from __future__ import annotations

import os
import shutil
import uuid
import logging
from typing import Callable, Optional

import cv2
import numpy as np

from domain.image_document import ImageDocument
from domain.layer_config import LayerConfig
from infrastructure.image_repository import ImageRepository
from application.app_state import AppState
from application.annotation_tools import (
    compute_pen_mask,
    apply_overwrite_guard,
    smooth_mask,
    expand_mask,
    shrink_mask,
    build_annotation_rgba,
)
from application.autolabel_service import AutolabelService
from viewer.interface import IImageAnnotationViewer
from infrastructure.metadata import ImageMetadata
from infrastructure.time_tracker import TimeTracker

logger = logging.getLogger(__name__)


class AnnotatorController:
    """Wires together the viewer, state, domain layer, and infrastructure.

    Construction
    ------------
    Pass a viewer, the initial layer configuration, and a repository.
    The presenter (main window) then wires up its own callbacks.

    Input flow
    ----------
    Viewer fires raw events → controller translates to domain operations →
    domain state mutates → controller syncs viewer overlays.
    """

    def __init__(
        self,
        viewer: IImageAnnotationViewer,
        layer_configs: list[LayerConfig],
        image_repo: ImageRepository,
    ) -> None:
        self._viewer = viewer
        self._layer_configs = layer_configs
        self._image_repo = image_repo

        layer_names = [lc.name for lc in layer_configs]
        self._state = AppState(layer_names)
        self._document: Optional[ImageDocument] = None
        self._current_filename: Optional[str] = None
        self._metadata: Optional[ImageMetadata] = None
        self._time_tracker: Optional[TimeTracker] = None
        self._autolabel = AutolabelService(layer_names)
        self._last_mouse_pos: Optional[tuple[int, int]] = None

        # Callbacks for the presentation layer to observe.
        self._on_progress_changed:        list[Callable[[int], None]] = []
        self._on_web_service_image_ready: list[Callable[[str, object], None]] = []

        # Wire viewer input events → our handlers.
        viewer.register_mouse_press(self._handle_mouse_press)
        viewer.register_mouse_release(self._handle_mouse_release)
        viewer.register_mouse_move(self._handle_mouse_move)
        viewer.register_scroll(self._handle_scroll)
        viewer.register_key_press(self._handle_key_press)
        viewer.register_key_release(self._handle_key_release)

    # ------------------------------------------------------------------
    # Public read-only API for the presentation layer
    # ------------------------------------------------------------------

    @property
    def state(self) -> AppState:
        return self._state

    @property
    def document(self) -> Optional[ImageDocument]:
        return self._document

    @property
    def layer_configs(self) -> list[LayerConfig]:
        return self._layer_configs

    @property
    def autolabel_service(self) -> AutolabelService:
        return self._autolabel

    # ------------------------------------------------------------------
    # Callback registration (for the presentation layer)
    # ------------------------------------------------------------------

    def on_progress_changed(self, cb: Callable[[int], None]) -> None:
        self._on_progress_changed.append(cb)

    def on_web_service_image_ready(self, cb: Callable[[str, object], None]) -> None:
        self._on_web_service_image_ready.append(cb)

    # ------------------------------------------------------------------
    # Image loading
    # ------------------------------------------------------------------

    def load_image(self, filename: str) -> bool:
        """Load *filename* from the image repository.

        Returns True on success.  The viewer is updated immediately.
        """
        self._finalize_autolabel_session()
        if self._metadata:
            self._metadata.save()

        nlayers = len(self._layer_configs)
        try:
            doc = self._image_repo.load(filename, nlayers)
        except FileNotFoundError as exc:
            logger.error("Cannot load image: %s", exc)
            return False

        self._document = doc
        self._current_filename = filename

        meta_path = self._image_repo.metadata_path(filename)
        layer_names = [lc.name for lc in self._layer_configs]
        self._metadata = ImageMetadata.load(meta_path, layer_names)
        self._metadata.update_pixel_stats(
            doc.annotations, image_size=(doc.height, doc.width)
        )
        self._time_tracker = TimeTracker(self._metadata)
        # Start timing layer 0 immediately so that the elapsed time up to the
        # first edit is captured.  change() starts the clock but only flushes
        # on the *next* call, so calling it here mirrors the old behaviour where
        # set_state("selected_layer") triggered track_time on image load.
        self._time_tracker.change(0)

        # Reset interaction state.
        s = self._state
        s.tool.is_drawing = False
        s.tool.selector_origin = None
        s.session.active_layer = 0
        s.session.selection_mask = None
        s.session.tool_preview_mask = None
        s.view.zoom = 5
        s.view.center_pos = None

        # Initialise the viewer.
        self._viewer.set_base_image(doc.image)
        self._sync_annotation_overlay()
        self._viewer.set_zoom(5)
        self._viewer.update_cursor(
            s.tool.active,
            self._layer_configs[0].color_rgb,
        )
        self._notify_progress()
        return True

    # ------------------------------------------------------------------
    # Tool selection
    # ------------------------------------------------------------------

    def select_tool(self, tool: str) -> None:
        """Switch the active annotation tool (``"pen"``, ``"selector"``, ``"fill"``)."""
        self._state.tool.active = tool
        self._state.tool.is_drawing = False
        self._state.session.selection_mask = None
        self._viewer.set_selection_mask(None, (255, 255, 255))
        self._viewer.set_tool_preview(None, (255, 255, 255))
        self._viewer.update_cursor(
            tool,
            self._layer_configs[self._state.session.active_layer].color_rgb,
        )
        self._state.notify("tool")

    def set_active_layer(self, layer_index: int) -> None:
        self._state.session.active_layer = layer_index
        if self._time_tracker:
            self._time_tracker.change(layer_index)
        self._sync_annotation_overlay()
        self._viewer.update_cursor(
            self._state.tool.active,
            self._layer_configs[layer_index].color_rgb,
        )
        self._state.notify("session")

    def set_pen_size(self, size: int) -> None:
        self._state.tool.pen_size = max(1, size)
        self._state.notify("tool")
        # Refresh the cursor preview circle at the current mouse position
        # so the user sees the new size immediately without moving the mouse.
        if (
            self._state.tool.active == "pen"
            and not self._state.tool.is_drawing
            and self._last_mouse_pos is not None
            and self._document is not None
        ):
            px, py = self._last_mouse_pos
            layer = self._state.session.active_layer
            color = self._layer_configs[layer].color_rgb
            preview = compute_pen_mask(
                self._document.height, self._document.width,
                px, py, self._state.tool.pen_size,
            )
            self._viewer.set_tool_preview(preview, color)

    def set_selector_threshold(self, threshold: int) -> None:
        self._state.tool.selector_threshold = max(1, min(128, threshold))
        # Recompute selector mask if a selection is in progress.
        if (
            self._state.tool.is_drawing
            and self._state.tool.selector_origin is not None
            and self._document is not None
        ):
            x, y = self._state.tool.selector_origin
            self._run_selector(x, y)
        self._state.notify("tool")

    def set_selector_auto_smooth(self, enabled: bool) -> None:
        self._state.tool.selector_auto_smooth = enabled

    def set_overwrite_annotations(self, enabled: bool) -> None:
        self._state.tool.overwrite_annotations = enabled

    def set_fill_all(self, enabled: bool) -> None:
        self._state.tool.fill_all = enabled

    # ------------------------------------------------------------------
    # Zoom
    # ------------------------------------------------------------------

    def zoom_in(self, center: Optional[tuple[float, float]] = None) -> None:
        zoom = self._state.view.zoom
        if zoom >= 40:
            return
        c = center or self._viewer.get_view_center()
        new_zoom = 5 if zoom < 5 else zoom + 5
        self._state.view.zoom = new_zoom
        self._state.view.center_pos = c
        self._viewer.set_zoom(new_zoom, c)

    def zoom_out(self, center: Optional[tuple[float, float]] = None) -> None:
        zoom = self._state.view.zoom
        if zoom <= 1:
            return
        c = center or self._viewer.get_view_center()
        new_zoom = 1 if zoom <= 5 else zoom - 5
        self._state.view.zoom = new_zoom
        self._state.view.center_pos = c
        self._viewer.set_zoom(new_zoom, c)

    # ------------------------------------------------------------------
    # View toggles
    # ------------------------------------------------------------------

    def toggle_show_image(self, visible: bool) -> None:
        self._state.view.show_image = visible
        self._viewer.set_base_image_visible(visible)

    def toggle_show_other_layers(self, visible: bool) -> None:
        self._state.view.show_other_layers = visible
        self._sync_annotation_overlay()

    def toggle_show_missing_pixels(self, visible: bool) -> None:
        self._state.view.show_missing_pixels = visible
        if self._document is not None:
            mask = self._document.get_missing_annotations_mask()
            self._viewer.set_missing_pixels_visible(mask, visible)
        else:
            self._viewer.set_missing_pixels_visible(None, False)

    def toggle_annotations_visible(self, visible: bool) -> None:
        self._viewer.set_annotations_visible(visible)
        if not visible:
            self._viewer.set_missing_pixels_visible(None, False)
        elif self._state.view.show_missing_pixels and self._document:
            mask = self._document.get_missing_annotations_mask()
            self._viewer.set_missing_pixels_visible(mask, True)

    # ------------------------------------------------------------------
    # Undo
    # ------------------------------------------------------------------

    def undo(self) -> None:
        if self._document and self._document.undo():
            self._image_repo.save_annotations(self._document, self._current_filename)
            self._sync_annotation_overlay()
            self._notify_progress()

    # ------------------------------------------------------------------
    # Autolabeling
    # ------------------------------------------------------------------

    def run_autolabel(self, plugin_id: str) -> Optional[str]:
        """Run *plugin_id* on the current document.

        Returns an error message string on failure, or ``None`` on success.
        """
        if not self._document or not self._metadata:
            return "No image loaded"

        success, error = self._autolabel.run(plugin_id, self._document, self._metadata)
        if not success:
            return error

        self._image_repo.save_annotations(self._document, self._current_filename)
        if self._time_tracker:
            self._time_tracker.reset()
        self._sync_annotation_overlay()
        self._notify_progress()
        return None

    # ------------------------------------------------------------------
    # Layers reconfiguration
    # ------------------------------------------------------------------

    def update_layers(self, layer_configs: list[LayerConfig]) -> None:
        """Reconfigure layers (called after the layers file is reloaded)."""
        self._layer_configs = layer_configs
        layer_names = [lc.name for lc in layer_configs]
        self._state.layer_names = layer_names
        self._autolabel.refresh_plugins(layer_names)

    # ------------------------------------------------------------------
    # Persistence / session lifecycle
    # ------------------------------------------------------------------

    def save_current(self) -> None:
        """Flush metadata to disk."""
        if self._metadata:
            self._metadata.save()

    def close(self) -> None:
        """Clean up before the application terminates."""
        self._finalize_autolabel_session()
        self.save_current()

    # ------------------------------------------------------------------
    # Web service entry point
    # ------------------------------------------------------------------

    def load_web_service_image(self, image_path: str, request: object) -> None:
        """Copy *image_path* to the images directory and load it."""
        unique_filename = f"ws_{uuid.uuid4().hex[:8]}.png"
        target_path = os.path.join(self._image_repo._images_dir, unique_filename)
        os.makedirs(self._image_repo._images_dir, exist_ok=True)
        shutil.copy2(image_path, target_path)
        self.load_image(unique_filename)
        for cb in self._on_web_service_image_ready:
            cb(unique_filename, request)

    # ------------------------------------------------------------------
    # Viewer event handlers (registered in __init__)
    # ------------------------------------------------------------------

    def _handle_mouse_press(self, px: int, py: int, button: str) -> None:
        if not self._document or button != "left":
            return
        tool = self._state.tool
        if tool.active == "pen":
            self._pen_draw(px, py)
            tool.is_drawing = True
        elif tool.active == "selector":
            tool.is_drawing = True
            tool.selector_origin = (px, py)
            self._run_selector(px, py)
        elif tool.active == "fill":
            self._run_fill(px, py)

    def _handle_mouse_release(self, px: int, py: int, button: str) -> None:
        if not self._document or button != "left":
            return
        if self._state.tool.active == "pen" and self._state.tool.is_drawing:
            self._commit_annotation()
            self._state.tool.is_drawing = False

    def _handle_mouse_move(self, px: int, py: int) -> None:
        self._last_mouse_pos = (px, py)
        if not self._document:
            return
        tool = self._state.tool
        layer = self._state.session.active_layer
        color = self._layer_configs[layer].color_rgb

        if tool.active == "pen" and tool.is_drawing:
            self._pen_draw(px, py)
        elif tool.active == "pen":
            preview = compute_pen_mask(
                self._document.height, self._document.width, px, py, tool.pen_size
            )
            self._viewer.set_tool_preview(preview, color)
        elif tool.active in ("selector", "fill"):
            preview = compute_pen_mask(
                self._document.height, self._document.width, px, py, 1
            )
            self._viewer.set_tool_preview(preview, color)

    def _handle_scroll(
        self,
        dy: int,
        dx: int,
        px: int,
        py: int,
        mods: frozenset,
    ) -> None:
        if "ctrl" in mods:
            if dy > 0:
                self.zoom_in(center=(px, py))
            else:
                self.zoom_out(center=(px, py))

    def _handle_key_press(self, key: str, mods: frozenset) -> None:
        if key == "Space":
            self._viewer.set_annotations_visible(False)

    def _handle_key_release(self, key: str, mods: frozenset) -> None:  # noqa: C901
        tool = self._state.tool

        if key == "Z" and "ctrl" in mods:
            if tool.is_drawing:
                self._cancel_tool()
            else:
                self.undo()

        elif key == "Plus":
            if tool.active == "pen":
                self.set_pen_size(tool.pen_size + 1)
            elif tool.active == "selector":
                self.set_selector_threshold(tool.selector_threshold + 1)

        elif key == "Minus":
            if tool.active == "pen":
                self.set_pen_size(tool.pen_size - 1)
            elif tool.active == "selector":
                self.set_selector_threshold(tool.selector_threshold - 1)

        elif key == "Space":
            self._viewer.set_annotations_visible(True)
            if self._state.view.show_missing_pixels and self._document:
                mask = self._document.get_missing_annotations_mask()
                self._viewer.set_missing_pixels_visible(mask, True)

        elif key == "Escape":
            self._cancel_tool()

        elif key == "E" and tool.active == "selector" and tool.is_drawing:
            mask = self._state.session.selection_mask
            if mask is not None and self._document is not None:
                layer = self._state.session.active_layer
                annotated = (
                    None
                    if tool.overwrite_annotations
                    else self._document.get_other_annotations_mask(layer)
                )
                self._state.session.selection_mask = expand_mask(mask, annotated)
                self._sync_selection_mask()

        elif key == "R" and tool.active == "selector" and tool.is_drawing:
            mask = self._state.session.selection_mask
            if mask is not None:
                self._state.session.selection_mask = shrink_mask(mask)
                self._sync_selection_mask()

        elif key == "Return" and tool.active == "selector" and tool.is_drawing:
            self._commit_annotation()
            tool.is_drawing = False

        else:
            # Numeric keys 1–9 → layer selection.
            try:
                idx = int(key) - 1
                if 0 <= idx < len(self._layer_configs):
                    self.set_active_layer(idx)
            except (ValueError, TypeError):
                pass

    # ------------------------------------------------------------------
    # Tool implementations (private)
    # ------------------------------------------------------------------

    def _pen_draw(self, px: int, py: int) -> None:
        doc = self._document
        tool = self._state.tool
        pen_mask = compute_pen_mask(doc.height, doc.width, px, py, tool.pen_size)
        cur = self._state.session.selection_mask
        merged = cv2.bitwise_or(
            cur if cur is not None else np.zeros_like(pen_mask),
            pen_mask,
        )
        if not tool.overwrite_annotations:
            annotated = doc.get_other_annotations_mask(self._state.session.active_layer)
            merged = apply_overwrite_guard(merged, annotated)
        self._state.session.selection_mask = merged
        self._sync_selection_mask()

    def _run_selector(self, px: int, py: int) -> None:
        doc = self._document
        tool = self._state.tool
        mask = doc.compute_similarity_mask(
            px, py, tool.selector_threshold, tool.overwrite_annotations
        )
        if tool.selector_auto_smooth:
            mask = smooth_mask(mask)
        self._state.session.selection_mask = mask
        self._sync_selection_mask()

    def _run_fill(self, px: int, py: int) -> None:
        doc = self._document
        layer = self._state.session.active_layer
        connected = not self._state.tool.fill_all
        mask = doc.get_unannotated_mask(px, py, connected=connected)
        self._autolabel.begin_correction(doc.annotations)
        doc.annotate_mask(mask, layer)
        self._autolabel.end_correction(doc.annotations)
        self._post_annotation_commit(layer)

    def _commit_annotation(self) -> None:
        mask = self._state.session.selection_mask
        if mask is None:
            return
        doc = self._document
        layer = self._state.session.active_layer
        if not self._state.tool.overwrite_annotations:
            annotated = doc.get_other_annotations_mask(layer)
            mask = apply_overwrite_guard(mask, annotated)
        self._autolabel.begin_correction(doc.annotations)
        doc.annotate_mask(mask, layer)
        self._autolabel.end_correction(doc.annotations)
        self._state.session.selection_mask = None
        self._sync_selection_mask()
        self._post_annotation_commit(layer)

    def _post_annotation_commit(self, layer: int) -> None:
        """Persist, update metadata, sync viewer, and notify progress."""
        doc = self._document
        self._image_repo.save_annotations(doc, self._current_filename)
        if self._metadata:
            self._metadata.update_pixel_stats(
                doc.annotations, image_size=(doc.height, doc.width)
            )
        if self._time_tracker:
            self._time_tracker.change(layer)
        self._sync_annotation_overlay()
        self._notify_progress()

    def _cancel_tool(self) -> None:
        self._state.tool.is_drawing = False
        self._state.session.selection_mask = None
        self._sync_selection_mask()

    # ------------------------------------------------------------------
    # Viewer synchronisation helpers
    # ------------------------------------------------------------------

    def _sync_annotation_overlay(self) -> None:
        if self._document is None:
            return
        colors = [lc.color_rgb for lc in self._layer_configs]
        layer = self._state.session.active_layer
        show_others = self._state.view.show_other_layers
        rgba = build_annotation_rgba(
            self._document.annotations, colors, layer, show_others
        )
        self._viewer.set_annotation_overlay(rgba)

    def _sync_selection_mask(self) -> None:
        mask = self._state.session.selection_mask
        color = self._layer_configs[self._state.session.active_layer].color_rgb
        self._viewer.set_selection_mask(mask, color)

    # ------------------------------------------------------------------
    # Misc helpers
    # ------------------------------------------------------------------

    def _finalize_autolabel_session(self) -> None:
        self._autolabel.finalize_session()

    def _notify_progress(self) -> None:
        if self._document:
            progress = self._document.get_progress()
            for cb in self._on_progress_changed:
                cb(progress)
