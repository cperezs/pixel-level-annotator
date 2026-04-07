"""Main application window — redesigned layout.

Layout (top-to-bottom, left-to-right):
    ┌──────────────────────────────────────────────────────┐
    │                   Top Navigation Bar                  │
    ├─────────┬──────────┬──────────────────┬──────────────┤
    │  Left   │ Gallery  │                  │    Right     │
    │ Toolbar │ (toggle) │     Canvas       │    Panel     │
    │         │          │                  │              │
    ├─────────┴──────────┴──────────────────┴──────────────┤
    │                     Status Bar                        │
    └──────────────────────────────────────────────────────┘

This module is intentionally thin.  Its only responsibilities are:
- Construct the widget tree.
- Create infrastructure and controller objects.
- Wire callbacks between panels and controller.
- React to controller output callbacks.
"""
from __future__ import annotations

import logging
from typing import Optional

from PyQt6.QtCore import Qt, QThread, QTimer, pyqtSignal
from PyQt6.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from domain.layer_config import LayerConfig, read_layers_file
from infrastructure.image_repository import ImageRepository
from application.annotator_controller import AnnotatorController
from presentation.toolbar_panel import ToolbarPanel
from presentation.right_panel import RightPanel
from presentation.gallery_panel import GalleryPanel
from presentation.status_bar import StatusBar
from presentation.style import (
    GLOBAL_STYLESHEET,
    PRIMARY,
    ON_SURFACE,
    ON_SURFACE_VARIANT,
    OUTLINE_VARIANT,
    SURFACE,
    SURFACE_CONTAINER_HIGH,
    SURFACE_CONTAINER_HIGHEST,
    SURFACE_CONTAINER_LOW,
    SURFACE_BRIGHT,
    FONT_SIZE_SM,
    FONT_SIZE_XS,
    TOPBAR_HEIGHT,
)
from infrastructure.webservice import WebService

logger = logging.getLogger(__name__)


class _AutolabelWorker(QThread):
    """Runs the autolabel plugin on a background thread."""
    finished = pyqtSignal(object)  # emits error string or None

    def __init__(self, controller, plugin_id: str, parent=None):
        super().__init__(parent)
        self._controller = controller
        self._plugin_id = plugin_id

    def run(self):
        error = self._controller.run_autolabel(self._plugin_id)
        self.finished.emit(error)


class MainWindow(QMainWindow):
    """Top-level Qt window for the Pixel Annotation Tool."""

    def __init__(self, viewer_class=None) -> None:
        super().__init__()
        self.setWindowTitle("PixelLabeler — Professional Annotation Suite")
        self.setGeometry(100, 100, 1440, 900)

        # Apply global stylesheet
        self.setStyleSheet(GLOBAL_STYLESHEET)

        layer_configs: list[LayerConfig] = read_layers_file()
        self._current_web_request = None

        # Infrastructure
        self._image_repo = ImageRepository()

        # Viewer
        if viewer_class is None:
            from viewer.gl_viewer import GLImageAnnotationViewer
            viewer_class = GLImageAnnotationViewer
        self._viewer = viewer_class()

        # Application controller
        self._controller = AnnotatorController(self._viewer, layer_configs, self._image_repo)
        self._controller.on_progress_changed(self._on_progress_changed)
        self._controller.on_web_service_image_ready(self._on_web_service_image_ready)
        self._controller.on_status_changed(self._on_status_changed)

        # Panels
        self._toolbar = ToolbarPanel(layer_configs)
        self._right_panel = RightPanel(layer_configs)
        self._gallery = GalleryPanel()
        self._gallery.setVisible(False)
        self._status_bar = StatusBar()

        # Progress tracking
        self._progress_value = 0

        # Layout
        self._build_layout()

        # Wire all callbacks
        self._wire_toolbar()
        self._wire_right_panel()
        self._wire_gallery()

        # Web service
        self._web_service = WebService(self)

        # Initial data
        self._populate_gallery()
        plugins = self._controller.autolabel_service.get_compatible_plugins()
        self._right_panel.refresh_autolabel_plugins(plugins)

        # Load first image
        filenames = self._image_repo.list_images()
        if filenames:
            self._controller.load_image(filenames[0])
            self._gallery.set_current_filename(filenames[0])
            self._update_status()

        # self.showMaximized()
        QTimer.singleShot(0, self._viewer.setFocus)

        # Return keyboard focus to the viewer after any sidebar interaction.
        app = QApplication.instance()
        if app is not None:
            app.focusChanged.connect(self._on_focus_changed)

    # ------------------------------------------------------------------
    # Web service integration
    # ------------------------------------------------------------------

    def load_web_service_image(self, image_path: str, request) -> None:
        self._current_web_request = request
        self._controller.load_web_service_image(image_path, request)

    # ------------------------------------------------------------------
    # Qt lifecycle
    # ------------------------------------------------------------------

    def closeEvent(self, event) -> None:
        self._controller.close()
        event.accept()

    # ------------------------------------------------------------------
    # Layout construction
    # ------------------------------------------------------------------

    def _build_layout(self) -> None:
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        root = QVBoxLayout(main_widget)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # Top bar
        root.addWidget(self._build_top_bar())

        # Middle area: toolbar | gallery? | canvas | right panel
        middle = QHBoxLayout()
        middle.setContentsMargins(0, 0, 0, 0)
        middle.setSpacing(0)

        middle.addWidget(self._toolbar, 0)
        middle.addWidget(self._gallery, 0)

        # Canvas wrapper
        canvas_wrapper = QWidget()
        canvas_wrapper.setStyleSheet(f"background-color: {SURFACE_CONTAINER_LOW};")
        canvas_layout = QVBoxLayout(canvas_wrapper)
        canvas_layout.setContentsMargins(0, 0, 0, 0)
        canvas_layout.addWidget(self._viewer, 1)
        middle.addWidget(canvas_wrapper, 1)
        self._canvas_wrapper = canvas_wrapper

        middle.addWidget(self._right_panel, 0)

        root.addLayout(middle, 1)

        # Status bar
        root.addWidget(self._status_bar)

    def _build_top_bar(self) -> QWidget:
        """Build the top navigation bar."""
        bar = QWidget()
        bar.setFixedHeight(TOPBAR_HEIGHT)
        bar.setStyleSheet(f"background-color: {SURFACE};")

        layout = QHBoxLayout(bar)
        layout.setContentsMargins(14, 0, 14, 0)
        layout.setSpacing(0)

        # Left: Brand + Progress
        left = QHBoxLayout()
        left.setSpacing(16)

        brand = QLabel("PixelLabeler")
        brand.setStyleSheet(
            f"color: {PRIMARY}; font-size: 16px; font-weight: 700; "
            f"letter-spacing: -0.5px;"
        )
        left.addWidget(brand)

        # Progress bar
        progress_container = QWidget()
        progress_container.setFixedWidth(180)
        p_layout = QVBoxLayout(progress_container)
        p_layout.setContentsMargins(0, 8, 0, 8)
        p_layout.setSpacing(2)

        p_header = QHBoxLayout()
        p_label = QLabel("ANNOTATED")
        p_label.setStyleSheet(
            f"color: {ON_SURFACE_VARIANT}; font-size: {FONT_SIZE_XS}px; "
            f"font-weight: 600; letter-spacing: 1px;"
        )
        self._q_progress_pct = QLabel("0%")
        self._q_progress_pct.setStyleSheet(
            f"color: {PRIMARY}; font-size: {FONT_SIZE_XS}px; font-weight: 700;"
        )
        p_header.addWidget(p_label)
        p_header.addStretch()
        p_header.addWidget(self._q_progress_pct)
        p_layout.addLayout(p_header)

        # Progress bar track
        self._q_progress_track = QWidget()
        self._q_progress_track.setFixedHeight(6)
        self._q_progress_track.setStyleSheet(
            f"background-color: {SURFACE_CONTAINER_HIGHEST}; border-radius: 3px;"
        )

        self._q_progress_fill = QWidget(self._q_progress_track)
        self._q_progress_fill.setFixedHeight(6)
        self._q_progress_fill.setStyleSheet(
            f"background-color: {PRIMARY}; border-radius: 3px;"
        )
        self._q_progress_fill.setGeometry(0, 0, 0, 6)

        p_layout.addWidget(self._q_progress_track)
        left.addWidget(progress_container)

        layout.addLayout(left)
        layout.addStretch()

        # Right: Undo, Redo, Zoom controls
        right = QHBoxLayout()
        right.setSpacing(2)

        self._q_undo_btn = self._topbar_button("↶", "Undo (Ctrl+Z)")
        right.addWidget(self._q_undo_btn)

        sep = QLabel()
        sep.setFixedSize(1, 16)
        sep.setStyleSheet(f"background-color: rgba(72, 72, 72, 0.2);")
        right.addWidget(sep)

        self._q_zoom_label = QLabel("100%")
        self._q_zoom_label.setStyleSheet(
            f"color: {PRIMARY}; font-size: {FONT_SIZE_SM}px; "
            f"font-weight: 700; padding: 0 8px; letter-spacing: -0.5px;"
        )
        self._q_zoom_label.setMinimumWidth(48)
        self._q_zoom_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        right.addWidget(self._q_zoom_label)

        self._q_zoom_in_btn = self._topbar_button("+", "Zoom In")
        self._q_zoom_in_btn.setShortcut("Ctrl++")
        right.addWidget(self._q_zoom_in_btn)

        self._q_zoom_out_btn = self._topbar_button("−", "Zoom Out")
        self._q_zoom_out_btn.setShortcut("Ctrl+-")
        right.addWidget(self._q_zoom_out_btn)

        layout.addLayout(right)
        return bar

    @staticmethod
    def _topbar_button(text: str, tooltip: str) -> QPushButton:
        btn = QPushButton(text)
        btn.setToolTip(tooltip)
        btn.setFixedSize(32, 32)
        btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        btn.setStyleSheet(f"""
            QPushButton {{
                background: transparent; border: none;
                color: {ON_SURFACE_VARIANT}; font-size: 16px;
                border-radius: 4px;
            }}
            QPushButton:hover {{ background: {SURFACE_CONTAINER_HIGHEST}; color: {ON_SURFACE}; }}
            QPushButton:pressed {{ background: {SURFACE_BRIGHT}; }}
        """)
        return btn

    # ------------------------------------------------------------------
    # Wiring
    # ------------------------------------------------------------------

    def _wire_toolbar(self) -> None:
        ctrl = self._controller
        tb = self._toolbar

        tb.on_tool_selected(ctrl.select_tool)
        tb.on_pen_size_changed(ctrl.set_pen_size)
        tb.on_eraser_size_changed(ctrl.set_eraser_size)
        tb.on_threshold_changed(ctrl.set_selector_threshold)
        tb.on_auto_smooth_changed(ctrl.set_selector_auto_smooth)
        tb.on_fill_all_changed(ctrl.set_fill_all)
        tb.on_gallery_clicked(self._toggle_gallery)
        tb.on_erase_all_clicked(self._on_erase_all)
        tb.on_web_service_mode_changed(self._toggle_web_service_mode)
        tb.on_submit_annotations(self._cb_submit_annotations)
        tb.on_cancel_annotations(self._cb_cancel_annotations)

        # Top bar buttons
        self._q_undo_btn.clicked.connect(ctrl.undo)
        self._q_zoom_in_btn.clicked.connect(lambda: ctrl.zoom_in())
        self._q_zoom_out_btn.clicked.connect(lambda: ctrl.zoom_out())

        ctrl.on_toolbar_state_changed(self._sync_toolbar)

    def _wire_right_panel(self) -> None:
        ctrl = self._controller
        rp = self._right_panel

        rp.on_layer_selected(ctrl.set_active_layer)
        rp.on_layer_visibility_changed(ctrl.toggle_layer_visibility)
        rp.on_show_image_changed(ctrl.toggle_show_image)
        rp.on_show_missing_pixels_changed(ctrl.toggle_show_missing_pixels)
        rp.on_show_grid_changed(ctrl.toggle_show_grid)
        rp.on_layer_lock_toggled(ctrl.toggle_layer_lock)
        rp.on_autolabel_run(self._cb_run_autolabel)

    def _wire_gallery(self) -> None:
        self._gallery.on_image_selected(self._on_gallery_image_selected)

    # ------------------------------------------------------------------
    # Sync & updates
    # ------------------------------------------------------------------

    def _sync_toolbar(self) -> None:
        state = self._controller.toolbar_state
        self._toolbar.sync(state)
        self._right_panel.sync(state)
        self._q_zoom_label.setText(f"{self._controller.get_zoom_percent()}%")

    def _on_progress_changed(self, progress: int) -> None:
        self._progress_value = progress
        self._q_progress_pct.setText(f"{progress}%")

        # Update progress fill bar width
        track_w = self._q_progress_track.width()
        fill_w = int(track_w * progress / 100) if track_w > 0 else 0
        self._q_progress_fill.setGeometry(0, 0, fill_w, 6)

        if self._current_web_request:
            self._toolbar.set_web_service_ui(
                mode_active=True,
                progress_complete=(progress == 100),
            )

    def _on_web_service_image_ready(self, filename: str, request) -> None:
        self._current_web_request = request
        self._toolbar.set_web_service_ui(mode_active=True, progress_complete=False)

    def _on_status_changed(self) -> None:
        self._update_status()

    def _update_status(self) -> None:
        ctrl = self._controller
        self._status_bar.set_filename(ctrl.current_filename)
        w, h = ctrl.image_dimensions
        if w > 0 and h > 0:
            self._status_bar.set_dimensions(w, h)
        pos = ctrl.mouse_position
        if pos:
            self._status_bar.set_coordinates(pos[0], pos[1])
        else:
            self._status_bar.set_coordinates(None, None)
        self._status_bar.set_active_tool(ctrl.active_tool_name)

    # ------------------------------------------------------------------
    # Gallery
    # ------------------------------------------------------------------

    def _on_erase_all(self) -> None:
        """Ask for confirmation, then erase all annotations on the current image."""
        if not self._controller.current_filename:
            return
        answer = QMessageBox.question(
            self,
            "Erase All Annotations",
            "This will permanently erase <b>all annotations</b> on the current image.<br>"
            "The action can be undone with Ctrl+Z.<br><br>"
            "Are you sure?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if answer == QMessageBox.StandardButton.Yes:
            self._controller.erase_all()

    def _toggle_gallery(self) -> None:
        visible = not self._gallery.isVisible()
        self._gallery.setVisible(visible)
        if visible:
            # populate() is a no-op when the file list hasn't changed, so it's
            # safe to call every time — it only does real work on the first open
            # or when a new image is added to the repo.
            self._populate_gallery()

    def _populate_gallery(self) -> None:
        filenames = self._image_repo.list_images()
        self._gallery.populate(filenames, self._image_repo._images_dir)
        # Always refresh the current-image highlight (cheap operation).
        if self._controller.current_filename:
            self._gallery.set_current_filename(self._controller.current_filename)

    def _on_gallery_image_selected(self, filename: str) -> None:
        self._controller.load_image(filename)
        self._gallery.set_current_filename(filename)
        self._update_status()

    # ------------------------------------------------------------------
    # Toolbar action handlers
    # ------------------------------------------------------------------

    def _toggle_web_service_mode(self, enabled: bool) -> None:
        if enabled:
            self._web_service.start_server()
            self._gallery.setVisible(False)
        else:
            self._web_service.stop_server()
            self._toolbar.set_web_service_ui(mode_active=False)
            self._current_web_request = None

    def _cb_run_autolabel(self) -> None:
        plugin_id = self._right_panel.get_selected_plugin_id()
        if not plugin_id:
            return

        self._show_busy_overlay("Running AI model…")

        self._autolabel_worker = _AutolabelWorker(self._controller, plugin_id, self)
        self._autolabel_worker.finished.connect(self._on_autolabel_finished)
        self._autolabel_worker.start()

    def _on_autolabel_finished(self, error) -> None:
        self._hide_busy_overlay()
        if error:
            QMessageBox.warning(self, "Autolabel Error", f"Plugin failed:\n{error}")
        else:
            self._controller.finalize_autolabel()
            plugins = self._controller.autolabel_service.get_compatible_plugins()
            self._right_panel.refresh_autolabel_plugins(plugins)

    def _show_busy_overlay(self, message: str = "Processing…") -> None:
        """Cover the entire window with a translucent blocking overlay."""
        overlay = QWidget(self)
        overlay.setObjectName("BusyOverlay")
        overlay.setStyleSheet("""
            QWidget#BusyOverlay {
                background-color: rgba(0, 0, 0, 0.65);
            }
        """)
        overlay.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, False)
        overlay.setGeometry(self.rect())
        overlay.raise_()

        lbl = QLabel(message, overlay)
        lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl.setStyleSheet(
            "color: #a1faff; font-size: 16px; font-weight: 700;"
            " background: transparent;"
        )
        lbl.setGeometry(0, 0, overlay.width(), overlay.height())
        lbl.raise_()

        overlay.show()
        self._busy_overlay = overlay
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)

    def _hide_busy_overlay(self) -> None:
        if hasattr(self, "_busy_overlay") and self._busy_overlay:
            self._busy_overlay.deleteLater()
            self._busy_overlay = None
        QApplication.restoreOverrideCursor()

    def _cb_submit_annotations(self) -> None:
        if not self._current_web_request:
            return
        doc = self._controller.document
        if not doc:
            return
        import numpy as np
        annotation_images = {
            lc.name: (
                doc.annotations[i]
                if i < doc.num_layers
                else np.zeros((doc.height, doc.width), dtype=np.uint8)
            )
            for i, lc in enumerate(self._controller.layer_configs)
        }
        success = self._web_service.submit_annotations(annotation_images)
        if success:
            self._toolbar.set_web_service_ui(mode_active=True, progress_complete=False)
            self._current_web_request = None
        else:
            logger.error("Failed to submit annotations")

    def _cb_cancel_annotations(self) -> None:
        if not self._current_web_request:
            return
        self._web_service.cancel_current_request()
        self._toolbar.set_web_service_ui(mode_active=True, progress_complete=False)
        self._current_web_request = None

    # ------------------------------------------------------------------
    # Focus management
    # ------------------------------------------------------------------

    def _on_focus_changed(self, old, new) -> None:
        """Redirect focus to the viewer whenever a sidebar panel widget gains it.

        This ensures keyboard shortcuts (zoom, tool keys, etc.) keep working
        immediately after the user clicks any toolbar or right-panel control.
        Widgets inside the panels have ``NoFocus`` where possible, but scroll
        areas and container widgets can still receive focus; this handles all
        such cases in one place.
        """
        if new is None:
            return
        w = new
        while w is not None:
            if w is self._toolbar or w is self._right_panel:
                QTimer.singleShot(0, self._viewer.setFocus)
                return
            w = w.parent()

    # ------------------------------------------------------------------
    # Resize event — update progress bar fill width
    # ------------------------------------------------------------------

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        QTimer.singleShot(0, lambda: self._on_progress_changed(self._progress_value))
