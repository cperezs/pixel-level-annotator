"""Main application window.

This module is intentionally thin.  Its only responsibilities are:
- Construct the widget tree (toolbar, viewer, image list, progress label).
- Create the infrastructure and controller objects.
- Wire toolbar callbacks to controller method calls.
- React to controller output callbacks (progress, web-service signals).
- Host the web service.

No domain logic, annotation computations, or rendering details live here.
"""
from __future__ import annotations

import logging
from typing import Optional

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QListWidget,
    QMainWindow,
    QMessageBox,
    QVBoxLayout,
    QWidget,
)

from domain.layer_config import LayerConfig, read_layers_file
from infrastructure.image_repository import ImageRepository
from application.annotator_controller import AnnotatorController
# from viewer.qt_viewer import QtImageAnnotationViewer as ImageAnnotationViewer
from viewer.gl_viewer import GLImageAnnotationViewer as ImageAnnotationViewer
from presentation.toolbar_panel import ToolbarPanel
from infrastructure.webservice import WebService

logger = logging.getLogger(__name__)


class MainWindow(QMainWindow):
    """Top-level Qt window for the Pixel Annotation Tool."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Pixel Annotation Tool")
        self.setGeometry(100, 100, 1024, 768)

        layer_configs: list[LayerConfig] = read_layers_file()
        self._current_web_request = None

        # Infrastructure
        image_repo = ImageRepository()

        # Viewer (Qt backend)
        self._viewer = ImageAnnotationViewer()

        # Application controller
        self._controller = AnnotatorController(self._viewer, layer_configs, image_repo)
        self._controller.on_progress_changed(self._on_progress_changed)
        self._controller.on_web_service_image_ready(self._on_web_service_image_ready)

        # Toolbar panel
        self._toolbar = ToolbarPanel(layer_configs)

        # Image list and progress label
        self._q_image_label = QLabel("Images")
        self._q_image_list = QListWidget()
        self._q_image_list.itemClicked.connect(
            lambda item: self._controller.load_image(item.text())
        )
        self._q_progress_bar = QLabel("0%")
        self._q_progress_bar.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._q_progress_bar.setFixedHeight(20)

        # Layout
        self._build_layout()

        # Wire toolbar actions to controller
        self._wire_toolbar()

        # Web service
        self._web_service = WebService(self)

        # Initial data
        self._populate_image_list(image_repo)
        plugins = self._controller.autolabel_service.get_compatible_plugins()
        self._toolbar.refresh_autolabel_plugins(plugins)

        if self._q_image_list.count() > 0:
            self._q_image_list.setCurrentRow(0)
            first_image = self._q_image_list.currentItem().text()
            self._controller.load_image(first_image)

        self.showMaximized()
        QTimer.singleShot(0, self._viewer.setFocus)

    # ------------------------------------------------------------------
    # Web service integration (called by WebService, not the controller)
    # ------------------------------------------------------------------

    def load_web_service_image(self, image_path: str, request) -> None:
        """Entry point for the FastAPI web service."""
        self._current_web_request = request
        self._controller.load_web_service_image(image_path, request)

    # ------------------------------------------------------------------
    # Qt lifecycle
    # ------------------------------------------------------------------

    def closeEvent(self, event) -> None:  # noqa: N802
        self._controller.close()
        event.accept()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_layout(self) -> None:
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)

        side_layout = QVBoxLayout()
        side_layout.addWidget(self._toolbar)
        side_layout.addWidget(self._q_image_label)
        side_layout.addWidget(self._q_image_list)

        image_layout = QVBoxLayout()
        image_layout.addWidget(self._q_progress_bar, 0)
        image_layout.addWidget(self._viewer, 1)

        main_layout.addLayout(side_layout, 1)
        main_layout.addLayout(image_layout, 10)

    def _wire_toolbar(self) -> None:
        ctrl = self._controller
        tb = self._toolbar

        tb.on_zoom_in(ctrl.zoom_in)
        tb.on_zoom_out(ctrl.zoom_out)
        tb.on_undo(ctrl.undo)
        tb.on_tool_selected(ctrl.select_tool)
        tb.on_layer_selected(ctrl.set_active_layer)
        tb.on_pen_size_changed(ctrl.set_pen_size)
        tb.on_threshold_changed(ctrl.set_selector_threshold)
        tb.on_auto_smooth_changed(ctrl.set_selector_auto_smooth)
        tb.on_overwrite_changed(ctrl.set_overwrite_annotations)
        tb.on_fill_all_changed(ctrl.set_fill_all)
        tb.on_show_image_changed(ctrl.toggle_show_image)
        tb.on_show_other_layers_changed(ctrl.toggle_show_other_layers)
        tb.on_show_missing_pixels_changed(ctrl.toggle_show_missing_pixels)
        tb.on_web_service_mode_changed(self._toggle_web_service_mode)
        tb.on_submit_annotations(self._cb_submit_annotations)
        tb.on_cancel_annotations(self._cb_cancel_annotations)
        tb.on_autolabel_run(self._cb_run_autolabel)

    def _populate_image_list(self, image_repo: ImageRepository) -> None:
        for filename in image_repo.list_images():
            self._q_image_list.addItem(filename)

    # ------------------------------------------------------------------
    # Controller callbacks
    # ------------------------------------------------------------------

    def _on_progress_changed(self, progress: int) -> None:
        self._q_progress_bar.setText(f"{progress}%")
        if self._current_web_request:
            self._toolbar.set_web_service_ui(
                mode_active=True,
                progress_complete=(progress == 100),
            )

    def _on_web_service_image_ready(self, filename: str, request) -> None:
        self._current_web_request = request
        self._toolbar.set_web_service_ui(mode_active=True, progress_complete=False)

    # ------------------------------------------------------------------
    # Toolbar action handlers
    # ------------------------------------------------------------------

    def _toggle_web_service_mode(self, enabled: bool) -> None:
        if enabled:
            self._web_service.start_server()
            self._q_image_list.setVisible(False)
            self._q_image_label.setVisible(False)
        else:
            self._web_service.stop_server()
            self._q_image_list.setVisible(True)
            self._q_image_label.setVisible(True)
            self._q_image_label.setText("Images")
            self._toolbar.set_web_service_ui(mode_active=False)
            self._current_web_request = None

    def _cb_run_autolabel(self) -> None:
        plugin_id = self._toolbar.get_selected_plugin_id()
        if not plugin_id:
            return
        error = self._controller.run_autolabel(plugin_id)
        if error:
            QMessageBox.warning(self, "Autolabel Error", f"Plugin failed:\n{error}")
        else:
            # Refresh plugin list after a successful run.
            plugins = self._controller.autolabel_service.get_compatible_plugins()
            self._toolbar.refresh_autolabel_plugins(plugins)

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
