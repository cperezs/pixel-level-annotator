"""Toolbar panel — all annotation tool controls in one widget.

Extracted from the monolithic main window so that the toolbar is
independently testable and the main window stays thin.

All user interactions are communicated to the caller through callbacks
registered with ``on_*`` methods.  The panel has no direct dependency on
the controller — it is the main window's job to wire the two together.
"""
from __future__ import annotations

from typing import Callable, Optional

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QLabel,
    QPushButton,
    QSlider,
    QSpinBox,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from domain.layer_config import LayerConfig


class ToolbarPanel(QWidget):
    """Left-side annotation toolbar.

    Builds all controls at construction time; subsequent state updates
    are applied via the ``set_*`` methods.  The caller registers callbacks
    with the ``on_*`` methods.
    """

    def __init__(
        self,
        layer_configs: list[LayerConfig],
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._layer_configs = layer_configs
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self._layout = layout

        # Deferred callback references (set by on_* methods or None).
        self._cb_tool_selected:            Optional[Callable[[str], None]] = None
        self._cb_layer_selected:           Optional[Callable[[int], None]] = None
        self._cb_autolabel_plugin_changed: Optional[Callable[[Optional[str]], None]] = None

        self._build_zoom_undo_section()
        self._build_separator()
        self._build_web_service_section()
        self._build_separator()
        self._build_tool_section()
        self._build_separator()
        self._build_layers_section(layer_configs)
        self._build_separator()
        self._build_autolabel_section()

    # ------------------------------------------------------------------
    # Callback registration
    # ------------------------------------------------------------------

    def on_zoom_in(self, cb: Callable) -> None:
        self._q_zoom_in_button.clicked.connect(cb)

    def on_zoom_out(self, cb: Callable) -> None:
        self._q_zoom_out_button.clicked.connect(cb)

    def on_undo(self, cb: Callable) -> None:
        self._q_undo_button.clicked.connect(cb)

    def on_tool_selected(self, cb: Callable[[str], None]) -> None:
        self._cb_tool_selected = cb

    def on_layer_selected(self, cb: Callable[[int], None]) -> None:
        self._cb_layer_selected = cb

    def on_pen_size_changed(self, cb: Callable[[int], None]) -> None:
        self._q_pen_spin.valueChanged.connect(cb)

    def on_threshold_changed(self, cb: Callable[[int], None]) -> None:
        self._q_threshold_slider.valueChanged.connect(cb)

    def on_auto_smooth_changed(self, cb: Callable[[bool], None]) -> None:
        self._q_autosmooth.stateChanged.connect(
            lambda: cb(self._q_autosmooth.isChecked())
        )

    def on_overwrite_changed(self, cb: Callable[[bool], None]) -> None:
        self._q_ignore_annotations.stateChanged.connect(
            lambda: cb(self._q_ignore_annotations.isChecked())
        )

    def on_fill_all_changed(self, cb: Callable[[bool], None]) -> None:
        self._q_fill_all.stateChanged.connect(
            lambda: cb(self._q_fill_all.isChecked())
        )

    def on_show_image_changed(self, cb: Callable[[bool], None]) -> None:
        self._q_show_image.stateChanged.connect(
            lambda: cb(self._q_show_image.isChecked())
        )

    def on_show_other_layers_changed(self, cb: Callable[[bool], None]) -> None:
        self._q_other_layers.stateChanged.connect(
            lambda: cb(self._q_other_layers.isChecked())
        )

    def on_show_missing_pixels_changed(self, cb: Callable[[bool], None]) -> None:
        self._q_missing_pixels_check.stateChanged.connect(
            lambda: cb(self._q_missing_pixels_check.isChecked())
        )

    def on_web_service_mode_changed(self, cb: Callable[[bool], None]) -> None:
        self._q_web_service_mode.stateChanged.connect(
            lambda: cb(self._q_web_service_mode.isChecked())
        )

    def on_submit_annotations(self, cb: Callable) -> None:
        self._q_submit_button.clicked.connect(cb)

    def on_cancel_annotations(self, cb: Callable) -> None:
        self._q_cancel_button.clicked.connect(cb)

    def on_autolabel_run(self, cb: Callable) -> None:
        self._q_autolabel_run_button.clicked.connect(cb)

    # ------------------------------------------------------------------
    # State update methods (called by the main window / controller)
    # ------------------------------------------------------------------

    def set_threshold(self, value: int) -> None:
        self._q_threshold_slider.blockSignals(True)
        self._q_threshold_slider.setValue(value)
        self._q_threshold_slider.blockSignals(False)

    def set_pen_size(self, size: int) -> None:
        self._q_pen_spin.blockSignals(True)
        self._q_pen_spin.setValue(size)
        self._q_pen_spin.blockSignals(False)

    def set_active_tool(self, tool: str) -> None:
        mapping = {
            "pen":      self._q_pen_button,
            "selector": self._q_selector_button,
            "fill":     self._q_fill_button,
        }
        btn = mapping.get(tool)
        if btn:
            btn.setChecked(True)
        self._update_tool_widget_states(tool)

    # TODO: this method is not called
    def set_active_layer(self, layer_index: int) -> None:
        for i, btn in enumerate(self._q_layer_buttons):
            lc = self._layer_configs[i]
            if i == layer_index:
                btn.setStyleSheet(
                    f"background-color: {lc.color_hex}; font-weight: bold;"
                )
                btn.setChecked(True)
            else:
                btn.setStyleSheet(
                    f"background-color: {lc.color_hex}; font-weight: normal;"
                )

    def set_web_service_ui(self, mode_active: bool, progress_complete: bool = False) -> None:
        """Update web-service-related widget visibility and enabled state."""
        self._q_submit_button.setVisible(mode_active)
        self._q_cancel_button.setVisible(mode_active)
        self._q_submit_button.setEnabled(mode_active and progress_complete)

    def refresh_autolabel_plugins(self, plugins) -> None:
        """Repopulate the autolabel combo with *plugins*."""
        self._q_autolabel_combo.blockSignals(True)
        self._q_autolabel_combo.clear()
        self._q_autolabel_combo.addItem("--Select model--", None)
        for plugin in plugins:
            self._q_autolabel_combo.addItem(plugin.display_name, plugin.id)
        self._q_autolabel_combo.setCurrentIndex(0)
        self._q_autolabel_combo.blockSignals(False)
        self._q_autolabel_run_button.setEnabled(False)

    def get_selected_plugin_id(self) -> Optional[str]:
        return self._q_autolabel_combo.currentData()

    # ------------------------------------------------------------------
    # Widget construction helpers
    # ------------------------------------------------------------------

    def _build_zoom_undo_section(self) -> None:
        self._q_zoom_in_button = QPushButton("Zoom In (Cmd +)")
        self._q_zoom_in_button.setShortcut("Ctrl++")
        self._q_zoom_out_button = QPushButton("Zoom Out (Cmd -)")
        self._q_zoom_out_button.setShortcut("Ctrl+-")
        self._q_undo_button = QPushButton("Undo (Cmd Z)")
        for w in (self._q_zoom_in_button, self._q_zoom_out_button, self._q_undo_button):
            w.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            self._layout.addWidget(w)

    def _build_web_service_section(self) -> None:
        self._q_web_service_mode = QCheckBox("Web Service Mode")
        self._q_web_service_mode.setChecked(False)
        self._q_web_service_mode.setFocusPolicy(Qt.FocusPolicy.NoFocus)

        self._q_submit_button = QPushButton("Submit Annotations")
        self._q_submit_button.setVisible(False)
        self._q_submit_button.setEnabled(False)
        self._q_submit_button.setFocusPolicy(Qt.FocusPolicy.NoFocus)

        self._q_cancel_button = QPushButton("Cancel Annotation")
        self._q_cancel_button.setVisible(False)
        self._q_cancel_button.setFocusPolicy(Qt.FocusPolicy.NoFocus)

        for w in (
            self._q_web_service_mode,
            self._q_submit_button,
            self._q_cancel_button,
        ):
            self._layout.addWidget(w)

    def _build_tool_section(self) -> None:
        tool_group = QButtonGroup(self)
        tool_group.setExclusive(True)

        self._q_pen_button      = QPushButton("&Pen")
        self._q_selector_button = QPushButton("&Selector")
        self._q_fill_button     = QPushButton("&Fill")

        for btn, shortcut, tool in [
            (self._q_pen_button,      "p", "pen"),
            (self._q_selector_button, "s", "selector"),
            (self._q_fill_button,     "f", "fill"),
        ]:
            btn.setCheckable(True)
            btn.setShortcut(shortcut)
            btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            btn.clicked.connect(
                lambda _checked=False, t=tool: (
                    self._cb_tool_selected(t) if self._cb_tool_selected else None
                )
            )
            tool_group.addButton(btn)

        self._q_pen_button.setChecked(True)

        self._q_ignore_annotations = QCheckBox("Over&write annotations")
        self._q_ignore_annotations.setShortcut("w")
        self._q_ignore_annotations.setFocusPolicy(Qt.FocusPolicy.NoFocus)

        # Overwrite checkbox — placed above Pen tool
        self._layout.addWidget(self._q_ignore_annotations)

        # Pen controls
        self._layout.addWidget(self._q_pen_button)
        self._q_pen_label = QLabel("Pen size")
        self._layout.addWidget(self._q_pen_label)
        self._q_pen_spin = QSpinBox()
        self._q_pen_spin.setMinimum(1)
        self._q_pen_spin.setMaximum(15)
        self._q_pen_spin.setValue(1)
        self._layout.addWidget(self._q_pen_spin)

        # Selector controls
        self._layout.addWidget(self._q_selector_button)
        self._q_threshold_label = QLabel("Threshold")
        self._layout.addWidget(self._q_threshold_label)
        self._q_threshold_slider = QSlider(Qt.Orientation.Horizontal)
        self._q_threshold_slider.setMinimum(1)
        self._q_threshold_slider.setMaximum(128)
        self._q_threshold_slider.setValue(32)
        self._q_threshold_slider.setTickInterval(16)
        self._q_threshold_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self._layout.addWidget(self._q_threshold_slider)

        self._q_autosmooth = QCheckBox("Auto-smooth")
        self._q_autosmooth.setChecked(True)
        self._q_autosmooth.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._layout.addWidget(self._q_autosmooth)

        # Fill controls
        self._layout.addWidget(self._q_fill_button)
        self._q_fill_all = QCheckBox("Fill all regions")
        self._q_fill_all.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._layout.addWidget(self._q_fill_all)

        # Initial visual state
        self._update_tool_widget_states("pen")

        # React to tool buttons to update internal widget enabled states.
        self._q_pen_button.clicked.connect(lambda: self._update_tool_widget_states("pen"))
        self._q_selector_button.clicked.connect(lambda: self._update_tool_widget_states("selector"))
        self._q_fill_button.clicked.connect(lambda: self._update_tool_widget_states("fill"))

    def _build_layers_section(self, layer_configs: list[LayerConfig]) -> None:
        btn_group = QButtonGroup(self)
        btn_group.setExclusive(True)
        self._q_layer_buttons: list[QPushButton] = []

        self._layout.addWidget(QLabel("Layers"))

        self._q_show_image = QCheckBox("Show &image")
        self._q_show_image.setShortcut("i")
        self._q_show_image.setChecked(True)
        self._q_show_image.setFocusPolicy(Qt.FocusPolicy.NoFocus)

        self._q_other_layers = QCheckBox("Show &other layers")
        self._q_other_layers.setChecked(True)
        self._q_other_layers.setShortcut("o")
        self._q_other_layers.setFocusPolicy(Qt.FocusPolicy.NoFocus)

        self._q_missing_pixels_check = QCheckBox("Show &missing pixels")
        self._q_missing_pixels_check.setShortcut("m")
        self._q_missing_pixels_check.setFocusPolicy(Qt.FocusPolicy.NoFocus)

        self._layout.addWidget(self._q_missing_pixels_check)
        self._layout.addWidget(self._q_show_image)
        self._layout.addWidget(self._q_other_layers)

        for i, lc in enumerate(layer_configs):
            label = f" ({i + 1})" if i < 9 else ""
            btn = QPushButton(f"{lc.name}{label}")
            btn.setCheckable(True)
            btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            btn.setStyleSheet(f"background-color: {lc.color_hex}; font-weight: normal;")
            if i < 9:
                btn.setShortcut(str(i + 1))
            btn.clicked.connect(
                lambda _checked=False, idx=i: (
                    self._cb_layer_selected(idx) if self._cb_layer_selected else None
                )
            )
            btn_group.addButton(btn)
            self._q_layer_buttons.append(btn)
            self._layout.addWidget(btn)

        if self._q_layer_buttons:
            self._q_layer_buttons[0].setChecked(True)

    def _build_autolabel_section(self) -> None:
        self._layout.addWidget(QLabel("Autolabeling"))
        self._q_autolabel_combo = QComboBox()
        self._q_autolabel_combo.addItem("--Select model--", None)
        self._q_autolabel_combo.currentIndexChanged.connect(self._on_autolabel_combo_changed)
        self._layout.addWidget(self._q_autolabel_combo)

        self._q_autolabel_run_button = QPushButton("Run")
        self._q_autolabel_run_button.setEnabled(False)
        self._q_autolabel_run_button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._layout.addWidget(self._q_autolabel_run_button)

    def _build_separator(self) -> None:
        sep = QToolBar()
        sep.addSeparator()
        self._layout.addWidget(sep)

    def _on_autolabel_combo_changed(self) -> None:
        plugin_id = self._q_autolabel_combo.currentData()
        self._q_autolabel_run_button.setEnabled(plugin_id is not None)
        if self._cb_autolabel_plugin_changed:
            self._cb_autolabel_plugin_changed(plugin_id)

    def _update_tool_widget_states(self, tool: str) -> None:
        pen_active      = tool == "pen"
        selector_active = tool == "selector"

        self._q_pen_label.setEnabled(pen_active)
        self._q_pen_spin.setEnabled(pen_active)
        self._q_threshold_label.setEnabled(selector_active)
        self._q_threshold_slider.setEnabled(selector_active)
        self._q_autosmooth.setEnabled(selector_active)
