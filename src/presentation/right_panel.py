"""Right sidebar panel — Workspace: layers, view options, autolabeling.

Provides layer management (selection, visibility toggle, lock toggle),
view options (show image, show other layers, show missing pixels), and
AI auto-labeling controls (plugin selection and run button).
"""
from __future__ import annotations

from typing import Callable, Optional

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from domain.layer_config import LayerConfig
from application.app_state import ToolbarState
from presentation.style import (
    PRIMARY,
    ON_PRIMARY,
    ON_SURFACE,
    ON_SURFACE_VARIANT,
    OUTLINE_VARIANT,
    SURFACE,
    SURFACE_BRIGHT,
    SURFACE_CONTAINER_HIGH,
    SURFACE_CONTAINER_HIGHEST,
    SURFACE_CONTAINER_LOWEST,
    SURFACE_VARIANT,
    FONT_SIZE_SM,
    FONT_SIZE_XS,
    SIDEBAR_WIDTH,
)


# ------------------------------------------------------------------
# Layer row widget
# ------------------------------------------------------------------

class _LayerRow(QFrame):
    """Single layer row with visibility, lock, name, and colored left border."""

    def __init__(
        self,
        index: int,
        config: LayerConfig,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._index = index
        self._config = config
        self._selected = False
        self._visible = True
        self._locked = False

        self._cb_selected: Optional[Callable[[int], None]] = None
        self._cb_visibility: Optional[Callable[[int, bool], None]] = None
        self._cb_lock: Optional[Callable[[int], None]] = None

        self.setFixedHeight(36)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setObjectName("LayerRow")

        layout = QHBoxLayout(self)
        layout.setContentsMargins(6, 0, 4, 0)
        layout.setSpacing(4)

        # Name label (left, stretches)
        label_text = f"{index + 1}. {config.name.capitalize()}"
        self._name_label = QLabel(label_text)
        self._name_label.setStyleSheet(
            f"color: {ON_SURFACE}; font-size: {FONT_SIZE_SM}px; "
            f"font-weight: 500;"
        )
        layout.addWidget(self._name_label, 1)

        # Shortcut (invisible)
        if index < 9:
            shortcut_btn = QPushButton()
            shortcut_btn.setShortcut(str(index + 1))
            shortcut_btn.setFixedSize(0, 0)
            shortcut_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            shortcut_btn.clicked.connect(lambda: self._fire_selected())
            layout.addWidget(shortcut_btn)

        # Visibility button (right side)
        self._vis_btn = QPushButton("\U0001F441")
        self._vis_btn.setFixedSize(28, 28)
        self._vis_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._vis_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._vis_btn.setToolTip("Toggle visibility")
        self._vis_btn.clicked.connect(self._toggle_visibility)
        layout.addWidget(self._vis_btn)

        # Lock button (right side)
        self._lock_btn = QPushButton("\U0001F512")
        self._lock_btn.setFixedSize(28, 28)
        self._lock_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._lock_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._lock_btn.setToolTip("Toggle lock")
        self._lock_btn.clicked.connect(self._toggle_lock)
        layout.addWidget(self._lock_btn)

        self._update_btn_styles()
        self._apply_style()

    def on_selected(self, cb: Callable[[int], None]):
        self._cb_selected = cb

    def on_visibility_changed(self, cb: Callable[[int, bool], None]):
        self._cb_visibility = cb

    def on_lock_toggled(self, cb: Callable[[int], None]):
        self._cb_lock = cb

    def set_selected(self, selected: bool):
        self._selected = selected
        self._apply_style()

    def set_visible(self, visible: bool):
        self._visible = visible
        self._update_btn_styles()
        self._apply_style()

    def set_locked(self, locked: bool):
        self._locked = locked
        self._update_btn_styles()
        self._apply_style()

    @property
    def is_locked(self) -> bool:
        return self._locked

    def mousePressEvent(self, event):
        self._fire_selected()

    def _fire_selected(self):
        if self._cb_selected:
            self._cb_selected(self._index)

    def _toggle_visibility(self):
        self._visible = not self._visible
        self._update_btn_styles()
        self._apply_style()
        if self._cb_visibility:
            self._cb_visibility(self._index, self._visible)

    def _toggle_lock(self):
        self._locked = not self._locked
        self._update_btn_styles()
        if self._cb_lock:
            self._cb_lock(self._index)

    def _btn_style(self, active: bool) -> str:
        bg = "rgba(161, 250, 255, 0.25)" if active else "transparent"
        hover_bg = "rgba(161, 250, 255, 0.40)" if active else "rgba(161, 250, 255, 0.15)"
        return (
            f"QPushButton {{ background: {bg}; border: none; "
            f"font-size: 16px; padding: 0; border-radius: 4px; }}"
            f"QPushButton:hover {{ background: {hover_bg}; }}"
        )

    def _update_btn_styles(self):
        self._vis_btn.setStyleSheet(self._btn_style(self._visible))
        self._lock_btn.setStyleSheet(self._btn_style(self._locked))

    def _apply_style(self):
        color = self._config.color_hex
        # Row frame: always transparent background, only the left border stripe
        self.setStyleSheet(f"""
            QFrame#LayerRow {{
                background-color: transparent;
                border-left: 4px solid {color};
                border-radius: 8px;
                border-top-left-radius: 0; border-bottom-left-radius: 0;
            }}
        """)
        if not self._visible:
            self._name_label.setStyleSheet(
                f"color: {ON_SURFACE_VARIANT}; font-size: {FONT_SIZE_SM}px;"
                f" font-weight: 400; background: transparent; border-radius: 4px;"
                f" padding: 1px 4px;"
            )
        elif self._selected:
            self._name_label.setStyleSheet(
                f"color: {ON_PRIMARY}; font-size: {FONT_SIZE_SM}px; font-weight: 700;"
                f" background-color: {PRIMARY}; border-radius: 4px; padding: 1px 4px;"
            )
        else:
            self._name_label.setStyleSheet(
                f"color: {ON_SURFACE}; font-size: {FONT_SIZE_SM}px; font-weight: 500;"
                f" background: transparent; border-radius: 4px; padding: 1px 4px;"
            )


# ------------------------------------------------------------------
# RightPanel
# ------------------------------------------------------------------

class RightPanel(QWidget):
    """Right sidebar with layers, view options, and auto-labeling."""

    def __init__(
        self,
        layer_configs: list[LayerConfig],
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._layer_configs = layer_configs
        self.setFixedWidth(SIDEBAR_WIDTH)
        self.setStyleSheet(f"background-color: {SURFACE_CONTAINER_HIGH};")

        self._cb_layer_selected: Optional[Callable[[int], None]] = None
        self._cb_autolabel_plugin_changed: Optional[Callable[[Optional[str]], None]] = None

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # Scrollable area for layers + view options
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setStyleSheet("QScrollArea { border: none; background: transparent; }")
        scroll_content = QWidget()
        self._layout = QVBoxLayout(scroll_content)
        self._layout.setContentsMargins(14, 16, 14, 16)
        self._layout.setSpacing(12)
        self._layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        scroll.setWidget(scroll_content)
        outer.addWidget(scroll, 1)

        # Sticky bottom area for autolabeling
        self._bottom_widget = QWidget()
        self._bottom_widget.setStyleSheet(f"background-color: {SURFACE_CONTAINER_HIGH};")
        self._bottom_layout = QVBoxLayout(self._bottom_widget)
        self._bottom_layout.setContentsMargins(14, 8, 14, 12)
        self._bottom_layout.setSpacing(8)
        outer.addWidget(self._bottom_widget, 0)

        self._build_workspace_header()
        self._build_layers_section(layer_configs)
        self._build_view_options()
        self._build_autolabel_section()
        self._layout.addStretch()

    # ------------------------------------------------------------------
    # Callback registration
    # ------------------------------------------------------------------

    def on_layer_selected(self, cb: Callable[[int], None]) -> None:
        self._cb_layer_selected = cb

    def on_show_image_changed(self, cb: Callable[[bool], None]) -> None:
        self._q_show_image.clicked.connect(lambda: cb(self._show_image_active))

    def on_show_missing_pixels_changed(self, cb: Callable[[bool], None]) -> None:
        self._q_missing_pixels.clicked.connect(lambda: cb(self._missing_pixels_active))

    def on_show_grid_changed(self, cb: Callable[[bool], None]) -> None:
        self._q_show_grid.clicked.connect(lambda: cb(self._grid_visible_active))

    def on_layer_visibility_changed(self, cb: Callable[[int, bool], None]) -> None:
        for row in self._layer_rows:
            row.on_visibility_changed(cb)

    def on_layer_lock_toggled(self, cb: Callable[[int], None]) -> None:
        for row in self._layer_rows:
            row.on_lock_toggled(cb)

    def on_autolabel_run(self, cb: Callable) -> None:
        self._q_autolabel_run_button.clicked.connect(cb)

    # ------------------------------------------------------------------
    # State updates
    # ------------------------------------------------------------------

    def set_active_layer(self, layer_index: int) -> None:
        for i, row in enumerate(self._layer_rows):
            row.set_selected(i == layer_index)

    def set_locked_layers(self, locked: set) -> None:
        for i, row in enumerate(self._layer_rows):
            row.set_locked(i in locked)

    def sync(self, state: ToolbarState) -> None:
        """Apply ToolbarState to the right panel."""
        self.set_active_layer(state.active_layer)
        self.set_locked_layers(state.locked_layers)
        for i, row in enumerate(self._layer_rows):
            row.set_visible(i not in state.hidden_layers)

        # View toggles
        self._show_image_active = state.show_image
        self._update_toggle_style(self._q_show_image, state.show_image)

        self._missing_pixels_active = state.show_missing_pixels
        self._update_toggle_style(self._q_missing_pixels, state.show_missing_pixels)

        self._grid_visible_active = state.show_grid
        self._update_toggle_style(self._q_show_grid, state.show_grid)

    def refresh_autolabel_plugins(self, plugins) -> None:
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
    # Widget construction
    # ------------------------------------------------------------------

    def _build_workspace_header(self) -> None:
        lbl = QLabel("Workspace")
        lbl.setStyleSheet(
            f"color: {PRIMARY}; font-size: {FONT_SIZE_SM}px; "
            f"font-weight: 700; letter-spacing: 1px;"
        )
        self._layout.addWidget(lbl)

    def _build_layers_section(self, layer_configs: list[LayerConfig]) -> None:
        # Layers header (simple text + count)
        h_row = QHBoxLayout()
        h_row.setContentsMargins(0, 0, 0, 4)
        title = QLabel("LAYERS")
        title.setStyleSheet(
            f"color: {ON_SURFACE_VARIANT}; font-size: {FONT_SIZE_XS}px; "
            f"font-weight: 700; letter-spacing: 1.5px;"
        )
        count = QLabel(str(len(layer_configs)))
        count.setStyleSheet(
            f"color: {ON_SURFACE_VARIANT}; font-size: {FONT_SIZE_XS}px; font-weight: 600;"
        )
        h_row.addWidget(title)
        h_row.addStretch()
        h_row.addWidget(count)
        self._layout.addLayout(h_row)

        # Layer rows
        self._layer_rows: list[_LayerRow] = []
        for i, lc in enumerate(layer_configs):
            row = _LayerRow(i, lc)
            row.on_selected(self._on_layer_row_selected)
            self._layer_rows.append(row)
            self._layout.addWidget(row)

        if self._layer_rows:
            self._layer_rows[0].set_selected(True)

    def _build_view_options(self) -> None:
        lbl = QLabel("VIEW OPTIONS")
        lbl.setStyleSheet(
            f"color: {ON_SURFACE_VARIANT}; font-size: {FONT_SIZE_XS}px; "
            f"font-weight: 700; letter-spacing: 1.5px; margin-top: 8px;"
        )
        self._layout.addWidget(lbl)

        self._show_image_active = True
        self._missing_pixels_active = False

        self._q_show_image = self._create_toggle_button("👁  Show Image (I)", True)
        self._q_show_image.setShortcut("i")
        self._q_show_image.clicked.connect(self._toggle_show_image)
        self._layout.addWidget(self._q_show_image)

        self._q_missing_pixels = self._create_toggle_button("⚠  Missing Pixels (M)", False)
        self._q_missing_pixels.setShortcut("m")
        self._q_missing_pixels.clicked.connect(self._toggle_missing_pixels)
        self._layout.addWidget(self._q_missing_pixels)

        self._grid_visible_active = True
        self._q_show_grid = self._create_toggle_button("⊞  Show Grid (G)", True)
        self._q_show_grid.setShortcut("g")
        self._q_show_grid.clicked.connect(self._toggle_show_grid)
        self._layout.addWidget(self._q_show_grid)

    def _build_autolabel_section(self) -> None:
        card = QFrame()
        card.setStyleSheet(f"""
            QFrame {{
                background-color: {SURFACE_CONTAINER_HIGHEST};
                border-radius: 12px;
                border: 1px solid rgba(72, 72, 72, 0.1);
            }}
        """)
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(14, 12, 14, 12)
        card_layout.setSpacing(10)

        # Header
        header_row = QHBoxLayout()
        icon = QLabel("⚡")
        icon.setStyleSheet(f"font-size: 14px; color: {PRIMARY};")
        title = QLabel("AI AUTO-LABELING")
        title.setStyleSheet(
            f"color: {PRIMARY}; font-size: {FONT_SIZE_XS}px; "
            f"font-weight: 700; letter-spacing: 1.5px;"
        )
        header_row.addWidget(icon)
        header_row.addWidget(title)
        header_row.addStretch()
        card_layout.addLayout(header_row)

        # Model selector
        self._q_autolabel_combo = QComboBox()
        self._q_autolabel_combo.addItem("--Select model--", None)
        self._q_autolabel_combo.currentIndexChanged.connect(self._on_autolabel_combo_changed)
        card_layout.addWidget(self._q_autolabel_combo)

        # Run button
        self._q_autolabel_run_button = QPushButton("RUN")
        self._q_autolabel_run_button.setEnabled(False)
        self._q_autolabel_run_button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._q_autolabel_run_button.setMinimumHeight(36)
        self._q_autolabel_run_button.setStyleSheet(f"""
            QPushButton {{
                background: {PRIMARY};
                color: {ON_PRIMARY};
                font-weight: 700;
                border-radius: 8px;
                font-size: {FONT_SIZE_SM}px;
                letter-spacing: 2px;
            }}
            QPushButton:hover {{ background: #b5fcff; }}
            QPushButton:pressed {{ background: #00e5ee; }}
            QPushButton:disabled {{
                background: {SURFACE_CONTAINER_HIGHEST};
                color: {OUTLINE_VARIANT};
            }}
        """)
        card_layout.addWidget(self._q_autolabel_run_button)

        self._bottom_layout.addWidget(card)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _on_layer_row_selected(self, index: int) -> None:
        for i, row in enumerate(self._layer_rows):
            row.set_selected(i == index)
        if self._cb_layer_selected:
            self._cb_layer_selected(index)

    def _on_autolabel_combo_changed(self) -> None:
        plugin_id = self._q_autolabel_combo.currentData()
        self._q_autolabel_run_button.setEnabled(plugin_id is not None)

    def _create_toggle_button(self, text: str, active: bool) -> QPushButton:
        btn = QPushButton(text)
        btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        btn.setMinimumHeight(34)
        self._update_toggle_style(btn, active)
        return btn

    def _update_toggle_style(self, btn: QPushButton, active: bool) -> None:
        if active:
            btn.setStyleSheet(f"""
                QPushButton {{
                    background: rgba(161, 250, 255, 0.1);
                    color: {PRIMARY};
                    border: 1px solid rgba(161, 250, 255, 0.2);
                    border-radius: 8px;
                    text-align: left; padding: 6px 12px;
                    font-size: {FONT_SIZE_SM}px; font-weight: 600;
                    letter-spacing: 0.5px;
                }}
            """)
        else:
            btn.setStyleSheet(f"""
                QPushButton {{
                    background: {SURFACE_CONTAINER_HIGHEST};
                    color: {ON_SURFACE_VARIANT};
                    border: none; border-radius: 8px;
                    text-align: left; padding: 6px 12px;
                    font-size: {FONT_SIZE_SM}px; font-weight: 600;
                    letter-spacing: 0.5px;
                }}
                QPushButton:hover {{ color: {ON_SURFACE}; }}
            """)

    def _toggle_show_image(self):
        self._show_image_active = not self._show_image_active
        self._update_toggle_style(self._q_show_image, self._show_image_active)

    def _toggle_missing_pixels(self):
        self._missing_pixels_active = not self._missing_pixels_active
        self._update_toggle_style(self._q_missing_pixels, self._missing_pixels_active)

    def _toggle_show_grid(self):
        self._grid_visible_active = not self._grid_visible_active
        self._update_toggle_style(self._q_show_grid, self._grid_visible_active)
