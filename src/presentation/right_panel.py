"""Right sidebar panel — Workspace: layers, view options, autolabeling.

Provides layer management (selection, visibility toggle, lock toggle),
view options (show image, show other layers, show missing pixels), and
AI auto-labeling controls (plugin selection and run button).
"""
from __future__ import annotations

from typing import Callable, Optional

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QRadioButton,
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
# Layer mapping dialog
# ------------------------------------------------------------------

# helper -----------------------------------------------------------
_ITEM_HEIGHT = 28  # px — height of one combo item in the popup list


def _fix_combo_scroll(combo: QComboBox) -> None:
    """Force the popup list to show all items without a scrollbar.

    ``setMaxVisibleItems`` is ignored on macOS (native style overrides it).
    Setting a fixed height on the underlying view is the reliable cross-
    platform solution.
    """
    n = combo.count()
    combo.setMaxVisibleItems(n)
    combo.view().setFixedHeight(n * _ITEM_HEIGHT + 8)  # 8px vertical padding
    combo.view().setSizeAdjustPolicy(
        QAbstractItemView.SizeAdjustPolicy.AdjustToContents
    )


class LayerMappingDialog(QDialog):
    """Modal dialog for configuring the plugin-to-app layer mapping and the
    conflict resolution strategy used when running the AI model.
    """

    _RADIO_STYLE = (
        f"QRadioButton {{ color: {ON_SURFACE}; font-size: {FONT_SIZE_SM}px; }}"
        f"QRadioButton::indicator {{ width: 14px; height: 14px; }}"
    )

    def __init__(
        self,
        plugin_display_name: str,
        plugin_layers: list[str],
        app_layers: list[str],
        current_mapping: dict[str, str],
        current_strategy: str = "argmax",
        current_priorities: dict[str, int] | None = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(f"Model Configuration — {plugin_display_name}")
        self.setModal(True)
        self.setMinimumWidth(480)
        self.setStyleSheet(f"""
            QDialog {{
                background-color: {SURFACE_CONTAINER_HIGH};
            }}
            QComboBox {{
                background-color: {SURFACE_CONTAINER_HIGHEST};
                color: {ON_SURFACE};
                border: 1px solid {OUTLINE_VARIANT};
                border-radius: 6px;
                padding: 4px 28px 4px 10px;
                font-size: {FONT_SIZE_SM}px;
                min-height: 28px;
            }}
            QComboBox::drop-down {{
                subcontrol-origin: padding;
                subcontrol-position: right center;
                width: 22px;
                border: none;
            }}
            QComboBox::down-arrow {{
                width: 10px;
                height: 10px;
            }}
            QComboBox QAbstractItemView {{
                background-color: {SURFACE_CONTAINER_HIGHEST};
                color: {ON_SURFACE};
                selection-background-color: {PRIMARY};
                selection-color: {ON_PRIMARY};
                padding: 4px;
                border: 1px solid {OUTLINE_VARIANT};
                border-radius: 4px;
                outline: none;
            }}
        """)

        self._plugin_layers = list(plugin_layers)
        self._combos: dict[str, QComboBox] = {}
        self._prio_combos: dict[str, QComboBox] = {}
        self._blocking = False  # re-entrancy guard for swap logic
        if current_priorities is None:
            current_priorities = {}

        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(14)

        # ---- Conflict resolution strategy (radio buttons) ----------------
        strategy_lbl = QLabel("CONFLICT RESOLUTION STRATEGY")
        strategy_lbl.setStyleSheet(
            f"color: {ON_SURFACE_VARIANT}; font-size: {FONT_SIZE_XS}px; "
            f"font-weight: 700; letter-spacing: 1.2px;"
        )
        layout.addWidget(strategy_lbl)

        self._rb_argmax = QRadioButton("Highest model confidence")
        self._rb_argmax.setStyleSheet(self._RADIO_STYLE)
        self._rb_priority = QRadioButton("Layer priority")
        self._rb_priority.setStyleSheet(self._RADIO_STYLE)

        self._strategy_group = QButtonGroup(self)
        self._strategy_group.addButton(self._rb_argmax, 0)
        self._strategy_group.addButton(self._rb_priority, 1)

        if current_strategy == "layer_priority":
            self._rb_priority.setChecked(True)
        else:
            self._rb_argmax.setChecked(True)

        radio_row = QHBoxLayout()
        radio_row.setSpacing(20)
        radio_row.addWidget(self._rb_argmax)
        radio_row.addWidget(self._rb_priority)
        radio_row.addStretch()
        layout.addLayout(radio_row)

        sep1 = QFrame()
        sep1.setFrameShape(QFrame.Shape.HLine)
        sep1.setStyleSheet(f"color: {OUTLINE_VARIANT}; max-height: 1px;")
        layout.addWidget(sep1)

        # ---- Layer mapping + priorities ----------------------------------
        mapping_lbl = QLabel("ASSIGN MODEL LAYERS TO ANNOTATION LAYERS")
        mapping_lbl.setStyleSheet(
            f"color: {ON_SURFACE_VARIANT}; font-size: {FONT_SIZE_XS}px; "
            f"font-weight: 700; letter-spacing: 1.2px;"
        )
        layout.addWidget(mapping_lbl)

        # Column headers
        col_header = QHBoxLayout()
        col_header.setSpacing(10)
        hdr_model = QLabel("Model layer")
        hdr_model.setStyleSheet(
            f"color: {ON_SURFACE_VARIANT}; font-size: {FONT_SIZE_XS}px; font-weight: 600;"
        )
        hdr_model.setMinimumWidth(100)
        hdr_app = QLabel("Annotation layer")
        hdr_app.setStyleSheet(
            f"color: {ON_SURFACE_VARIANT}; font-size: {FONT_SIZE_XS}px; font-weight: 600;"
        )
        self._hdr_prio = QLabel("Priority")
        hdr_prio = self._hdr_prio
        hdr_prio.setFixedWidth(72)
        hdr_prio.setStyleSheet(
            f"color: {ON_SURFACE_VARIANT}; font-size: {FONT_SIZE_XS}px; font-weight: 600;"
        )
        col_header.addWidget(hdr_model)
        col_header.addWidget(QLabel(""), 0)  # arrow spacer
        col_header.addWidget(hdr_app, 1)
        col_header.addWidget(hdr_prio)
        layout.addLayout(col_header)

        rows_widget = QWidget()
        rows_layout = QVBoxLayout(rows_widget)
        rows_layout.setContentsMargins(0, 0, 0, 0)
        rows_layout.setSpacing(6)

        n_layers = len(plugin_layers)

        for idx, plugin_layer in enumerate(plugin_layers):
            row = QHBoxLayout()
            row.setSpacing(10)

            lbl = QLabel(plugin_layer.capitalize())
            lbl.setStyleSheet(
                f"color: {ON_SURFACE}; font-size: {FONT_SIZE_SM}px; "
                f"font-weight: 600; min-width: 100px;"
            )
            arrow = QLabel("→")
            arrow.setStyleSheet(
                f"color: {ON_SURFACE_VARIANT}; font-size: {FONT_SIZE_SM}px;"
            )

            # Annotation layer combo
            combo = QComboBox()
            combo.addItem("-- No assignment --", None)
            for app_layer in app_layers:
                combo.addItem(app_layer.capitalize(), app_layer)
            _fix_combo_scroll(combo)

            if plugin_layer in current_mapping:
                target = current_mapping[plugin_layer]
                restore_idx = next(
                    (i for i in range(combo.count()) if combo.itemData(i) == target),
                    0,
                )
                combo.setCurrentIndex(restore_idx)

            # Priority combo (1 = highest)
            prio_combo = QComboBox()
            for p in range(1, n_layers + 1):
                prio_combo.addItem(str(p), p)
            prio_combo.setFixedWidth(72)
            prio_combo.setToolTip(
                "Layer priority (1 = highest). Used with the \"Layer priority\" strategy."
            )
            saved_prio = current_priorities.get(plugin_layer, idx + 1)
            prio_idx = next(
                (i for i in range(prio_combo.count()) if prio_combo.itemData(i) == saved_prio),
                idx,
            )
            prio_combo.setCurrentIndex(prio_idx)
            _fix_combo_scroll(prio_combo)

            self._combos[plugin_layer] = combo
            self._prio_combos[plugin_layer] = prio_combo

            # Wire priority swap
            prio_combo.currentIndexChanged.connect(
                lambda _checked, layer=plugin_layer: self._on_priority_changed(layer)
            )

            row.addWidget(lbl)
            row.addWidget(arrow)
            row.addWidget(combo, 1)
            row.addWidget(prio_combo)
            rows_layout.addLayout(row)

        layout.addWidget(rows_widget)

        sep2 = QFrame()
        sep2.setFrameShape(QFrame.Shape.HLine)
        sep2.setStyleSheet(f"color: {OUTLINE_VARIANT}; max-height: 1px;")
        layout.addWidget(sep2)

        # ---- Buttons -----------------------------------------------------
        btn_row = QHBoxLayout()
        btn_row.setSpacing(8)
        btn_row.addStretch()

        cancel_btn = QPushButton("Cancel")
        cancel_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        cancel_btn.setMinimumHeight(32)
        cancel_btn.setMinimumWidth(90)
        cancel_btn.setStyleSheet(f"""
            QPushButton {{
                background: {SURFACE_CONTAINER_HIGHEST};
                color: {ON_SURFACE_VARIANT};
                border: none; border-radius: 6px;
                font-size: {FONT_SIZE_SM}px;
            }}
            QPushButton:hover {{ color: {ON_SURFACE}; }}
        """)
        cancel_btn.clicked.connect(self.reject)

        apply_btn = QPushButton("Apply")
        apply_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        apply_btn.setMinimumHeight(32)
        apply_btn.setMinimumWidth(90)
        apply_btn.setStyleSheet(f"""
            QPushButton {{
                background: {PRIMARY}; color: {ON_PRIMARY};
                border: none; border-radius: 6px;
                font-size: {FONT_SIZE_SM}px; font-weight: 700;
            }}
            QPushButton:hover {{ background: #b5fcff; }}
            QPushButton:pressed {{ background: #00e5ee; }}
        """)
        apply_btn.clicked.connect(self.accept)

        btn_row.addWidget(cancel_btn)
        btn_row.addWidget(apply_btn)
        layout.addLayout(btn_row)

        # Wire radio buttons to enable/disable priority combos
        self._rb_argmax.toggled.connect(self._on_strategy_changed)
        self._on_strategy_changed()  # set initial enabled state

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _on_strategy_changed(self) -> None:
        is_priority = self._rb_priority.isChecked()
        self._hdr_prio.setVisible(is_priority)
        for pc in self._prio_combos.values():
            pc.setVisible(is_priority)

    def _on_priority_changed(self, changed_layer: str) -> None:
        """Swap priority values so no two layers share the same priority."""
        if self._blocking:
            return
        new_val = self._prio_combos[changed_layer].currentData()
        # Find another layer that already holds this value
        for other_layer, pc in self._prio_combos.items():
            if other_layer == changed_layer:
                continue
            if pc.currentData() == new_val:
                # Find what the changed layer previously held and put it there
                # We don't store previous values, so use the first free value
                used = {c.currentData() for n, c in self._prio_combos.items() if n != other_layer}
                free_val = next(
                    v for v in range(1, len(self._plugin_layers) + 1) if v not in used
                )
                self._blocking = True
                other_idx = next(
                    i for i in range(pc.count()) if pc.itemData(i) == free_val
                )
                pc.setCurrentIndex(other_idx)
                self._blocking = False
                break

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def get_config(self) -> dict:
        """Return the full configuration as a plain dict.

        Keys: ``layer_mapping``, ``conflict_strategy``, ``layer_priorities``.
        """
        return {
            "layer_mapping": {
                layer: combo.currentData()
                for layer, combo in self._combos.items()
            },
            "conflict_strategy": (
                "layer_priority" if self._rb_priority.isChecked() else "argmax"
            ),
            "layer_priorities": {
                layer: pc.currentData()
                for layer, pc in self._prio_combos.items()
            },
        }


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
        self._cb_autolabel_configure: Optional[Callable[[str], None]] = None
        self._cb_open_project: Optional[Callable[[], None]] = None

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # Project header (sticky top, above scroll area)
        self._build_project_header()
        outer.addWidget(self._project_header_widget, 0)

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
        self._scroll_area = scroll

        # Sticky bottom area for autolabeling
        self._bottom_widget = QWidget()
        self._bottom_widget.setStyleSheet(f"background-color: {SURFACE_CONTAINER_HIGH};")
        self._bottom_layout = QVBoxLayout(self._bottom_widget)
        self._bottom_layout.setContentsMargins(14, 8, 14, 12)
        self._bottom_layout.setSpacing(8)
        outer.addWidget(self._bottom_widget, 0)

        self._build_layers_section(layer_configs)
        self._build_view_options()
        self._build_autolabel_section()
        self._layout.addStretch()

        # Nothing to interact with until an image is loaded
        self.set_image_loaded(False)

    # ------------------------------------------------------------------
    # Callback registration
    # ------------------------------------------------------------------

    def set_image_loaded(self, loaded: bool) -> None:
        """Enable or disable everything below the project header."""
        self._scroll_area.setEnabled(loaded)
        self._bottom_widget.setEnabled(loaded)

    def on_layer_selected(self, cb: Callable[[int], None]) -> None:
        self._cb_layer_selected = cb

    def on_open_project(self, cb: Callable[[], None]) -> None:
        self._cb_open_project = cb

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

    def on_autolabel_configure(self, cb: Callable[[str], None]) -> None:
        """Register *cb* to be called with the selected plugin_id when the
        configure button is clicked."""
        self._cb_autolabel_configure = cb

    def on_autolabel_plugin_changed(self, cb: Callable[[Optional[str]], None]) -> None:
        """Register *cb* to be called with the plugin_id whenever the combo changes."""
        self._cb_autolabel_plugin_changed = cb

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

    def set_project_name(self, name: str) -> None:
        """Update the displayed project name."""
        self._project_name_label.setText(name or "—")

    def refresh_autolabel_plugins(self, plugins, initial_plugin_id: Optional[str] = None) -> None:
        # Use the explicit initial_plugin_id if provided, else preserve the current selection.
        current_id = initial_plugin_id if initial_plugin_id is not None else self._q_autolabel_combo.currentData()
        self._q_autolabel_combo.blockSignals(True)
        self._q_autolabel_combo.clear()
        self._q_autolabel_combo.addItem("--Select model--", None)
        for plugin in plugins:
            self._q_autolabel_combo.addItem(plugin.display_name, plugin.id)
        # Try to restore previous selection
        restored = False
        if current_id is not None:
            for i in range(self._q_autolabel_combo.count()):
                if self._q_autolabel_combo.itemData(i) == current_id:
                    self._q_autolabel_combo.setCurrentIndex(i)
                    restored = True
                    break
        if not restored:
            self._q_autolabel_combo.setCurrentIndex(0)
        self._q_autolabel_combo.blockSignals(False)
        # Sync button states and fire the plugin-changed callback
        self._on_autolabel_combo_changed()

    def get_selected_plugin_id(self) -> Optional[str]:
        return self._q_autolabel_combo.currentData()

    def set_selected_plugin(self, plugin_id: str) -> None:
        """Select the combo item matching *plugin_id*, if present."""
        for i in range(self._q_autolabel_combo.count()):
            if self._q_autolabel_combo.itemData(i) == plugin_id:
                self._q_autolabel_combo.setCurrentIndex(i)
                return

    def update_mapping_indicator(self, has_mapping: bool) -> None:
        """Update the configure button appearance when a mapping is saved/cleared."""
        if has_mapping:
            self._q_autolabel_configure_btn.setToolTip("Mapping configurado — haz clic para editar")
            self._q_autolabel_configure_btn.setStyleSheet(
                f"QPushButton {{ background: rgba(161,250,255,0.18); border: none; "
                f"border-radius: 6px; color: {PRIMARY}; font-size: 14px; padding: 0; }}"
                f"QPushButton:hover {{ background: rgba(161,250,255,0.32); }}"
            )
        else:
            self._q_autolabel_configure_btn.setToolTip("Configurar correspondencia de capas")
            self._q_autolabel_configure_btn.setStyleSheet(
                f"QPushButton {{ background: transparent; border: none; "
                f"border-radius: 6px; color: {ON_SURFACE_VARIANT}; font-size: 14px; padding: 0; }}"
                f"QPushButton:hover {{ color: {ON_SURFACE}; background: rgba(255,255,255,0.07); }}"
            )

    # ------------------------------------------------------------------
    # Widget construction
    # ------------------------------------------------------------------

    def _build_project_header(self) -> None:
        """Build the sticky project section header at the top."""
        self._project_header_widget = QWidget()
        self._project_header_widget.setStyleSheet(
            f"background-color: {SURFACE_CONTAINER_HIGH};"
        )
        outer = QVBoxLayout(self._project_header_widget)
        outer.setContentsMargins(14, 8, 8, 8)
        outer.setSpacing(4)

        # Top row: "Project" section label + open-project button
        top_row = QWidget()
        top_row.setStyleSheet("background: transparent; border: none;")
        top_layout = QHBoxLayout(top_row)
        top_layout.setContentsMargins(0, 0, 0, 0)
        top_layout.setSpacing(6)

        section_lbl = QLabel("Project")
        section_lbl.setStyleSheet(
            f"color: {PRIMARY}; font-size: {FONT_SIZE_SM}px; "
            f"font-weight: 700; letter-spacing: 1px;"
        )
        top_layout.addWidget(section_lbl)
        top_layout.addStretch()

        open_btn = QPushButton("📂")
        open_btn.setFixedSize(28, 28)
        open_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        open_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        open_btn.setToolTip("Open another project folder")
        open_btn.setStyleSheet(
            f"QPushButton {{ background: transparent; border: none; "
            f"border-radius: 6px; font-size: 14px; padding: 0; }}"
            f"QPushButton:hover {{ background: rgba(255,255,255,0.07); }}"
        )
        open_btn.clicked.connect(self._on_open_project_clicked)
        top_layout.addWidget(open_btn)
        outer.addWidget(top_row)

        # Project name label (read-only — always the folder name)
        self._project_name_label = QLabel("—")
        self._project_name_label.setStyleSheet(
            f"color: {ON_SURFACE}; font-size: {FONT_SIZE_SM}px; "
            f"border: none; padding-bottom: 2px;"
        )
        outer.addWidget(self._project_name_label)

    def _on_open_project_clicked(self) -> None:
        if self._cb_open_project:
            self._cb_open_project()

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
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet(f"color: {OUTLINE_VARIANT}; max-height: 1px; margin-top: 4px;")
        self._layout.addWidget(sep)

        workspace_lbl = QLabel("Workspace")
        workspace_lbl.setStyleSheet(
            f"color: {PRIMARY}; font-size: {FONT_SIZE_SM}px; "
            f"border-top: 1px solid {OUTLINE_VARIANT}; padding-top: 12px; "
            f"font-weight: 700; letter-spacing: 1px;"
        )
        self._layout.addWidget(workspace_lbl)

        lbl = QLabel("VIEW OPTIONS")
        lbl.setStyleSheet(
            f"color: {ON_SURFACE_VARIANT}; font-size: {FONT_SIZE_XS}px; "
            f"font-weight: 700; letter-spacing: 1.5px; margin-top: 4px;"
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

        # Model selector row: combo + configure button
        combo_row = QHBoxLayout()
        combo_row.setSpacing(6)

        self._q_autolabel_combo = QComboBox()
        self._q_autolabel_combo.addItem("--Select model--", None)
        self._q_autolabel_combo.currentIndexChanged.connect(self._on_autolabel_combo_changed)
        combo_row.addWidget(self._q_autolabel_combo, 1)

        self._q_autolabel_configure_btn = QPushButton("⚙")
        self._q_autolabel_configure_btn.setFixedSize(28, 28)
        self._q_autolabel_configure_btn.setEnabled(False)
        self._q_autolabel_configure_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._q_autolabel_configure_btn.setToolTip("Configurar correspondencia de capas")
        self._q_autolabel_configure_btn.setStyleSheet(
            f"QPushButton {{ background: transparent; border: none; "
            f"border-radius: 6px; color: {ON_SURFACE_VARIANT}; font-size: 14px; padding: 0; }}"
            f"QPushButton:hover {{ color: {ON_SURFACE}; background: rgba(255,255,255,0.07); }}"
            f"QPushButton:disabled {{ color: {OUTLINE_VARIANT}; }}"
        )
        self._q_autolabel_configure_btn.clicked.connect(self._on_configure_clicked)
        combo_row.addWidget(self._q_autolabel_configure_btn)

        card_layout.addLayout(combo_row)

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
        has_plugin = plugin_id is not None
        self._q_autolabel_run_button.setEnabled(has_plugin)
        self._q_autolabel_configure_btn.setEnabled(has_plugin)
        if self._cb_autolabel_plugin_changed:
            self._cb_autolabel_plugin_changed(plugin_id)

    def _on_configure_clicked(self) -> None:
        plugin_id = self._q_autolabel_combo.currentData()
        if plugin_id and self._cb_autolabel_configure:
            self._cb_autolabel_configure(plugin_id)

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
