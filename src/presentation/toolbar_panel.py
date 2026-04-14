"""Left toolbar panel — precision annotation tools.

Redesigned to match the design system from design/DESIGN.md.
Tool groups are presented in cards with sliders and controls.
Includes: Selector, Pen, Fill, Erase tools + Gallery button.
"""
from __future__ import annotations

from typing import Callable, Optional

from PyQt6.QtCore import Qt, QSize
from PyQt6.QtGui import QKeySequence, QShortcut
from PyQt6.QtWidgets import (
    QCheckBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from domain.layer_config import LayerConfig
from application.app_state import ToolbarState
from presentation.style import (
    PRIMARY,
    ON_SURFACE,
    ON_SURFACE_VARIANT,
    SURFACE_BRIGHT,
    SURFACE_CONTAINER_HIGH,
    SURFACE_CONTAINER_HIGHEST,
    SURFACE_CONTAINER_LOWEST,
    OUTLINE_VARIANT,
    FONT_SIZE_SM,
    FONT_SIZE_XS,
    SIDEBAR_WIDTH,
)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _section_header(text: str) -> QLabel:
    lbl = QLabel(text)
    lbl.setStyleSheet(
        f"color: {PRIMARY}; font-size: {FONT_SIZE_SM}px; "
        f"font-weight: 700; letter-spacing: 1px;"
    )
    return lbl


class _ToolCard(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet(f"""
            _ToolCard {{
                background-color: {SURFACE_CONTAINER_HIGHEST};
                border-radius: 12px;
                border: none;
            }}
        """)
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(10, 10, 10, 10)
        self._layout.setSpacing(8)

    @property
    def card_layout(self) -> QVBoxLayout:
        return self._layout


class _ToolButton(QPushButton):
    def __init__(self, label: str, shortcut_key: str = "",
                 parent=None):
        super().__init__(parent)
        self.setCheckable(True)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        label_text = f"{label}  ({shortcut_key.upper()})" if shortcut_key else label
        self.setText(f"  {label_text}")
        self.setMinimumHeight(36)
        self._update_style(False)

    def _update_style(self, active: bool):
        if active:
            self.setStyleSheet(f"""
                QPushButton {{
                    background-color: {SURFACE_BRIGHT};
                    color: {PRIMARY};
                    border: none; border-radius: 6px;
                    text-align: left; padding: 6px 10px;
                    font-size: {FONT_SIZE_SM}px; font-weight: 700;
                    letter-spacing: 0.5px;
                }}
            """)
        else:
            self.setStyleSheet(f"""
                QPushButton {{
                    background: transparent;
                    color: {ON_SURFACE_VARIANT};
                    border: none; border-radius: 6px;
                    text-align: left; padding: 6px 10px;
                    font-size: {FONT_SIZE_SM}px; font-weight: 600;
                    letter-spacing: 0.5px;
                }}
                QPushButton:hover {{
                    background: {SURFACE_CONTAINER_HIGHEST};
                    color: {ON_SURFACE};
                }}
            """)

    def set_active(self, active: bool):
        self._update_style(active)


# ------------------------------------------------------------------
# ToolbarPanel
# ------------------------------------------------------------------

class ToolbarPanel(QWidget):
    """Left-side annotation toolbar.

    Contains tool groups (Selector, Pen, Fill, Erase), Gallery button,
    and Web Service mode toggle.  All callbacks are wired by the caller.
    """

    def __init__(
        self,
        layer_configs: list[LayerConfig],
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._layer_configs = layer_configs
        self.setFixedWidth(SIDEBAR_WIDTH)
        self.setStyleSheet(f"background-color: {SURFACE_CONTAINER_HIGH};")

        self._cb_tool_selected: Optional[Callable[[str], None]] = None

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

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

        self._bottom_widget = QWidget()
        self._bottom_widget.setStyleSheet(f"background-color: {SURFACE_CONTAINER_HIGH};")
        self._bottom_layout = QVBoxLayout(self._bottom_widget)
        self._bottom_layout.setContentsMargins(14, 8, 14, 12)
        self._bottom_layout.setSpacing(8)
        outer.addWidget(self._bottom_widget, 0)

        self._build_header()
        self._build_selector_group()
        self._build_pen_group()
        self._build_fill_group()
        self._build_erase_group()
        self._build_erase_all_button()
        self._build_gallery_button()
        self._build_web_service_section()
        self._layout.addStretch()
        self._build_shortcuts()

        # Select the default tool visually
        self.set_active_tool("selector")

        # Nothing to annotate until an image is loaded
        self.set_image_loaded(False)

    # ------------------------------------------------------------------
    # Callback registration
    # ------------------------------------------------------------------

    def set_image_loaded(self, loaded: bool) -> None:
        """Enable or disable all tool controls (no effect on the panel frame itself)."""
        self._scroll_area.setEnabled(loaded)
        self._bottom_widget.setEnabled(loaded)

    def on_tool_selected(self, cb: Callable[[str], None]) -> None:
        self._cb_tool_selected = cb
    def on_pen_size_changed(self, cb: Callable[[int], None]) -> None:
        self._q_pen_spin.valueChanged.connect(cb)
        self._q_pen_slider.valueChanged.connect(
            lambda v: (self._q_pen_spin.blockSignals(True),
                       self._q_pen_spin.setValue(v),
                       self._q_pen_spin.blockSignals(False),
                       cb(v))
        )

    def on_eraser_size_changed(self, cb: Callable[[int], None]) -> None:
        self._q_eraser_spin.valueChanged.connect(cb)
        self._q_eraser_slider.valueChanged.connect(
            lambda v: (self._q_eraser_spin.blockSignals(True),
                       self._q_eraser_spin.setValue(v),
                       self._q_eraser_spin.blockSignals(False),
                       cb(v))
        )

    def on_threshold_changed(self, cb: Callable[[int], None]) -> None:
        self._q_threshold_slider.valueChanged.connect(cb)

    def on_auto_smooth_changed(self, cb: Callable[[bool], None]) -> None:
        self._q_autosmooth.stateChanged.connect(
            lambda: cb(self._q_autosmooth.isChecked())
        )

    def on_fill_all_changed(self, cb: Callable[[bool], None]) -> None:
        self._q_fill_all.stateChanged.connect(
            lambda: cb(self._q_fill_all.isChecked())
        )

    def on_erase_all_clicked(self, cb: Callable) -> None:
        self._q_erase_all_button.clicked.connect(cb)

    def on_gallery_clicked(self, cb: Callable) -> None:
        self._q_gallery_button.clicked.connect(cb)

    def on_web_service_mode_changed(self, cb: Callable[[bool], None]) -> None:
        self._q_web_service_mode.stateChanged.connect(
            lambda: cb(self._q_web_service_mode.isChecked())
        )

    def on_submit_annotations(self, cb: Callable) -> None:
        self._q_submit_button.clicked.connect(cb)

    def on_cancel_annotations(self, cb: Callable) -> None:
        self._q_cancel_button.clicked.connect(cb)

    # ------------------------------------------------------------------
    # State update methods
    # ------------------------------------------------------------------

    def set_active_tool(self, tool: str) -> None:
        mapping = {
            "selector": self._q_selector_btn,
            "pen":      self._q_pen_btn,
            "fill":     self._q_fill_btn,
            "erase":    self._q_erase_btn,
        }
        for t, btn in mapping.items():
            btn.setChecked(t == tool)
            btn.set_active(t == tool)
        self._update_tool_widget_states(tool)

    def set_pen_size(self, size: int) -> None:
        for w in (self._q_pen_spin, self._q_pen_slider):
            w.blockSignals(True)
            w.setValue(size)
            w.blockSignals(False)

    def set_eraser_size(self, size: int) -> None:
        for w in (self._q_eraser_spin, self._q_eraser_slider):
            w.blockSignals(True)
            w.setValue(size)
            w.blockSignals(False)

    def set_threshold(self, value: int) -> None:
        self._q_threshold_slider.blockSignals(True)
        self._q_threshold_slider.setValue(value)
        self._q_threshold_slider.blockSignals(False)
        self._q_threshold_value_label.setText(f"{int(value / 128 * 100)}%")

    def set_web_service_ui(self, mode_active: bool, progress_complete: bool = False) -> None:
        self._q_submit_button.setVisible(mode_active)
        self._q_cancel_button.setVisible(mode_active)
        self._q_submit_button.setEnabled(mode_active and progress_complete)

    def sync(self, state: ToolbarState) -> None:
        self.set_active_tool(state.active_tool)
        self.set_pen_size(state.pen_size)
        self.set_eraser_size(state.eraser_size)
        self.set_threshold(state.selector_threshold)

        for widget, value in [
            (self._q_autosmooth,  state.selector_auto_smooth),
            (self._q_fill_all,    state.fill_all),
        ]:
            widget.blockSignals(True)
            widget.setChecked(value)
            widget.blockSignals(False)

    # ------------------------------------------------------------------
    # Widget construction
    # ------------------------------------------------------------------

    def _build_header(self) -> None:
        self._layout.addWidget(_section_header("Annotation Tools"))

    def _build_selector_group(self) -> None:
        card = _ToolCard()
        self._q_selector_btn = _ToolButton("SELECTOR", "s")
        self._q_selector_btn.clicked.connect(lambda: self._fire_tool("selector"))
        card.card_layout.addWidget(self._q_selector_btn)

        row = QHBoxLayout()
        row.setContentsMargins(4, 0, 4, 0)
        lbl = QLabel("Threshold")
        lbl.setStyleSheet(f"color: {ON_SURFACE_VARIANT}; font-size: {FONT_SIZE_XS}px; font-weight: 500;")
        self._q_threshold_value_label = QLabel("25%")
        self._q_threshold_value_label.setStyleSheet(f"color: {ON_SURFACE_VARIANT}; font-size: {FONT_SIZE_XS}px;")
        row.addWidget(lbl)
        row.addStretch()
        row.addWidget(self._q_threshold_value_label)
        card.card_layout.addLayout(row)

        self._q_threshold_slider = QSlider(Qt.Orientation.Horizontal)
        self._q_threshold_slider.setMinimum(1)
        self._q_threshold_slider.setMaximum(128)
        self._q_threshold_slider.setValue(32)
        self._q_threshold_slider.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        card.card_layout.addWidget(self._q_threshold_slider)

        self._q_autosmooth = QCheckBox("Auto-smooth")
        self._q_autosmooth.setChecked(True)
        self._q_autosmooth.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        card.card_layout.addWidget(self._q_autosmooth)

        self._layout.addWidget(card)

    def _build_pen_group(self) -> None:
        card = _ToolCard()
        self._q_pen_btn = _ToolButton("PEN", "p")
        self._q_pen_btn.clicked.connect(lambda: self._fire_tool("pen"))
        card.card_layout.addWidget(self._q_pen_btn)

        row = QHBoxLayout()
        row.setContentsMargins(4, 0, 4, 0)
        lbl = QLabel("Pen Size")
        lbl.setStyleSheet(f"color: {ON_SURFACE_VARIANT}; font-size: {FONT_SIZE_XS}px; font-weight: 500;")
        self._q_pen_spin = QSpinBox()
        self._q_pen_spin.setMinimum(1)
        self._q_pen_spin.setMaximum(50)
        self._q_pen_spin.setValue(5)
        self._q_pen_spin.setFixedWidth(48)
        self._q_pen_spin.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        row.addWidget(lbl)
        row.addStretch()
        row.addWidget(self._q_pen_spin)
        card.card_layout.addLayout(row)

        self._q_pen_slider = QSlider(Qt.Orientation.Horizontal)
        self._q_pen_slider.setMinimum(1)
        self._q_pen_slider.setMaximum(50)
        self._q_pen_slider.setValue(1)
        self._q_pen_slider.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        card.card_layout.addWidget(self._q_pen_slider)

        self._layout.addWidget(card)

    def _build_fill_group(self) -> None:
        card = _ToolCard()
        self._q_fill_btn = _ToolButton("FILL", "f")
        self._q_fill_btn.clicked.connect(lambda: self._fire_tool("fill"))
        card.card_layout.addWidget(self._q_fill_btn)

        self._q_fill_all = QCheckBox("Fill non-contiguous regions")
        self._q_fill_all.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        card.card_layout.addWidget(self._q_fill_all)

        self._layout.addWidget(card)

    def _build_erase_group(self) -> None:
        card = _ToolCard()
        self._q_erase_btn = _ToolButton("ERASE", "e")
        self._q_erase_btn.clicked.connect(lambda: self._fire_tool("erase"))
        card.card_layout.addWidget(self._q_erase_btn)

        row = QHBoxLayout()
        row.setContentsMargins(4, 0, 4, 0)
        lbl = QLabel("Eraser Size")
        lbl.setStyleSheet(f"color: {ON_SURFACE_VARIANT}; font-size: {FONT_SIZE_XS}px; font-weight: 500;")
        self._q_eraser_spin = QSpinBox()
        self._q_eraser_spin.setMinimum(1)
        self._q_eraser_spin.setMaximum(50)
        self._q_eraser_spin.setValue(5)
        self._q_eraser_spin.setFixedWidth(48)
        self._q_eraser_spin.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        row.addWidget(lbl)
        row.addStretch()
        row.addWidget(self._q_eraser_spin)
        card.card_layout.addLayout(row)

        self._q_eraser_slider = QSlider(Qt.Orientation.Horizontal)
        self._q_eraser_slider.setMinimum(1)
        self._q_eraser_slider.setMaximum(50)
        self._q_eraser_slider.setValue(5)
        self._q_eraser_slider.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        card.card_layout.addWidget(self._q_eraser_slider)

        self._layout.addWidget(card)

    def _build_erase_all_button(self) -> None:
        self._q_erase_all_button = QPushButton("  \U0001F5D1  Erase All")
        self._q_erase_all_button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._q_erase_all_button.setMinimumHeight(38)
        self._q_erase_all_button.setStyleSheet(f"""
            QPushButton {{
                background: transparent;
                color: #CF6679;
                border: 1px solid rgba(207, 102, 121, 0.35);
                border-radius: 10px;
                text-align: left; padding: 6px 12px;
                font-size: {FONT_SIZE_SM}px; font-weight: 600;
                letter-spacing: 0.5px;
            }}
            QPushButton:hover {{
                background: rgba(207, 102, 121, 0.12);
                color: #F28B9A;
                border-color: rgba(207, 102, 121, 0.6);
            }}
            QPushButton:pressed {{
                background: rgba(207, 102, 121, 0.22);
            }}
        """)
        self._layout.addWidget(self._q_erase_all_button)

    def _build_gallery_button(self) -> None:
        self._q_gallery_button = QPushButton("  \U0001F5BC  Gallery")
        self._q_gallery_button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._q_gallery_button.setMinimumHeight(42)
        self._q_gallery_button.setStyleSheet(f"""
            QPushButton {{
                background: transparent;
                color: {ON_SURFACE_VARIANT};
                border: 1px solid rgba(72, 72, 72, 0.1);
                border-radius: 12px;
                text-align: left; padding: 8px 12px;
                font-size: {FONT_SIZE_SM}px; font-weight: 600;
                letter-spacing: 0.5px;
            }}
            QPushButton:hover {{
                background: rgba(161, 250, 255, 0.1);
                color: {PRIMARY};
            }}
        """)
        self._bottom_layout.insertWidget(0, self._q_gallery_button)

    def _build_web_service_section(self) -> None:
        card = QFrame()
        card.setStyleSheet("""
            QFrame {
                background-color: rgba(37, 38, 38, 0.3);
                border-radius: 12px;
                border: 1px solid rgba(72, 72, 72, 0.1);
            }
        """)
        card_layout = QHBoxLayout(card)
        card_layout.setContentsMargins(10, 10, 10, 10)

        icon_row = QHBoxLayout()
        icon_lbl = QLabel("\u2601")
        icon_lbl.setStyleSheet(f"font-size: 16px; color: {ON_SURFACE_VARIANT};")
        ws_lbl = QLabel("WEB SERVICE")
        ws_lbl.setStyleSheet(
            f"color: {ON_SURFACE_VARIANT}; font-size: {FONT_SIZE_XS}px; "
            f"font-weight: 600; letter-spacing: 1px;"
        )
        left = QVBoxLayout()
        icon_row.addWidget(icon_lbl)
        icon_row.addWidget(ws_lbl)
        icon_row.addStretch()
        left.addLayout(icon_row)
        card_layout.addLayout(left, 1)

        self._q_web_service_mode = QCheckBox()
        self._q_web_service_mode.setChecked(False)
        self._q_web_service_mode.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        card_layout.addWidget(self._q_web_service_mode)

        self._bottom_layout.addWidget(card)

        self._q_submit_button = QPushButton("Submit Annotations")
        self._q_submit_button.setVisible(False)
        self._q_submit_button.setEnabled(False)
        self._q_submit_button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._q_submit_button.setStyleSheet(f"""
            QPushButton {{
                background: {PRIMARY};
                color: #006165; font-weight: 700;
                border-radius: 8px; padding: 8px;
                font-size: {FONT_SIZE_SM}px; letter-spacing: 1px;
            }}
            QPushButton:disabled {{ background: {SURFACE_CONTAINER_HIGHEST}; color: {OUTLINE_VARIANT}; }}
        """)
        self._bottom_layout.addWidget(self._q_submit_button)

        self._q_cancel_button = QPushButton("Cancel")
        self._q_cancel_button.setVisible(False)
        self._q_cancel_button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._bottom_layout.addWidget(self._q_cancel_button)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _build_shortcuts(self) -> None:
        """Register keyboard shortcuts via QShortcut so they work regardless
        of which widget holds focus.

        * S / P / F / W → ApplicationShortcut: fire from anywhere.
        * E (erase)      → WidgetWithChildrenShortcut: fires only when the
          toolbar (or one of its children) has focus.  When the GL canvas has
          focus the controller handles 'E' context-sensitively (expand-
          selection while drawing, otherwise switch to erase).
        """
        for key, tool in [("s", "selector"), ("p", "pen"), ("f", "fill")]:
            sc = QShortcut(QKeySequence(key), self)
            sc.setContext(Qt.ShortcutContext.ApplicationShortcut)
            sc.activated.connect(lambda t=tool: self._fire_tool(t))

        # 'e' must not be ApplicationShortcut: that would swallow the key event
        # before the viewer can use it for selection expansion.
        sc_e = QShortcut(QKeySequence("e"), self)
        sc_e.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
        sc_e.activated.connect(lambda: self._fire_tool("erase"))

    def _fire_tool(self, tool: str) -> None:
        mapping = {
            "selector": self._q_selector_btn,
            "pen":      self._q_pen_btn,
            "fill":     self._q_fill_btn,
            "erase":    self._q_erase_btn,
        }
        for t, btn in mapping.items():
            btn.set_active(t == tool)
            btn.setChecked(t == tool)
        self._update_tool_widget_states(tool)
        if self._cb_tool_selected:
            self._cb_tool_selected(tool)

    def _update_tool_widget_states(self, tool: str) -> None:
        pen_active      = tool == "pen"
        selector_active = tool == "selector"
        erase_active    = tool == "erase"

        self._q_pen_spin.setEnabled(pen_active)
        self._q_pen_slider.setEnabled(pen_active)
        self._q_threshold_slider.setEnabled(selector_active)
        self._q_autosmooth.setEnabled(selector_active)
        self._q_eraser_spin.setEnabled(erase_active)
        self._q_eraser_slider.setEnabled(erase_active)
