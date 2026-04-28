"""Centralised keyboard shortcut handler.

Installed as a QApplication event filter so that shortcuts respond regardless
of which widget currently has focus, while explicitly NOT firing when the user
is typing in a text input widget.

Layer shortcuts:
- V          → toggle all layers visibility.
- L          → toggle all layers lock.
- Ctrl + 1-9 → toggle visibility of that specific layer.
- Alt  + 1-9 → toggle lock of that specific layer.
"""
from __future__ import annotations

from typing import Optional, TYPE_CHECKING

from PyQt6.QtCore import QEvent, QObject, Qt
from PyQt6.QtWidgets import QAbstractSpinBox, QApplication, QComboBox, QLineEdit

if TYPE_CHECKING:
    from application.annotator_controller import AnnotatorController

_TEXT_INPUT_TYPES = (QLineEdit, QAbstractSpinBox, QComboBox)

# Keys mapped to symbolic names; must stay consistent with qt_viewer._key_name.
_KEY_MAP = {
    Qt.Key.Key_P:      "P",
    Qt.Key.Key_S:      "S",
    Qt.Key.Key_F:      "F",
    Qt.Key.Key_Z:      "Z",
    Qt.Key.Key_Y:      "Y",
    Qt.Key.Key_E:      "E",
    Qt.Key.Key_R:      "R",
    Qt.Key.Key_I:      "I",
    Qt.Key.Key_M:      "M",
    Qt.Key.Key_G:      "G",
    Qt.Key.Key_V:      "V",
    Qt.Key.Key_L:      "L",
    Qt.Key.Key_Plus:   "Plus",
    Qt.Key.Key_Equal:  "Plus",   # unshifted + on some keyboards
    Qt.Key.Key_Minus:  "Minus",
    Qt.Key.Key_Space:  "Space",
    Qt.Key.Key_Escape: "Escape",
    Qt.Key.Key_Return: "Return",
    Qt.Key.Key_Enter:  "Return",
    Qt.Key.Key_F1:     "F1",
    **{getattr(Qt.Key, f"Key_{i}"): str(i) for i in range(1, 10)},
}

_DIGITS = frozenset("123456789")


class ShortcutManager(QObject):
    """Application-level event filter that routes key events to the controller.

    Usage::

        mgr = ShortcutManager(main_window)
        QApplication.instance().installEventFilter(mgr)
        # Later, when a controller becomes available:
        mgr.set_controller(controller)
    """

    def __init__(self, main_window, parent=None) -> None:
        super().__init__(parent)
        self._window = main_window
        self._controller: Optional["AnnotatorController"] = None
        # Digit keys consumed on press (Ctrl/Alt+digit) so their release
        # is also consumed, preventing an unintended layer-selection.
        self._consumed_digits: set[str] = set()

    def set_controller(self, controller: Optional["AnnotatorController"]) -> None:
        self._controller = controller

    # ------------------------------------------------------------------
    # Event filter
    # ------------------------------------------------------------------

    def eventFilter(self, obj, event) -> bool:  # noqa: N802
        t = event.type()
        if t not in (QEvent.Type.KeyPress, QEvent.Type.KeyRelease):
            return False

        if self._controller is None:
            return False

        focused = QApplication.focusWidget()
        if isinstance(focused, _TEXT_INPUT_TYPES):
            return False

        key_name = _KEY_MAP.get(event.key(), "")
        if not key_name:
            return False

        mods: set[str] = set()
        m = event.modifiers()
        if m & Qt.KeyboardModifier.ControlModifier:
            mods.add("ctrl")
        if m & Qt.KeyboardModifier.ShiftModifier:
            mods.add("shift")
        if m & Qt.KeyboardModifier.AltModifier:
            mods.add("alt")
        mods_fs = frozenset(mods)

        # Keys always handled by ShortcutManager regardless of which widget
        # has focus (the viewer included).
        is_manager_key = (
            key_name in ("V", "L")
            or (key_name in _DIGITS and ("ctrl" in mods_fs or "alt" in mods_fs))
            or (t == QEvent.Type.KeyRelease and key_name in self._consumed_digits)
        )

        # When the viewer has focus it handles most keys via its own
        # keyPressEvent → avoid double-handling for non-manager keys.
        if hasattr(self._window, "_viewer") and focused is self._window._viewer:
            if not is_manager_key:
                return False

        if t == QEvent.Type.KeyPress:
            return self._handle_press(key_name, mods_fs)
        else:
            return self._handle_release(key_name, mods_fs)

    # ------------------------------------------------------------------
    # Key press / release dispatch
    # ------------------------------------------------------------------

    def _handle_press(self, key_name: str, mods: frozenset) -> bool:
        if key_name == "V":
            if self._controller:
                self._controller.toggle_all_visibility()
            return True

        if key_name == "L":
            if self._controller:
                self._controller.toggle_all_lock()
            return True

        if key_name in _DIGITS and "ctrl" in mods:
            self._consumed_digits.add(key_name)
            if self._controller:
                self._controller.toggle_layer_visibility_by_index(int(key_name) - 1)
            return True

        if key_name in _DIGITS and "alt" in mods:
            self._consumed_digits.add(key_name)
            if self._controller:
                self._controller.toggle_layer_lock(int(key_name) - 1)
            return True

        self._controller.handle_key_press(key_name, mods)
        return False

    def _handle_release(self, key_name: str, mods: frozenset) -> bool:
        if key_name in ("V", "L"):
            return True  # no action on release

        if key_name in self._consumed_digits:
            self._consumed_digits.discard(key_name)
            return True

        self._controller.handle_key_release(key_name, mods)
        return False
