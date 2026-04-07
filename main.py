"""Entry point for the Pixel Annotation Tool.

Architecture:
  domain/           — ImageDocument, LayerConfig (pure domain, no I/O)
  infrastructure/   — ImageRepository, ImageMetadata, TimeTracker, WebService
  application/      — AppState, AnnotatorController, AutolabelService, tools
  viewer/           — IImageAnnotationViewer contract + QtImageAnnotationViewer
  presentation/     — ToolbarPanel, MainWindow
"""
import sys
import logging

from PyQt6.QtWidgets import QApplication

logging.basicConfig(level=logging.INFO)


def _ensure_layers_file() -> None:
    """Create a default layers.txt if the file is missing or empty."""
    import os
    from domain.layer_config import _DEFAULT_LAYERS, write_layers_file
    path = "layers.txt"
    if not os.path.exists(path) or os.stat(path).st_size == 0:
        write_layers_file(list(_DEFAULT_LAYERS))


if __name__ == "__main__":
    _ensure_layers_file()
    from presentation.main_window import MainWindow
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
