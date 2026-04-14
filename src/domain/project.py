"""Project model — represents an opened project folder.

A project is a folder that the user has opened.  It stores its own
configuration file (``.pixellabeler/project.toml``) with all the
application settings relevant to that project.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ProjectConfig:
    """Serialisable project-level configuration.

    Stores all application state worth persisting between sessions for a
    given project folder.
    """
    # Display name — always derived from the folder name, not user-editable.
    name: str = ""

    # Viewer / tool settings
    viewer_backend: str = "gl"
    pen_size: int = 5
    eraser_size: int = 5
    selector_threshold: int = 32
    selector_auto_smooth: bool = True
    fill_all: bool = False

    # View settings
    show_image: bool = True
    show_other_layers: bool = True
    show_missing_pixels: bool = False
    show_grid: bool = True

    # Session
    active_layer: int = 0
    locked_layers: list[int] = field(default_factory=list)
    hidden_layers: list[int] = field(default_factory=list)
    last_image: Optional[str] = None

    # Last selected AI model plugin id
    selected_plugin_id: Optional[str] = None

    # Per-plugin configurations: {plugin_id -> {key: value}}
    plugin_configs: dict[str, dict] = field(default_factory=dict)
