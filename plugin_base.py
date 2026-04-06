"""Base class for autolabeling plugins.

Each plugin must subclass AutolabelPlugin and implement all abstract
properties and methods.  The plugin's ``run`` method receives the raw
BGR image (as loaded by OpenCV) and must return a 2-D NumPy label map
whose pixel values are indices into the plugin's ``supported_layers``
list.
"""

from abc import ABC, abstractmethod
import numpy as np


class AutolabelPlugin(ABC):
    """Abstract base class that every autolabeling plugin must implement."""

    @property
    @abstractmethod
    def id(self) -> str:
        """Unique plugin identifier (matches the subdirectory name)."""
        ...

    @property
    @abstractmethod
    def display_name(self) -> str:
        """Human-readable name shown in the UI dropdown."""
        ...

    @property
    @abstractmethod
    def supported_layers(self) -> list:
        """Ordered list of layer names this plugin can produce.

        The index of each name in this list corresponds to the integer
        value used in the label map returned by ``run()``.
        """
        ...

    @abstractmethod
    def run(self, image: np.ndarray) -> np.ndarray:
        """Run the plugin on the given image.

        Parameters
        ----------
        image : np.ndarray
            Input image in BGR format (OpenCV convention), shape ``(H, W, C)``.

        Returns
        -------
        np.ndarray
            Label map of shape ``(H, W)`` with dtype compatible with
            integer comparison.  Each pixel value must be a valid index
            into ``supported_layers``.
        """
        ...
