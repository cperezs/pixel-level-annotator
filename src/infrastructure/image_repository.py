"""File-system repository for images and annotation masks.

All path-building and format details are encapsulated here so that
the domain layer (``ImageDocument``) remains free of I/O concerns.
"""
from __future__ import annotations

import os
import logging

import cv2
import numpy as np

from domain.image_document import ImageDocument

logger = logging.getLogger(__name__)

_IMAGES_DIR = "images"
_ANNOTATIONS_DIR = "annotations"


class ImageRepository:
    """Loads and saves :class:`~domain.image_document.ImageDocument` objects.

    The repository owns all knowledge of:
    - which directories images and annotations live in,
    - how annotation layer files are named (``<stem>_<i>.png``),
    - how to convert between NumPy arrays and image files.
    """

    def __init__(
        self,
        images_dir: str = _IMAGES_DIR,
        annotations_dir: str = _ANNOTATIONS_DIR,
    ) -> None:
        self._images_dir = images_dir
        self._annotations_dir = annotations_dir

    # ------------------------------------------------------------------
    # Listing
    # ------------------------------------------------------------------

    def list_images(self) -> list[str]:
        """Return sorted base-filenames of every image in *images_dir*."""
        exts = {".jpg", ".jpeg", ".png", ".gif"}
        if not os.path.isdir(self._images_dir):
            return []
        names = [
            f for f in os.listdir(self._images_dir)
            if os.path.splitext(f.lower())[1] in exts
        ]
        return sorted(names)

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load(self, filename: str, nlayers: int) -> ImageDocument:
        """Load *filename* and its annotation masks from disk.

        Parameters
        ----------
        filename:
            Base filename (e.g. ``"img01.png"``), relative to *images_dir*.
        nlayers:
            Expected number of annotation layers.

        Returns
        -------
        ImageDocument with annotations loaded from disk, or initialised
        to zero-filled masks when no annotation files are found.
        """
        image_path = os.path.join(self._images_dir, filename)
        bgr = cv2.imread(image_path)
        if bgr is None:
            raise FileNotFoundError(f"Cannot load image: {image_path}")

        h, w = bgr.shape[:2]
        annotations = self._load_annotations(filename, nlayers, h, w)
        doc = ImageDocument(bgr, annotations, image_path)

        if np.all(annotations == 0):
            self.save_annotations(doc, filename)

        logger.info("Loaded image: %s", image_path)
        return doc

    # ------------------------------------------------------------------
    # Saving
    # ------------------------------------------------------------------

    def save_annotations(self, document: ImageDocument, filename: str) -> None:
        """Persist all annotation layers of *document* to disk."""
        os.makedirs(self._annotations_dir, exist_ok=True)
        stem = os.path.splitext(os.path.basename(filename))[0]
        for i, mask in enumerate(document.annotations):
            path = os.path.join(self._annotations_dir, f"{stem}_{i}.png")
            cv2.imwrite(path, mask)
        logger.debug("Saved annotations for: %s", filename)

    # ------------------------------------------------------------------
    # Path helpers
    # ------------------------------------------------------------------

    def metadata_path(self, filename: str) -> str:
        """Return the ``<stem>.metadata`` path for *filename*."""
        stem = os.path.splitext(os.path.basename(filename))[0]
        return os.path.join(self._annotations_dir, f"{stem}.metadata")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_annotations(
        self, filename: str, nlayers: int, h: int, w: int
    ) -> np.ndarray:
        stem = os.path.splitext(os.path.basename(filename))[0]
        layers: list[np.ndarray] = []
        for i in range(nlayers):
            path = os.path.join(self._annotations_dir, f"{stem}_{i}.png")
            if os.path.exists(path):
                img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                layers.append(img)
                logger.debug("Loaded annotation: %s", path)
            else:
                break

        if len(layers) < nlayers:
            logger.debug(
                "Annotation files incomplete for %s — initialising %d blank layers.",
                filename, nlayers,
            )
            layers = [np.zeros((h, w), dtype=np.uint8) for _ in range(nlayers)]

        return np.array(layers, dtype=np.uint8)
