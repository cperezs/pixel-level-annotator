import os
import logging
import cv2
from PyQt6.QtGui import QPixmap, QImage, QPainter, QColor
from PyQt6.QtCore import Qt
import numpy as np
from collections import deque

class ImageLoader:
    IMAGES = "images"
    ANNOTATIONS = "annotations"
    _logger = logging.getLogger("ImageLoader")

    @staticmethod
    def get_images():
        """Loads the images from the specified folder."""
        images = []
        folder = ImageLoader.IMAGES
        if os.path.exists(folder):
            for filename in os.listdir(folder):
                if filename.lower().endswith((".jpg", ".jpeg", ".png", ".gif")):
                    images.append(filename)
        return images

    @staticmethod
    def load_image(filename):
        """Loads the specified image."""
        return Image(filename, 4)

class Image:
    def __init__(self, filename, nlayers = 4):
        self._logger = logging.getLogger("Image")
        self.filename = os.path.join(ImageLoader.IMAGES, filename)
        # open image with openCV
        self.image = cv2.imread(self.filename)
        self.height, self.width, self.channels = self.image.shape
        self.annotations = []
        for i in range(nlayers):
            binary_image = np.zeros((self.height, self.width), dtype=np.uint8)
            self.annotations.append(binary_image)
        self._saved_annotations = deque(maxlen=10)
        self._logger.info("Image loaded: %s", self.filename)

    def get_annotations_pixmap(self, layer):
        """Returns the annotations as a QPixmap."""

        # Obtener dimensiones de la imagen base
        height, width = self.annotations[0].shape

        # Crear un QPixmap transparente
        pixmap = QPixmap(width, height)
        pixmap.fill(Qt.GlobalColor.transparent)

        painter = QPainter(pixmap)
        painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceOver)

        for i, img in enumerate(self.annotations):
            if i == layer:
                color = (255, 0, 0, 128)
            else:
                color = (0, 0, 255, 128)

            # Convertir imagen binaria en un QImage de 32 bits con canal alfa
            qimage = QImage(width, height, QImage.Format.Format_ARGB32)
            qimage.fill(Qt.GlobalColor.transparent)

            for y in range(height):
                for x in range(width):
                    if img[y, x] > 0:  # Si es blanco en la imagen binaria
                        qimage.setPixelColor(x, y, QColor(*color))

            # Dibujar la imagen en el QPixmap
            painter.drawImage(0, 0, qimage)

        painter.end()
        return pixmap
    
    def _save_state(self):
        """Saves the current state of the annotations."""
        saved_annotations = []
        for img in self.annotations:
            saved_annotations.append(img.copy())
        self._saved_annotations.append(saved_annotations)

    def annotate_pixel(self, x, y, layer):
        """Sets the specified pixel to 1 in layer."""
        self._save_state()
        self.annotations[layer][y, x] = 1
        for i, img in enumerate(self.annotations):
            if i != layer:
                img[y, x] = 0

    def annotate_area(self, start, end, layer):
        """Sets the specified area to 1 in layer."""
        pos0 = (min(start[0], end[0]), min(start[1], end[1]))
        pos1 = (max(start[0], end[0]), max(start[1], end[1]))
        x0, y0 = pos0
        x1, y1 = pos1
        self.annotations[layer][y0:y1, x0:x1] = 1
        for i, img in enumerate(self.annotations):
            if i != layer:
                img[y0:y1, x0:x1] = 0
    
    def undo(self):
        """Restores the previous state of the annotations."""
        if hasattr(self, "_saved_annotations"):
            self.annotations = self._saved_annotations.pop()
        else:
            self._logger.warning("No annotations to undo.")
    
    def _get_adjacent_pixels(self, x, y, layer, threshold):
        """
        Encuentra todos los píxeles adyacentes con una diferencia de color que no supere el threshold.

        :param image: Imagen en formato OpenCV (numpy.ndarray).
        :param x: Coordenada X del píxel inicial.
        :param y: Coordenada Y del píxel inicial.
        :param threshold: Umbral de diferencia de color permitido.
        :return: Lista de coordenadas [(x1, y1), (x2, y2), ...] de píxeles conectados.
        """
        # Asegurar que la imagen está en formato correcto (convertir a escala de grises si es necesario)
        image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

        height, width = image.shape
        start_color = image[y, x]  # Color del píxel inicial
        visited = np.zeros((height, width), dtype=bool)
        result = []
        queue = deque([(x, y)])

        # Desplazamientos en 8 direcciones (para conectar en diagonal también)
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)] # , (-1, -1), (-1, 1), (1, -1), (1, 1)]
        annotated = self.annotations[layer]

        while queue:
            cx, cy = queue.popleft()
            if visited[cy, cx] or annotated[cy, cx]:
                continue
            
            visited[cy, cx] = True
            result.append((cx, cy))

            for dx, dy in directions:
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < width and 0 <= ny < height and not visited[ny, nx]:
                    if abs(int(image[ny, nx]) - int(start_color)) <= threshold:
                        queue.append((nx, ny))

        return result
       
    def annotate_similar(self, x, y, layer, threshold=50):
        """Sets the specified area to 1 in layer."""
        self._save_state()
        pixels = self._get_adjacent_pixels(x, y, layer, threshold)
        for px, py in pixels:
            self.annotate_pixel(px, py, layer)

