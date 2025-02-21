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
        self.annotations = self._load_annotations(nlayers)
        self._save_annotations()
        self._saved_states = deque(maxlen=10)
        self._logger.info("Image loaded: %s", self.filename)
    
    def _load_annotations(self, nlayers):
        annotations = []
        for i in range(nlayers):
            filename = os.path.join(ImageLoader.ANNOTATIONS, f"{os.path.splitext(os.path.basename(self.filename))[0]}_{i}.png")
            if os.path.exists(filename):
                img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
                annotations.append(img)
                self._logger.info("Annotations loaded: %s", filename)
            else:
                self._logger.info("Annotations not found: %s", filename)
                break
        if len(annotations) < nlayers:
            self._init_annotations(nlayers)
        return annotations

    def _init_annotations(self, nlayers):
        for i in range(nlayers):
            binary_image = np.zeros((self.height, self.width), dtype=np.uint8)
            self.annotations.append(binary_image)

    def get_annotations_pixmap(self, layer, all_layers=True, invert=False):
        """Returns the annotations as a QPixmap."""

        # Obtener dimensiones de la imagen base
        height, width = self.annotations[0].shape

        # Crear un QPixmap transparente
        pixmap = QPixmap(width, height)
        pixmap.fill(Qt.GlobalColor.transparent)

        painter = QPainter(pixmap)
        painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceOver)

        for i, img in enumerate(self.annotations):
            if i == layer or invert:
                color = (255, 0, 0, 128)
            else:
                if not invert and not all_layers:
                    continue
                color = (0, 0, 255, 128)

            # Convertir imagen binaria en un QImage de 32 bits con canal alfa
            qimage = QImage(width, height, QImage.Format.Format_ARGB32)
            qimage.fill(Qt.GlobalColor.transparent)

            for y in range(height):
                for x in range(width):
                    if img[y, x] > 0 or (invert and img[y, x] == 0):  # Si es blanco en la imagen binaria
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
        self._saved_states.append(saved_annotations)

    def annotate_pixel(self, x, y, layer, save_state=True):
        """Sets the specified pixel to white in layer."""
        if save_state:
            self._save_state()
        self.annotations[layer][y, x] = 255
        for i, img in enumerate(self.annotations):
            if i != layer:
                img[y, x] = 0
        if save_state:
            self._save_annotations()
    
    def undo(self):
        """Restores the previous state of the annotations."""
        if len(self._saved_states) > 0:
            self.annotations = self._saved_states.pop()
            self._save_annotations()
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
       
    def annotate_similar(self, x, y, layer, threshold):
        """Sets the specified area to 1 in layer."""
        self._save_state()
        pixels = self._get_adjacent_pixels(x, y, layer, threshold)
        for px, py in pixels:
            self.annotate_pixel(px, py, layer, save_state=False)
        self._save_annotations()

    def _save_annotations(self):
        """Saves the annotations to separate files."""
        os.makedirs(ImageLoader.ANNOTATIONS, exist_ok=True)
        for i, img in enumerate(self.annotations):
            basename = os.path.splitext(os.path.basename(self.filename))[0]
            filename = os.path.join(ImageLoader.ANNOTATIONS, f"{basename}_{i}.png")
            cv2.imwrite(filename, img)
            self._logger.info("Annotations saved: %s", filename)
    
    def get_progress(self):
        """Returns the percentage of annotated pixels."""
        combined_annotations = np.zeros((self.height, self.width), dtype=np.uint8)
        for img in self.annotations:
            combined_annotations = np.maximum(combined_annotations, img)
        total_pixels = self.height * self.width
        annotated_pixels = np.count_nonzero(combined_annotations)
        percentage = int(annotated_pixels / total_pixels * 100)
        if percentage == 100  and annotated_pixels < total_pixels:
            percentage = 99
        return percentage