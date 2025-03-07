import os
import logging
import cv2
import numpy as np
import time
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
    def load_image(filename, nlayers = 4):
        """Loads the specified image."""
        return Image(filename, nlayers)


class TimeTracker:
    def __init__(self, filename, nlayers = 4):
        self._logger = logging.getLogger(__name__)

        # Load the time spent on each layer
        self._filename = os.path.join(ImageLoader.ANNOTATIONS, f"{os.path.splitext(os.path.basename(filename))[0]}.metadata")
        if os.path.exists(self._filename):
            with open(self._filename, "r") as file:
                times = file.read().splitlines()
                # Each line is the amount of time spent on each layer (in seconds)
                self.times = [int(t) for t in times]
        else:
            self.times = [0] * nlayers
            self.save()

        self._current_layer = None

    def save(self):
        """Saves the time spent on each layer."""
        with open(self._filename, "w") as file:
            for t in self.times:
                file.write(str(int(t)) + "\n")
        self._logger.info("Times saved: %s", self._filename)    

    def change(self, layer):
        """Starts the timer for the specified layer."""
        # Log the previous layer
        if self._current_layer is not None:
            self.times[self._current_layer] += time.time() - self._start_time

        self._start_time = time.time()
        self._current_layer = layer
        self.save()
    
    def tick(self):
        """Increments the time spent on the current layer."""
        self.times[self._current_layer] += time.time() - self._start_time
        self._start_time = time.time()
        self.save()


class Image:
    def __init__(self, filename, nlayers = 4):
        self._logger = logging.getLogger("Image")
        self.filename = os.path.join(ImageLoader.IMAGES, filename)
        # open image with openCV
        self.image = cv2.imread(self.filename)
        self.height, self.width, self.channels = self.image.shape
        self._load_annotations(nlayers)
        self._saved_states = deque(maxlen=20)
        self._logger.info("Image loaded: %s", self.filename)
    
    def _load_annotations(self, nlayers):
        """Loads the annotations from the specified folder
        or creates new annotations if they don't exist."""

        # Load the annotations
        self.annotations = []
        for i in range(nlayers):
            filename = os.path.join(ImageLoader.ANNOTATIONS, f"{os.path.splitext(os.path.basename(self.filename))[0]}_{i}.png")
            if os.path.exists(filename):
                img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
                self.annotations.append(img)
                self._logger.info("Annotations loaded: %s", filename)
            else:
                self._logger.info("Annotations not found: %s", filename)
                break
        if len(self.annotations) < nlayers:
            self.annotations = np.zeros((nlayers, self.height, self.width), dtype=np.uint8)
            self._save_annotations()
    
    def _save_state(self):
        """Saves the current state of the annotations."""
        saved_annotations = []
        for img in self.annotations:
            saved_annotations.append(img.copy())
        self._saved_states.append(saved_annotations)
    
    def undo(self):
        """Restores the previous state of the annotations."""
        if len(self._saved_states) > 0:
            self.annotations = self._saved_states.pop()
            self._save_annotations()
        else:
            self._logger.warning("No annotations to undo.")
    
    def _get_adjacent_pixels(self, x, y, layer, threshold, ignore_annotations=False):
        """
        Encuentra todos los píxeles adyacentes con una diferencia de color que no supere el threshold.

        :param image: Imagen en formato OpenCV (numpy.ndarray).
        :param x: Coordenada X del píxel inicial.
        :param y: Coordenada Y del píxel inicial.
        :param threshold: Umbral de diferencia de color permitido.
        :return: Máscara binaria con los píxeles conectados.
        """
        # Asegurar que la imagen está en formato correcto (convertir a escala de grises si es necesario)
        image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

        height, width = image.shape
        start_color = image[y, x]  # Color del píxel inicial
        visited = np.zeros((height, width), dtype=bool)
        mask = np.zeros((height, width), np.uint8)  # Máscara con bordes adicionales
        queue = deque([(x, y)])

        # Desplazamientos en 8 direcciones (para conectar en diagonal también)
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1) , (-1, -1), (-1, 1), (1, -1), (1, 1)]
        annotated = self.get_all_annotations_mask() if not ignore_annotations else np.zeros((height, width), np.uint8)

        while queue:
            cx, cy = queue.popleft()
            if visited[cy, cx] or annotated[cy, cx]:
                continue

            visited[cy, cx] = True
            mask[cy, cx] = 255

            for dx, dy in directions:
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < width and 0 <= ny < height and not visited[ny, nx]:
                    if abs(int(image[ny, nx]) - int(start_color)) <= threshold:
                        queue.append((nx, ny))

        return mask
    
    def annotate_mask(self, mask, layer):
        """Sets the specified mask to 1 in layer."""
        self._save_state()
        # set mask to 255 in all values greater than 0, just in case the mask is not binary
        mask[mask > 0] = 255
        self.annotations[layer] = np.maximum(self.annotations[layer], mask)
        # remove the annotations from other layers
        self._remove_mask_from_other_layers(layer)
        self._save_annotations()
    
    def get_other_annotations_mask(self, layer):
        """Returns the mask with all annotations in all other layers."""
        mask = np.delete(self.annotations, layer, axis=0)
        mask = np.bitwise_or.reduce(mask, axis=0)
        return mask

    def get_all_annotations_mask(self):
        """Returns the mask with all annotations in all layers."""
        return np.bitwise_or.reduce(self.annotations, axis=0)
    
    def get_missing_annotations_mask(self):
        """Returns the mask with all missing annotations."""
        mask = self.get_all_annotations_mask()
        return np.bitwise_not(mask)
        
    def get_unannotated_mask(self, x, y, connected=True):
        """Returns the mask with all unannotated pixels."""
        if connected:
            return self._get_adjacent_pixels(x, y, 0, threshold=255, ignore_annotations=False)
        else:
            return self.get_missing_annotations_mask()
    
    def _remove_mask_from_other_layers(self, layer):
        """Removes the specified mask from other layers."""
        # Update other layers
        mask = np.bitwise_not(self.annotations[layer])
        for i in range(len(self.annotations)):
            if i != layer:
                self.annotations[i] = np.bitwise_and(self.annotations[i], mask)

    # def annotate_similar(self, x, y, layer, threshold):
    #     """Sets the specified area to 1 in layer."""
    #     self._save_state()
    #     mask = self._get_adjacent_pixels(x, y, layer, threshold)

    #     # Update the specified layer
    #     self.annotations[layer] = np.maximum(self.annotations[layer], mask)
    #     # Remove the mask from other layers
    #     self._remove_mask_from_other_layers(layer)

    #     self._save_annotations()
    
    def get_similarity_mask(self, x, y, layer, threshold, ignore_annotations=False):
        """Returns the mask of the specified area."""
        return self._get_adjacent_pixels(x, y, layer, threshold, ignore_annotations)

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
        combined_annotations = np.any(self.annotations, axis=0).astype(np.uint8) * 255
        total_pixels = self.height * self.width
        annotated_pixels = np.count_nonzero(combined_annotations)
        percentage = int(annotated_pixels / total_pixels * 100)
        if percentage == 100 and annotated_pixels < total_pixels:
            percentage = 99
        return percentage