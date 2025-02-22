import os
import logging
import cv2
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
        self._load_annotations(nlayers)
        self._save_annotations()
        self._saved_states = deque(maxlen=10)
        self._logger.info("Image loaded: %s", self.filename)
    
    def _load_annotations(self, nlayers):
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
            self._init_annotations(nlayers)

    def _init_annotations(self, nlayers):
        self.annotations = []
        for i in range(nlayers):
            binary_image = np.zeros((self.height, self.width), dtype=np.uint8)
            self.annotations.append(binary_image)
    
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
        annotated = self.get_all_annotations_mask(layer) if not ignore_annotations else np.zeros((height, width), np.uint8)

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
        # set mask to 255 in all values greater than 0
        mask[mask > 0] = 255
        self.annotations[layer] = np.maximum(self.annotations[layer], mask)
        # remove the annotations from other layers
        self._remove_mask_from_other_layers(mask, layer)
        self._save_annotations()
    
    def get_other_annotations_mask(self, layer):
        """Returns the mask with all annotations in all layers."""
        mask = np.zeros((self.height, self.width), np.uint8)
        for i, img in enumerate(self.annotations):
            if i != layer:
                mask = np.maximum(mask, img)
        return mask

    def get_all_annotations_mask(self, layer):
        """Returns the mask with all annotations in all layers."""
        mask = np.zeros((self.height, self.width), np.uint8)
        for img in self.annotations:
            mask = np.maximum(mask, img)
        return mask
    
    def get_unannotated_mask(self, x, y, connected=True):
        """Returns the mask with all unannotated pixels."""
        mask = np.zeros((self.height, self.width), np.uint8)
        for i, img in enumerate(self.annotations):
            mask = np.maximum(mask, img)
        mask = 255 - mask
        if connected:
            mask = self._get_adjacent_pixels(x, y, 0, threshold=255, ignore_annotations=False)
        return mask
    
    def _remove_mask_from_other_layers(self, mask, layer):
        """Removes the specified mask from other layers."""
        # Update other layers
        inverted_mask = 255 - mask
        for i in range(len(self.annotations)):
            if i != layer:
                self.annotations[i] = np.minimum(self.annotations[i], inverted_mask)

    def annotate_similar(self, x, y, layer, threshold):
        """Sets the specified area to 1 in layer."""
        self._save_state()
        mask = self._get_adjacent_pixels(x, y, layer, threshold)

        # Update the specified layer
        self.annotations[layer] = np.maximum(self.annotations[layer], mask)
        # Remove the mask from other layers
        self._remove_mask_from_other_layers(mask, layer)

        self._save_annotations()
    
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