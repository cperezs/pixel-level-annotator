import sys
import logging
import numpy as np

from PyQt6.QtWidgets import QApplication, QMainWindow, QListWidget, QPushButton, QLabel, QVBoxLayout, QHBoxLayout, QWidget, QFileDialog, QGraphicsView, QGraphicsScene, QGraphicsPixmapItem, QButtonGroup, QCheckBox, QSlider, QFrame, QSizePolicy, QToolBar, QSpinBox
from PyQt6.QtGui import QPixmap, QImage, QMouseEvent
from PyQt6.QtCore import Qt, QPropertyAnimation

import cv2

logging.basicConfig(level=logging.INFO)

from annotations import ImageLoader

class PixelAnnotationApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self._logger = logging.getLogger("PixelAnnotationApp")
        self.setWindowTitle("Pixel Annotation Tool")
        self.setGeometry(100, 100, 1024, 768)
        
        # Layouts
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout()
        main_widget.setLayout(main_layout)
              
        # Toolbar
        toolbar_layout = QVBoxLayout()
        self.q_zoom_in_button = QPushButton("Zoom In")
        self.q_zoom_in_button.clicked.connect(self.cb_zoom_in)
        self.q_zoom_out_button = QPushButton("Zoom Out")
        self.q_zoom_out_button.clicked.connect(self.cb_zoom_out)
        self.q_undo_button = QPushButton("Undo (Ctrl+Z)")
        self.q_undo_button.clicked.connect(self.cb_undo)  
        toolbar_layout.addWidget(self.q_zoom_in_button)
        toolbar_layout.addWidget(self.q_zoom_out_button)
        toolbar_layout.addWidget(self.q_undo_button)

        # Native separator using QToolBar
        q_separator = QToolBar()
        q_separator.addSeparator()
        toolbar_layout.addWidget(q_separator)

        # Tool buttons
        self.q_tool_group = QButtonGroup(self)
        self.q_tool_group.setExclusive(True)
        self.q_pen_button = QPushButton("Pen (p)")
        self.q_pen_button.setCheckable(True)
        self.q_pen_button.setChecked(True)
        self.q_pen_button.setShortcut("p")
        self.q_pen_button.clicked.connect(lambda: self.cb_select_tool("pen"))
        self.q_selector_button = QPushButton("Selector (s)")
        self.q_selector_button.setCheckable(True)
        self.q_selector_button.setShortcut("s")
        self.q_selector_button.clicked.connect(lambda: self.cb_select_tool("selector"))
        self.q_fill_button = QPushButton("Fill (f)")
        self.q_fill_button.setCheckable(True)
        self.q_fill_button.setShortcut("f")
        self.q_fill_button.clicked.connect(lambda: self.cb_select_tool("fill"))
        self.q_tool_group.addButton(self.q_pen_button)
        self.q_tool_group.addButton(self.q_selector_button)
        self.q_tool_group.addButton(self.q_fill_button)

        # Selector ignore annotations
        self.q_ignore_annotations = QCheckBox("Ignore annotations")
        self.q_ignore_annotations.setChecked(False)
        toolbar_layout.addWidget(self.q_ignore_annotations)
        self.q_ignore_annotations.stateChanged.connect(self.cb_ignore_annotations)

        # Pen tool
        toolbar_layout.addWidget(self.q_pen_button)
        self.q_pen_label = QLabel("Pen size")
        toolbar_layout.addWidget(self.q_pen_label)
        self.q_pen_spin = QSpinBox()
        self.q_pen_spin.setMinimum(1)
        self.q_pen_spin.setMaximum(15)
        self.q_pen_spin.setValue(1)
        self.q_pen_spin.valueChanged.connect(self.cb_update_pen_size)
        toolbar_layout.addWidget(self.q_pen_spin)

        # Threshold selector tool
        toolbar_layout.addWidget(self.q_selector_button)
        self.q_threshold_label = QLabel("Threshold")
        toolbar_layout.addWidget(self.q_threshold_label)
        self.q_threshold_slider = QSlider(Qt.Orientation.Horizontal)
        self.q_threshold_slider.setMinimum(1)
        self.q_threshold_slider.setMaximum(128)
        self.q_threshold_slider.setValue(32)
        self.q_threshold_slider.setTickInterval(16)
        self.q_threshold_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.q_threshold_slider.valueChanged.connect(self.cb_update_threshold)
        toolbar_layout.addWidget(self.q_threshold_slider)

        # Selector auto-smoothing
        self.q_autosmooth = QCheckBox("Auto-smooth")
        self.q_autosmooth.setChecked(True)
        toolbar_layout.addWidget(self.q_autosmooth)
        self.q_autosmooth.stateChanged.connect(self.cb_autosmooth)

        # Fill tool
        toolbar_layout.addWidget(self.q_fill_button)

        # Native separator using QToolBar
        q_separator = QToolBar()
        q_separator.addSeparator()
        toolbar_layout.addWidget(q_separator)

        # Layer buttons
        self.q_layer_0_button = QPushButton("background (1)")
        self.q_layer_0_button.clicked.connect(lambda: self.cb_select_layer(0))
        self.q_layer_0_button.setShortcut("1")
        self.q_layer_1_button = QPushButton("staff (2)")
        self.q_layer_1_button.clicked.connect(lambda: self.cb_select_layer(1))
        self.q_layer_1_button.setShortcut("2")
        self.q_layer_2_button = QPushButton("notes (3)")
        self.q_layer_2_button.clicked.connect(lambda: self.cb_select_layer(2))
        self.q_layer_2_button.setShortcut("3")
        self.q_layer_3_button = QPushButton("lyrics (4)")
        self.q_layer_3_button.clicked.connect(lambda: self.cb_select_layer(3))
        self.q_layer_3_button.setShortcut("4")

        # Crear un grupo de botones exclusivos
        self.q_button_group = QButtonGroup(self)
        self.q_button_group.setExclusive(True)

        # Agregar botones al grupo
        self.q_button_group.addButton(self.q_layer_0_button)
        self.q_button_group.addButton(self.q_layer_1_button)
        self.q_button_group.addButton(self.q_layer_2_button)
        self.q_button_group.addButton(self.q_layer_3_button)

        # Habilitar el comportamiento tipo "toggle"
        for button in self.q_button_group.buttons():
            button.setCheckable(True)
        self.q_layer_0_button.setChecked(True)

        toolbar_layout.addWidget(QLabel("Layers"))

        # Show original image
        self.q_show_image = QCheckBox("Show image (i)")
        self.q_show_image.setChecked(True)
        self.q_show_image.setShortcut("i")
        self.q_show_image.stateChanged.connect(self.cb_show_image)

        # Show other layers
        self.q_other_layers = QCheckBox("Show other layers (o)")
        self.q_other_layers.setChecked(True)
        self.q_other_layers.setShortcut("o")
        self.q_other_layers.stateChanged.connect(self.cb_toggle_other_layers)

        toolbar_layout.addWidget(self.q_show_image)
        toolbar_layout.addWidget(self.q_other_layers)
        toolbar_layout.addWidget(self.q_layer_0_button)
        toolbar_layout.addWidget(self.q_layer_1_button)
        toolbar_layout.addWidget(self.q_layer_2_button)
        toolbar_layout.addWidget(self.q_layer_3_button)

        # Sidebar
        side_layout = QVBoxLayout()
        side_layout.addLayout(toolbar_layout)
        self.q_image_label = QLabel("Images")
        side_layout.addWidget(self.q_image_label)
        self.q_image_list = QListWidget()
        self.q_image_list.itemClicked.connect(self.load_image)
        side_layout.addWidget(self.q_image_list)
        
        # Image display
        self.q_image_view = QGraphicsView()
        self.q_image_view.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        self.q_image_view.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.q_image_view.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.q_image_view.wheelEvent = self.cb_wheel_event
        
        self.q_image_scene = QGraphicsScene()
        self.q_image_view.setScene(self.q_image_scene)
        self.q_image_view.mousePressEvent = self.cb_mouse_press_event
        self.q_image_view.mouseMoveEvent = self.cb_mouse_move_event
        self.q_image_view.mouseReleaseEvent = self.cb_mouse_release_event
        self.q_image_view.setMouseTracking(True)
        self.q_image = None
        self.q_annotations = None
        self.q_grid = None
        
        image_layout = QVBoxLayout()

        # Progress bar
        self.q_progress_bar = QLabel("Progress")
        self.q_progress_bar.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.q_progress_bar.setText("0%")

        image_layout.addWidget(self.q_progress_bar)
        image_layout.addWidget(self.q_image_view)
        
        main_layout.addLayout(side_layout, 1)
        main_layout.addLayout(image_layout, 10)

        self.keyPressEvent = self.cb_key_press_event
        self.keyReleaseEvent = self.cb_key_release_event
        
        # Variables de estado
        self.state = {
            "zoom": 10,
            "image": None,
            "selected_layer": 0,
            "area_selection_start": None,
            "area_selection_end": None,
            "show_other_layers": True,
            "show_image": True,
            "mask": None,
            "pen_tool": True,
            "pen_tool_drawing": False,
            "pen_tool_size": 1,
            "selector_tool": False,
            "selector_tool_drawing": False,
            "selector_tool_threshold": 32,
            "selector_tool_auto_smooth": True,
            "fill_tool": False,
            "ignore_annotations": False,
            "mouse_pos": None
        }
        self.listeners = {
            "zoom": [self.update_image_view],
            "image": [self.update_image_view],
            "selected_layer": [self.update_image_view],
            "show_other_layers": [self.update_image_view],
            "show_image": [self.update_image_view],
            "mask": [self.update_mask_view],
            "pen_tool": [self.tool_change],
            "selector_tool": [self.tool_change],
            "mouse_pos": [self.mouse_pos_updated],
            "selector_tool_threshold": [self.update_threshold],
            "ignore_annotations": [self.update_ignore_annotations]
        }

        self.load_images()
        
        # Cargar la primera imagen de la lista si hay al menos una
        if self.q_image_list.count() > 0:
            self.q_image_list.setCurrentRow(0)
            self.load_image(self.q_image_list.currentItem())
        
        self.showMaximized()

    def cb_select_tool(self, tool):
        self.set_state({"pen_tool": tool == "pen", "selector_tool": tool == "selector", "fill_tool": tool == "fill"})

    def set_state(self, state):
        """Establece el estado de la aplicación."""
        self._logger.info("State: %s", state)
        self.state.update(state)
        keys = set()
        for key, value in state.items():
            keys.add(key)
        self.notify(keys)
    
    def register_listener(self, key, listener):
        """Registra un listener para un estado específico."""
        self.listeners[key].append(listener)
    
    def notify(self, keys):
        listeners = set()
        for key in keys:
            if key in self.listeners:
                for listener in self.listeners[key]:
                    listeners.add(listener)
        for listener in listeners:
            listener()
    
    def cb_key_press_event(self, event):
        if event.key() == Qt.Key.Key_Space and not event.isAutoRepeat():
            self.q_annotations.setVisible(False)
            self.set_state({"ignore_annotations": True})

    def cb_key_release_event(self, event):
        """Maneja los eventos de teclado para deshacer y hacer zoom."""
        if event.key() == Qt.Key.Key_Z and event.modifiers() == Qt.KeyboardModifier.ControlModifier:
            if self.state["pen_tool_drawing"] or self.state["selector_tool_drawing"]:
                self.cancel_tool()
            else:
                self.cb_undo()
        elif event.key() == Qt.Key.Key_Plus:
            #self.cb_zoom_in()
            if self.state["pen_tool"]:
                self.update_pen_size(1)
            elif self.state["selector_tool"]:
                self.set_state({"selector_tool_threshold": self.state["selector_tool_threshold"] + 1})
        elif event.key() == Qt.Key.Key_Minus:
            #self.cb_zoom_out()
            if self.state["pen_tool"]:
                self.update_pen_size(-1)
            elif self.state["selector_tool"]:
                self.set_state({"selector_tool_threshold": self.state["selector_tool_threshold"] - 1})
        elif event.key() == Qt.Key.Key_Space:
            self.q_annotations.setVisible(True)
            self.set_state({"ignore_annotations": False})
        elif event.key() == Qt.Key.Key_Escape:
            self.cancel_tool()
        elif event.key() == Qt.Key.Key_E:
            if self.state["selector_tool"] and self.state["selector_tool_drawing"]:
                self.expand_mask()
        elif event.key() == Qt.Key.Key_R:
            if self.state["selector_tool"] and self.state["selector_tool_drawing"]:
                self.shrink_mask()
        elif event.key() == Qt.Key.Key_Return or event.key() == Qt.Key.Key_Enter:
            if self.state["selector_tool"] and self.state["selector_tool_drawing"]:
                self.annotate_drawing()

    def cb_wheel_event(self, event):
        """Maneja el evento de la rueda del mouse para hacer zoom o desplazarse."""
        if event.modifiers() == Qt.KeyboardModifier.ShiftModifier:
            # Desplazamiento horizontal
            self.q_image_view.horizontalScrollBar().setValue(
                self.q_image_view.horizontalScrollBar().value() - event.angleDelta().y()
            )
        elif event.modifiers() == Qt.KeyboardModifier.ControlModifier:
            # Zoom
            if event.angleDelta().y() > 0:
                self.cb_zoom_in()
            else:
                self.cb_zoom_out()
        else:
            # Desplazamiento
            self.q_image_view.verticalScrollBar().setValue(
                self.q_image_view.verticalScrollBar().value() - event.angleDelta().y()
            )
            self.q_image_view.horizontalScrollBar().setValue(
                self.q_image_view.horizontalScrollBar().value() - event.angleDelta().x()
            )

    def load_images(self):
        """Carga las imágenes de la carpeta especificada en la lista."""
        for filename in ImageLoader.get_images():
            self.q_image_list.addItem(filename)
    
    def load_image(self, item):
        """Carga la imagen seleccionada en el visor."""
        if not item:
            return
        
        image = ImageLoader.load_image(item.text())
        q_pixmap = QPixmap(image.filename)
        
        if q_pixmap.isNull():  # Verifica si la imagen se cargó correctamente
            return
        
        # Reinicializar la escena
        self.q_image_scene.clear()
        self.q_image = None
        self.q_annotations = None
        self.q_grid = None
        
        # Cargar la nueva imagen
        self.q_image = QGraphicsPixmapItem(q_pixmap)
        zoom = 10
        self.q_image.setScale(zoom)  # Aplica el zoom inicial (20 veces)
        self.q_image.setPos(0, 0)
        self.q_image_scene.addItem(self.q_image)
        
        # Cargar la imagen de anotaciones
        selected_layer = self.state["selected_layer"]
        show_other_layers = self.state["show_other_layers"]
        ann_img = self.get_annotations_image(image, selected_layer, show_other_layers)
        alto, ancho = ann_img.shape[:2]
        bytes_per_line = ancho * 4
        qimage = QImage(ann_img, ancho, alto, bytes_per_line, QImage.Format.Format_ARGB32)
        q_annotations_pixmap = QPixmap.fromImage(qimage)
        self.q_annotations = QGraphicsPixmapItem(q_annotations_pixmap)
        self.q_annotations.setScale(zoom)  # Aplica el mismo zoom
        self.q_annotations.setPos(0, 0)
        self.q_image_scene.addItem(self.q_annotations)

        # Añade otra imagen para la máscara de selección encima de las anotaciones
        self.q_selection = QGraphicsPixmapItem()
        self.q_image_scene.addItem(self.q_selection)
        self.q_selection.setZValue(1)  # Asegura que la máscara de selección esté sobre las anotaciones
        self.q_selection.setVisible(False)

        # Create a mask with the size of the image
        mask = np.zeros((image.height, image.width), dtype=np.uint8)
                      
        # Estado
        self.annotating = False
        self.start_pixel = None
        self.select_nearest = False
        self.q_threshold_slider.setValue(32)
        self.thresholding = False
        self.q_annotations.setVisible(True)

        # set focus to the image view
        self.q_image_view.setFocus()

        self.set_state({
            "zoom": zoom,
            "image": image,
            "mask": mask,
            "selected_layer": 0,
            "area_selection_start": None,
            "area_selection_end": None,
            "pen_tool": True,
            "selector_tool": False,
        })
    
    def update_image_view(self):
        """Actualiza la vista de la imagen."""
        if self.state["image"]:
            # obtiene la posición de la parte visible de la imagen
            hbar = self.q_image_view.horizontalScrollBar()
            vbar = self.q_image_view.verticalScrollBar()
            hvalue = hbar.value()
            vvalue = vbar.value()
            # Calcula la posición del punto central de la vista
            hcenter = hvalue + self.q_image_view.width() // 2
            vcenter = vvalue + self.q_image_view.height() // 2

            # Show annotations
            image = self.state["image"]
            selected_layer = self.state["selected_layer"]
            show_other_layers = self.state["show_other_layers"]
            show_image = self.state["show_image"]
            ann_img = self.get_annotations_image(image, selected_layer, show_other_layers)
            alto, ancho = ann_img.shape[:2]
            bytes_per_line = ancho * 4
            qimage = QImage(ann_img.data, ancho, alto, bytes_per_line, QImage.Format.Format_ARGB32)
            q_annotations_pixmap = QPixmap.fromImage(qimage)
            self.q_annotations.setPixmap(q_annotations_pixmap)

            # Show progress
            progress = image.get_progress()
            self.q_progress_bar.setText(f"{progress}%")

            zoom = self.state["zoom"]
            self.q_image.setScale(zoom)  # Aplica el nuevo zoom
            self.q_image.setVisible(show_image)
            self.q_annotations.setScale(zoom)  # Aplica el mismo zoom a las anotaciones
            self.q_selection.setScale(zoom)  # Aplica el mismo zoom a la máscara de selección

            scaled_width = self.q_image.pixmap().width() * zoom
            scaled_height = self.q_image.pixmap().height() * zoom
            self.q_image_scene.setSceneRect(0, 0, scaled_width, scaled_height)
            # Ajusta la posición de la vista para mantener el punto central
            hbar.setValue(hcenter - self.q_image_view.width() // 2)
            vbar.setValue(vcenter - self.q_image_view.height() // 2)

            self.update_grid()  # Actualiza la cuadrícula
    
    def update_mask_view(self):
        # Show mask
        mask = self.state["mask"]
        if mask is not None:
            _, width = mask.shape[:2]
            bytes_per_line = width * 4
            mask_img = np.zeros((mask.shape[0], mask.shape[1], 4), dtype=np.uint8)
            mask_img[:, :, :3] = mask[:, :, None]
            mask_img[:, :, 3] = mask * 0.75 # Set opacity to 50%
            qmask = QImage(mask_img.data, mask.shape[1], mask.shape[0], bytes_per_line, QImage.Format.Format_ARGB32)
            qmask_pixmap = QPixmap.fromImage(qmask)
            self.q_selection.setPixmap(qmask_pixmap)
            self.q_selection.setScale(self.state["zoom"])
            self.q_selection.setPos(0, 0)
            self.q_selection.setVisible(True)
        else:
            self.q_selection.setVisible(False)
    
    def cb_update_pen_size(self):
        self.set_state({"pen_tool_size": self.q_pen_spin.value()})

    def cb_update_threshold(self):
        """Establece el umbral para seleccionar los píxeles más cercanos."""
        value = self.q_threshold_slider.value()
        self.set_state({"selector_tool_threshold": value})
    
    def cb_mouse_press_event(self, event: QMouseEvent):
        """Maneja el evento de clic del mouse para anotar píxeles."""

        # Obtener la posición del clic en la escena
        scene_pos = self.q_image_view.mapToScene(event.pos())
        
        # Convertir la posición a coordenadas de píxel en la imagen original
        zoom = self.state["zoom"]
        pixel_x = int(scene_pos.x() / zoom)
        pixel_y = int(scene_pos.y() / zoom)

        image = self.state["image"]
        selected_layer = self.state["selected_layer"]

        if event.button() == Qt.MouseButton.LeftButton and image:
            # Anotar el píxel en la capa seleccionada
            #image.annotate_pixel(pixel_x, pixel_y, selected_layer)
            #self.set_state({"image": image})
            if self.state["pen_tool"]:
                self.set_state({"pen_tool_drawing": True})
            elif self.state["selector_tool"]:
                self.set_state({"selector_tool_drawing": True, "selector_tool_position": (pixel_x, pixel_y)})
                self.mask_selection(pixel_x, pixel_y, selected_layer)
            elif self.state["fill_tool"]:
                self.fill(pixel_x, pixel_y, selected_layer)
    
    def cb_mouse_release_event(self, event: QMouseEvent):
        if event.button() == Qt.MouseButton.LeftButton:
            if self.state["pen_tool"] and self.state["pen_tool_drawing"]:
                self.annotate_drawing()
                self.set_state({"pen_tool_drawing": False})

    def cb_mouse_move_event(self, event: QMouseEvent):
        # Get the position of the mouse in the scene
        scene_pos = self.q_image_view.mapToScene(event.pos())
        
        # Convert the position to pixel coordinates in the original image
        zoom = self.state["zoom"]
        pixel_x = int(scene_pos.x() / zoom)
        pixel_y = int(scene_pos.y() / zoom)
        self._logger.info("Mouse move: (%d, %d)", pixel_x, pixel_y)
        self.set_state({"mouse_pos": (pixel_x, pixel_y)})
    
    def cb_toggle_other_layers(self):
        """Activa o desactiva el selector de vecinos."""
        state = self.q_other_layers.checkState()
        self.set_state({"show_other_layers": state == Qt.CheckState.Checked})
    
    def cb_show_image(self):
        """Activa o desactiva la visualización de la imagen original."""
        state = self.q_show_image.checkState()
        self.set_state({"show_image": state == Qt.CheckState.Checked})

    def cb_select_layer(self, layer):
        """Selecciona la capa de anotación."""
        self.set_state({"selected_layer": layer})
    
    def cb_undo(self):
        """Deshace la última anotación."""
        image = self.state["image"]
        image.undo()
        self.set_state({"image": image})

    def cb_zoom_in(self):
        """Aumenta el zoom."""
        zoom = self.state["zoom"]
        if zoom < 40:  # Límite máximo de zoom
            self.set_state({"zoom": zoom + 5})
    
    def cb_zoom_out(self):
        """Disminuye el zoom."""
        zoom = self.state["zoom"]
        if zoom > 5:  # Límite mínimo de zoom
            self.set_state({"zoom": zoom - 5})
    
    def update_grid(self):
        """Dibuja una cuadrícula sobre la imagen."""
        if not self.q_image:
            return

        pixmap = self.q_image.pixmap()
        if pixmap.isNull():
            return

        # Calcula el tamaño escalado de la imagen
        zoom = self.state["zoom"]
        width = int(pixmap.width() * zoom)
        height = int(pixmap.height() * zoom)

        if width <= 0 or height <= 0:
            return

        # Crear una cuadrícula usando NumPy
        grid = np.zeros((height, width, 4), dtype=np.uint8)
        grid[..., 3] = 0  # Establecer el canal alfa a 0 (completamente transparente)
        grid[::zoom, :, :3] = 128  # Líneas horizontales grises
        grid[:, ::zoom, :3] = 128  # Líneas verticales grises
        grid[::zoom, :, 3] = 255  # Establecer el canal alfa a 255 (completamente opaco) para las líneas horizontales
        grid[:, ::zoom, 3] = 255  # Establecer el canal alfa a 255 (completamente opaco) para las líneas verticales

        # Convertir la cuadrícula a QPixmap
        qimage = QImage(grid.data, width, height, QImage.Format.Format_RGBA8888)
        grid_pixmap = QPixmap.fromImage(qimage)

        # Elimina la cuadrícula anterior si existe
        if self.q_grid:
            self.q_image_scene.removeItem(self.q_grid)

        # Añade la nueva cuadrícula
        self.q_grid = QGraphicsPixmapItem(grid_pixmap)
        self.q_image_scene.addItem(self.q_grid)
        self.q_grid.setZValue(2)  # Asegura que la cuadrícula esté sobre las imágenes

    def get_annotations_image(self, image, layer, all_layers=True, opacity=128):
        height, width = image.annotations[0].shape
        highlight = image.annotations[layer]

        # Create a 4-channel image assigning red to the highlight and blue to the others, with 50% transparency
        combined = np.zeros((height, width, 4), dtype=np.uint8)
        combined[:, :, 2] = highlight
        combined[:, :, 3] = highlight

        if all_layers:
            others = np.max(np.stack([image.annotations[i] for i in range(len(image.annotations)) if i != layer]), axis=0)
            combined[:, :, 0] = others
            combined[:, :, 3] = np.maximum(combined[:, :, 3], others)

        # Set the alpha channel to the specified opacity for non-transparent pixels
        combined[combined[:, :, 3] > 0, 3] = opacity

        return combined

    def mask_selection(self, x, y, layer):
        """Selecciona los píxeles adyacentes a la posición especificada."""
        image = self.state["image"]
        threshold = self.state["selector_tool_threshold"]
        ignore_annotations = self.state["ignore_annotations"]
        mask = image.get_similarity_mask(x, y, layer, threshold, ignore_annotations)
        kernel = np.ones((3, 3), dtype=np.uint8)
        if self.state["selector_tool_auto_smooth"]:
            mask = cv2.dilate(mask, kernel, iterations=1)
            mask = cv2.erode(mask, kernel, iterations=1)
        self.set_state({"mask": mask})
    
    def tool_change(self):
        pen  = self.state["pen_tool"]
        selector = self.state["selector_tool"]

        if pen:
            self.q_pen_button.setChecked(True)
            self.q_pen_label.setEnabled(True)
            self.q_pen_spin.setEnabled(True)
        elif selector:
            self.q_pen_label.setEnabled(False)
            self.q_pen_spin.setEnabled(False)

        if selector:
            self.q_selector_button.setChecked(True)
            self.q_threshold_label.setEnabled(True)
            self.q_threshold_slider.setEnabled(True)
            self.q_autosmooth.setEnabled(True)
        else:
            self.q_threshold_label.setEnabled(False)
            self.q_threshold_slider.setEnabled(False)
            self.q_autosmooth.setEnabled(False)

        mask = np.zeros_like(self.state["mask"])
        self.set_state({"mask": mask})
        self.mouse_pos_updated()

    def annotate_drawing(self):
        """Annotate the drawing in the mask."""
        mask = self.state["mask"]
        image = self.state["image"]
        layer = self.state["selected_layer"]
        if not self.state["ignore_annotations"]:
            annotated = image.get_other_annotations_mask(layer)
            mask = cv2.bitwise_and(mask, cv2.bitwise_not(annotated))
        image.annotate_mask(mask, layer)
        mask = np.zeros_like(mask)
        self.set_state({"mask": mask, "image": image})
    
    def update_pen_size(self, delta):
        """Update the pen size."""
        size = self.state["pen_tool_size"]
        size = max(1, size + delta)
        self.set_state({"pen_tool_size": size})
        self.q_pen_spin.setValue(size)
        self.mouse_pos_updated()
    
    def cancel_tool(self):
        mask = np.zeros_like(self.state["mask"])
        self.set_state({"mask": mask, "pen_tool_drawing": False, "selector_tool_drawing": False})
    
    def mouse_pos_updated(self):
        """Update the mouse position."""
        pos = self.state.get("mouse_pos")
        if pos is None:
            return
        
        pixel_x, pixel_y = pos

        if self.state["pen_tool"] and not self.state["pen_tool_drawing"]:
            mask = np.zeros((self.state["image"].height, self.state["image"].width), dtype=np.uint8)
            # set mask to a square of size pen_tool_size with the top-left corner at (pixel_x, pixel_y)
            size = self.state["pen_tool_size"] 
            mask[pixel_y:pixel_y+size, pixel_x:pixel_x+size] = 225
            self.set_state({"mask": mask})
        elif self.state["pen_tool"] and self.state["pen_tool_drawing"]:
            mask = self.state["mask"]
            size = self.state["pen_tool_size"] 
            mask[pixel_y:pixel_y+size, pixel_x:pixel_x+size] = 255
            if not self.state["ignore_annotations"]:
                annotated = self.state["image"].get_other_annotations_mask(self.state["selected_layer"])
                mask = cv2.bitwise_and(mask, cv2.bitwise_not(annotated))
            self.set_state({"mask": mask})
    
    def update_threshold(self):
        """Update the threshold for the selector tool."""
        threshold = self.state["selector_tool_threshold"]
        self.q_threshold_slider.setValue(threshold)
        if self.state["selector_tool_drawing"]:
            pixel_x, pixel_y = self.state["selector_tool_position"]
            selected_layer = self.state["selected_layer"]
            self.mask_selection(pixel_x, pixel_y, selected_layer)
    
    def expand_mask(self):
        """
        Expande la máscara binaria haciendo crecer los píxeles con valor 255 a los píxeles adyacentes.

        :param mask: numpy.ndarray de forma (alto, ancho) con valores 0 y 255.
        :return: Máscara expandida, donde cada píxel 255 se ha extendido a sus vecinos.
        """
        mask = self.state["mask"]
        # Definir un kernel de 3x3 (vecindario inmediato)
        kernel = np.ones((3, 3), dtype=np.uint8)
        # Aplicar la dilatación: cada píxel 255 "crece" a los píxeles vecinos
        mask = cv2.dilate(mask, kernel, iterations=1)
        if not self.state["ignore_annotations"]:
            annotated = self.state["image"].get_other_annotations_mask(self.state["selected_layer"])
            # Eliminar los píxeles que coinciden con las anotaciones
            mask = cv2.bitwise_and(mask, cv2.bitwise_not(annotated))
            
        self.set_state({"mask": mask})
    
    def shrink_mask(self):
        """
        Reduce la máscara binaria haciendo decrecer los píxeles con valor 255 a los píxeles adyacentes.

        :param mask: numpy.ndarray de forma (alto, ancho) con valores 0 y 255.
        :return: Máscara reducida, donde cada píxel 255 se ha reducido a sus vecinos.
        """
        mask = self.state["mask"]
        # Definir un kernel de 3x3 (vecindario inmediato)
        kernel = np.ones((3, 3), dtype=np.uint8)
        # Aplicar la erosión: cada píxel 255 "encoge" a los píxeles vecinos
        mask = cv2.erode(mask, kernel, iterations=1)
        self.set_state({"mask": mask})
    
    def cb_autosmooth(self):
        """Toggle auto-smoothing for the selector tool."""
        self.set_state({"selector_tool_auto_smooth": self.q_autosmooth.isChecked()})

    def cb_ignore_annotations(self):
        """Toggle ignoring annotations for the selector tool."""
        self.set_state({"ignore_annotations": self.q_ignore_annotations.isChecked()})
    
    def update_ignore_annotations(self):
        ignore = self.state["ignore_annotations"]
        self.q_ignore_annotations.setChecked(ignore)
        self.mouse_pos_updated()
    
    def fill(self, x, y, layer):
        """Fill the selected area with the selected layer."""
        image = self.state["image"]
        mask = image.get_unannotated_mask(x, y)
        image.annotate_mask(mask, layer)
        self.set_state({"image": image})

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = PixelAnnotationApp()
    window.show()
    sys.exit(app.exec())
