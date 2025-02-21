import sys
import os
from PyQt6.QtWidgets import QApplication, QMainWindow, QListWidget, QPushButton, QLabel, QVBoxLayout, QHBoxLayout, QWidget, QFileDialog, QGraphicsView, QGraphicsScene, QGraphicsPixmapItem, QButtonGroup, QCheckBox, QSlider
from PyQt6.QtGui import QPixmap, QPainter, QPen, QColor, QImage, QMouseEvent
from PyQt6.QtCore import Qt, QPoint
import logging

logging.basicConfig(level=logging.INFO)

from annotations import ImageLoader

class PixelAnnotationApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self._logger = logging.getLogger("PixelAnnotationApp")
        self.setWindowTitle("Pixel Annotation Tool")
        self.setGeometry(100, 100, 1024, 768)
        
        self.zoom = 20  # Zoom inicial (20 veces el tamaño original)
        
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
        self.q_undo_button = QPushButton("Undo")
        self.q_undo_button.clicked.connect(self.cb_undo)  
        toolbar_layout.addWidget(self.q_zoom_in_button)
        toolbar_layout.addWidget(self.q_zoom_out_button)
        toolbar_layout.addWidget(self.q_undo_button)

        # Threshold selector
        toolbar_layout.addWidget(QLabel("Threshold"))
        self.q_nearest_slider = QSlider(Qt.Orientation.Horizontal)
        self.q_nearest_slider.setMinimum(1)
        self.q_nearest_slider.setMaximum(255)
        self.q_nearest_slider.setValue(32)
        self.q_nearest_slider.setTickInterval(16)
        self.q_nearest_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.q_nearest_slider.valueChanged.connect(self.cb_update_threshold)
        toolbar_layout.addWidget(self.q_nearest_slider)

        # Layer buttons
        self.q_layer_0_button = QPushButton("background")
        self.q_layer_0_button.clicked.connect(lambda: self.cb_select_layer(0))
        self.q_layer_0_button.setShortcut("1")
        self.q_layer_1_button = QPushButton("staff")
        self.q_layer_1_button.clicked.connect(lambda: self.cb_select_layer(1))
        self.q_layer_1_button.setShortcut("2")
        self.q_layer_2_button = QPushButton("notes")
        self.q_layer_2_button.clicked.connect(lambda: self.cb_select_layer(2))
        self.q_layer_2_button.setShortcut("3")
        self.q_layer_3_button = QPushButton("lyrics")
        self.q_layer_3_button.clicked.connect(lambda: self.cb_select_layer(3))
        self.q_layer_3_button.setShortcut("4")

        # Crear un grupo de botones exclusivos
        self.button_group = QButtonGroup(self)
        self.button_group.setExclusive(True)

        # Agregar botones al grupo
        self.button_group.addButton(self.q_layer_0_button)
        self.button_group.addButton(self.q_layer_1_button)
        self.button_group.addButton(self.q_layer_2_button)
        self.button_group.addButton(self.q_layer_3_button)

        # Habilitar el comportamiento tipo "toggle"
        for button in self.button_group.buttons():
            button.setCheckable(True)
        self.q_layer_0_button.setChecked(True)

        toolbar_layout.addWidget(QLabel("Layers"))

        # Show other layers
        self.q_other_layers = QCheckBox("Show other layers")
        self.q_other_layers.stateChanged.connect(self.toggle_other_layers)

        toolbar_layout.addWidget(self.q_other_layers)
        toolbar_layout.addWidget(self.q_layer_0_button)
        toolbar_layout.addWidget(self.q_layer_1_button)
        toolbar_layout.addWidget(self.q_layer_2_button)
        toolbar_layout.addWidget(self.q_layer_3_button)

        # Sidebar
        side_layout = QVBoxLayout()
        side_layout.addLayout(toolbar_layout)
        self.image_label = QLabel("Images")
        side_layout.addWidget(self.image_label)
        self.image_list = QListWidget()
        self.image_list.itemClicked.connect(self.load_image)
        side_layout.addWidget(self.image_list)
        
        # Image display
        self.q_image_view = QGraphicsView()
        self.q_image_view.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        self.q_image_view.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.q_image_view.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.q_image_view.wheelEvent = self.wheel_event
        
        self.q_image_scene = QGraphicsScene()
        self.q_image_view.setScene(self.q_image_scene)
        self.q_image = None
        self.q_annotations = None
        self.q_grid = None
        
        image_layout = QVBoxLayout()
        image_layout.addWidget(self.q_image_view)
        
        main_layout.addLayout(side_layout, 1)
        main_layout.addLayout(image_layout, 10)

        self.keyPressEvent = self.key_press_event
        self.keyReleaseEvent = self.key_release_event
        
        # Variables de estado
        self.state = {
            "zoom": 20,
            "image": None,
            "selected_layer": 0,
            "threshold": 32,
            "area_selection_start": None,
            "area_selection_end": None,
        }
        self.listeners = {
            "zoom": [self.update_image_view],
            "image": [self.update_image_view],
            "selected_layer": [self.update_image_view],
        }

        self.load_images()
        
        # Cargar la primera imagen de la lista si hay al menos una
        if self.image_list.count() > 0:
            self.image_list.setCurrentRow(0)
            self.load_image(self.image_list.currentItem())
        
        self.showMaximized()


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
    
    def key_press_event(self, event):
        if event.key() == Qt.Key.Key_Space:
            self.q_annotations.setVisible(False)

    def key_release_event(self, event):
        """Maneja los eventos de teclado para deshacer y hacer zoom."""
        if event.key() == Qt.Key.Key_Z and event.modifiers() == Qt.KeyboardModifier.ControlModifier:
            self.cb_undo()
        elif event.key() == Qt.Key.Key_Plus:
            self.cb_zoom_in()
        elif event.key() == Qt.Key.Key_Minus:
            self.cb_zoom_out()
        elif event.key() == Qt.Key.Key_Space:
            self.q_annotations.setVisible(True)

    def wheel_event(self, event):
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
            # Desplazamiento vertical
            self.q_image_view.verticalScrollBar().setValue(
                self.q_image_view.verticalScrollBar().value() - event.angleDelta().y()
            )

    def load_images(self):
        """Carga las imágenes de la carpeta especificada en la lista."""
        for filename in ImageLoader.get_images():
            self.image_list.addItem(filename)
    
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
        self.q_image.setScale(self.zoom)  # Aplica el zoom inicial (20 veces)
        self.q_image.setPos(0, 0)
        self.q_image_scene.addItem(self.q_image)
        
        # Cargar la imagen de anotaciones
        selected_layer = self.state["selected_layer"]
        q_annotations_pixmap = image.get_annotations_pixmap(selected_layer)
        self.q_annotations = QGraphicsPixmapItem(q_annotations_pixmap)
        self.q_annotations.setScale(self.zoom)  # Aplica el mismo zoom
        self.q_annotations.setPos(0, 0)
        self.q_image_scene.addItem(self.q_annotations)
                      
        # Conectar el evento de clic del mouse
        self.q_image_view.mousePressEvent = self.mouse_press_event
        self.q_image_view.mouseMoveEvent = self.mouse_move_event
        self.q_image_view.mouseReleaseEvent = self.mouse_release_event

        # Estado
        self.annotating = False
        self.start_pixel = None
        self.select_nearest = False
        self.q_nearest_slider.setValue(32)
        self.thresholding = False
        self.q_annotations.setVisible(True)

        self.set_state({
            "zoom": 20,
            "image": image,
            "selected_layer": 0,
            "threshold": 32,
            "area_selection_start": None,
            "area_selection_end": None,
        })
    
    def update_image_view(self):
        """Actualiza la vista de la imagen."""
        if self.state["image"]:
            image = self.state["image"]
            selected_layer = self.state["selected_layer"]
            q_annotations_pixmap = image.get_annotations_pixmap(selected_layer)
            self.q_annotations.setPixmap(q_annotations_pixmap)

            zoom = self.state["zoom"]
            self.q_image.setScale(zoom)  # Aplica el nuevo zoom
            self.q_annotations.setScale(zoom)  # Aplica el mismo zoom a las anotaciones
            scaled_width = self.q_image.pixmap().width() * zoom
            scaled_height = self.q_image.pixmap().height() * zoom
            self.q_image_scene.setSceneRect(0, 0, scaled_width, scaled_height)
            self.update_grid()  # Actualiza la cuadrícula
    
    def cb_update_threshold(self):
        """Establece el umbral para seleccionar los píxeles más cercanos."""
        value = self.q_nearest_slider.value()
        self.set_state({"threshold": value})
    
    def mouse_press_event(self, event: QMouseEvent):
        """Maneja el evento de clic del mouse para anotar píxeles."""

        # Obtener la posición del clic en la escena
        scene_pos = self.q_image_view.mapToScene(event.pos())
        
        # Convertir la posición a coordenadas de píxel en la imagen original
        pixel_x = int(scene_pos.x() / self.zoom)
        pixel_y = int(scene_pos.y() / self.zoom)

        if event.button() == Qt.MouseButton.LeftButton and self.q_annotations:
            # Información para drag
            self.annotating = True
            self.thresholding = False
            self.start_pixel = (pixel_x, pixel_y)

            # Anotar el píxel en la capa 0
            selected_layer = self.state["selected_layer"]
            image = self.state["image"]
            image.annotate_pixel(pixel_x, pixel_y, selected_layer)
            self.set_state({"image": image})
        elif event.button() == Qt.MouseButton.RightButton:
            selected_layer = self.state["selected_layer"]
            self.start_pixel = (pixel_x, pixel_y)
            self.annotating = False
            self.thresholding = True
            image = self.state["image"]
            threshold = self.state["threshold"]
            image.annotate_similar(pixel_x, pixel_y, selected_layer, threshold)
            q_annotations_pixmap = image.get_annotations_pixmap(selected_layer)
            self.q_annotations.setPixmap(q_annotations_pixmap)
    
    def mouse_move_event(self, event: QMouseEvent):
        """Si está anotando y se mueve a un pixel diferente, anota el pixel."""
        # Obtener la posición del clic en la escena
        scene_pos = self.q_image_view.mapToScene(event.pos())
        
        # Convertir la posición a coordenadas de píxel en la imagen original
        pixel_x = int(scene_pos.x() / self.zoom) + 1
        pixel_y = int(scene_pos.y() / self.zoom) + 1

        if self.annotating and self.q_annotations:
            # Si el pixel es diferente al anterior, anótalo
            if (pixel_x, pixel_y) != self.start_pixel:
                self.end_pixel = (pixel_x, pixel_y)
                selected_layer = self.state["selected_layer"]
                image = self.state["image"]
                image.annotate_area(self.start_pixel, self.end_pixel, selected_layer)

                # Actualizar la imagen de anotaciones
                q_annotations_pixmap = image.get_annotations_pixmap(selected_layer)
                self.q_annotations.setPixmap(q_annotations_pixmap)
    
    def mouse_release_event(self, event: QMouseEvent):
        """Detiene la anotación cuando se suelta el botón del mouse."""
        self.annotating = False
        self.thresholding = False
        self.start_pixel = None
    
    def toggle_other_layers(self):
        """Activa o desactiva el selector de vecinos."""
        state = self.q_other_layers.checkState()
        self.select_nearest = (state == Qt.CheckState.Checked)
        self._logger.info("Select nearest pixels: %s", self.select_nearest)

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
    
    def update_zoom(self):
        """Actualiza la escala de la imagen y la cuadrícula según el zoom actual."""
        if self.q_image and self.q_annotations:
            self.q_image.setScale(self.zoom)  # Aplica el nuevo zoom
            self.q_annotations.setScale(self.zoom)  # Aplica el mismo zoom a las anotaciones
            scaled_width = self.q_image.pixmap().width() * self.zoom
            scaled_height = self.q_image.pixmap().height() * self.zoom
            self.q_image_scene.setSceneRect(0, 0, scaled_width, scaled_height)
            self.update_grid()  # Actualiza la cuadrícula
    
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
        
        # Crea un QPixmap para la cuadrícula
        grid_pixmap = QPixmap(width, height)
        grid_pixmap.fill(Qt.GlobalColor.transparent)
        
        painter = QPainter(grid_pixmap)
        if painter.isActive():  # Verifica si el painter está activo
            pen = QPen(QColor("gray"))
            pen.setWidth(1)
            painter.setPen(pen)
            
            # Dibuja la cuadrícula
            for x in range(0, width, zoom):
                painter.drawLine(x, 0, x, height)
            for y in range(0, height, zoom):
                painter.drawLine(0, y, width, y)
            
            painter.end()
        
        # Elimina la cuadrícula anterior si existe
        if self.q_grid:
            self.q_image_scene.removeItem(self.q_grid)
        
        # Añade la nueva cuadrícula
        self.q_grid = QGraphicsPixmapItem(grid_pixmap)
        self.q_image_scene.addItem(self.q_grid)
        self.q_grid.setZValue(2)  # Asegura que la cuadrícula esté sobre las imágenes

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = PixelAnnotationApp()
    window.show()
    sys.exit(app.exec())
