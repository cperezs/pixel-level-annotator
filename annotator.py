import sys
import os
from PyQt6.QtWidgets import QApplication, QMainWindow, QListWidget, QPushButton, QLabel, QVBoxLayout, QHBoxLayout, QWidget, QFileDialog, QGraphicsView, QGraphicsScene, QGraphicsPixmapItem
from PyQt6.QtGui import QPixmap, QPainter, QPen, QColor
from PyQt6.QtCore import Qt

class PixelAnnotationApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Pixel Annotation Tool")
        self.setGeometry(100, 100, 800, 600)
        
        self.zoom = 20  # Zoom inicial (20 veces el tamaño original)
        
        # Layouts
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout()
        main_widget.setLayout(main_layout)
        
        # Sidebar
        self.image_list = QListWidget()
        self.image_list.itemClicked.connect(self.load_image)
        
        # Toolbar
        toolbar_layout = QHBoxLayout()
        self.zoom_in_button = QPushButton("Zoom In")
        self.zoom_in_button.clicked.connect(self.zoom_in)
        self.zoom_out_button = QPushButton("Zoom Out")
        self.zoom_out_button.clicked.connect(self.zoom_out)
        toolbar_layout.addWidget(self.zoom_in_button)
        toolbar_layout.addWidget(self.zoom_out_button)
        
        # Image display
        self.graphics_view = QGraphicsView()
        self.graphics_view.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.graphics_view.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        self.graphics_view.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.graphics_view.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.scene = QGraphicsScene()
        self.graphics_view.setScene(self.scene)
        self.image_item = None
        
        image_layout = QVBoxLayout()
        image_layout.addLayout(toolbar_layout)
        image_layout.addWidget(self.graphics_view)
        
        main_layout.addWidget(self.image_list, 1)
        main_layout.addLayout(image_layout, 4)
        
        self.load_images()
    
    def load_images(self):
        image_folder = "images"
        if os.path.exists(image_folder):
            for filename in os.listdir(image_folder):
                if filename.lower().endswith((".jpg", ".jpeg", ".png", ".gif")):
                    self.image_list.addItem(filename)
    
    def load_image(self, item):
        image_path = os.path.join("images", item.text())
        pixmap = QPixmap(image_path)
        
        if pixmap.isNull():  # Verifica si la imagen se cargó correctamente
            return
        
        self.scene.clear()
        
        self.image_item = QGraphicsPixmapItem(pixmap)
        self.image_item.setScale(self.zoom)  # Aplica el zoom inicial (20 veces)
        self.image_item.setPos(0, 0)
        self.scene.addItem(self.image_item)
        
        # Ajusta el sceneRect para que coincida con el tamaño de la imagen escalada
        scaled_width = pixmap.width() * self.zoom
        scaled_height = pixmap.height() * self.zoom
        self.scene.setSceneRect(0, 0, scaled_width, scaled_height)
        
        self.update_grid()  # Actualiza la cuadrícula
    
    def zoom_in(self):
        if self.zoom < 40:  # Límite máximo de zoom
            self.zoom += 1
            self.update_zoom()
    
    def zoom_out(self):
        if self.zoom > 1:  # Límite mínimo de zoom
            self.zoom -= 1
            self.update_zoom()
    
    def update_zoom(self):
        if self.image_item:
            self.image_item.setScale(self.zoom)  # Aplica el nuevo zoom
            scaled_width = self.image_item.pixmap().width() * self.zoom
            scaled_height = self.image_item.pixmap().height() * self.zoom
            self.scene.setSceneRect(0, 0, scaled_width, scaled_height)
            self.update_grid()  # Actualiza la cuadrícula
    
    def update_grid(self):
        if not self.image_item:
            return
        
        pixmap = self.image_item.pixmap()
        if pixmap.isNull():
            return
        
        # Calcula el tamaño escalado de la imagen
        width = int(pixmap.width() * self.zoom)
        height = int(pixmap.height() * self.zoom)
        
        if width <= 0 or height <= 0:
            return
        
        # Crea un QPixmap para la cuadrícula
        grid_pixmap = QPixmap(width, height)
        grid_pixmap.fill(Qt.GlobalColor.transparent)
        
        painter = QPainter(grid_pixmap)
        pen = QPen(QColor("gray"))
        pen.setWidth(1)
        painter.setPen(pen)
        
        # Dibuja la cuadrícula
        for x in range(0, width, self.zoom):
            painter.drawLine(x, 0, x, height)
        for y in range(0, height, self.zoom):
            painter.drawLine(0, y, width, y)
        
        painter.end()
        
        # Elimina la cuadrícula anterior si existe
        if hasattr(self, 'grid_item') and self.grid_item:
            self.scene.removeItem(self.grid_item)
        
        # Añade la nueva cuadrícula
        self.grid_item = QGraphicsPixmapItem(grid_pixmap)
        self.scene.addItem(self.grid_item)
        self.grid_item.setZValue(1)  # Asegura que la cuadrícula esté sobre la imagen

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = PixelAnnotationApp()
    window.show()
    sys.exit(app.exec())