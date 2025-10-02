import sys, json, cv2, csv
import os
import mediapipe as mp
from PySide6.QtWidgets import (
    QApplication, QGraphicsScene, QGraphicsView,
    QGraphicsPixmapItem, QGraphicsItem, QPushButton,
    QVBoxLayout, QWidget, QGraphicsEllipseItem, QSlider,
    QLabel, QHBoxLayout
)
from PySide6.QtGui import QPixmap, QImage, QPen, QColor
from PySide6.QtCore import Qt

mp_face_mesh = mp.solutions.face_mesh

class FilterTool(QWidget):
    def __init__(self, face_img, filter_img,annotation_file):
        super().__init__()
        self.setWindowTitle("Face Filter Mapper")

        self.face_img = face_img
        self.filter_img = filter_img
        self.annotation_file=annotation_file
        self.landmarks = []  # list of (id, x, y)

        # Scene + View
        self.scene = QGraphicsScene()
        self.view = QGraphicsView(self.scene)

        # Load face
        self.face_qpix = self.cv2_to_qpixmap(cv2.imread(face_img))
        self.face_item = self.scene.addPixmap(self.face_qpix)
        self.landmark_items = []
        self.face_group = None

        # Extract mediapipe landmarks
        self.detect_landmarks()

        # Draw landmarks
        self.draw_landmarks()
        # Group face image and its landmarks so they scale together
        if self.landmark_items:
            self.face_group = self.scene.createItemGroup([self.face_item, *self.landmark_items])
        else:
            self.face_group = self.scene.createItemGroup([self.face_item])
        # Scale around center
        self.face_group.setTransformOriginPoint(self.face_item.boundingRect().center())

        # Filter image
        self.filter_item = QGraphicsPixmapItem(QPixmap(filter_img))
        self.filter_item.setFlags(
            QGraphicsItem.ItemIsMovable |
            QGraphicsItem.ItemIsSelectable |
            QGraphicsItem.ItemIsFocusable
        )
        # Scale around center
        self.filter_item.setTransformOriginPoint(self.filter_item.boundingRect().center())
        self.scene.addItem(self.filter_item)

        # Save button
        save_btn = QPushButton("Save Mapping")
        save_btn.clicked.connect(self.save_mapping)

        # Layout
        layout = QVBoxLayout()
        layout.addWidget(self.view)

        # Face scale controls
        face_scale_row = QHBoxLayout()
        face_scale_label = QLabel("Face scale (%):")
        self.face_scale_slider = QSlider(Qt.Horizontal)
        self.face_scale_slider.setRange(10, 300)
        self.face_scale_slider.setValue(100)
        self.face_scale_slider.valueChanged.connect(self.on_face_scale_changed)
        face_scale_row.addWidget(face_scale_label)
        face_scale_row.addWidget(self.face_scale_slider)
        layout.addLayout(face_scale_row)

        # Filter scale controls
        filter_scale_row = QHBoxLayout()
        filter_scale_label = QLabel("Filter scale (%):")
        self.filter_scale_slider = QSlider(Qt.Horizontal)
        self.filter_scale_slider.setRange(10, 300)
        self.filter_scale_slider.setValue(100)
        self.filter_scale_slider.valueChanged.connect(self.on_filter_scale_changed)
        filter_scale_row.addWidget(filter_scale_label)
        filter_scale_row.addWidget(self.filter_scale_slider)
        layout.addLayout(filter_scale_row)

        layout.addWidget(save_btn)
        self.setLayout(layout)

    def cv2_to_qpixmap(self, cv_img):
        """Convert cv2 image (BGR) to QPixmap"""
        rgb_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_img.shape
        qimg = QImage(rgb_img.data, w, h, ch * w, QImage.Format_RGB888)
        return QPixmap.fromImage(qimg)

    def detect_landmarks(self):
        """Run MediaPipe FaceMesh on the image"""
        img = cv2.imread(self.face_img)
        h, w, _ = img.shape
        with mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True
        ) as face_mesh:
            results = face_mesh.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            if results.multi_face_landmarks:
                for lm_id, lm in enumerate(results.multi_face_landmarks[0].landmark):
                    x, y = int(lm.x * w), int(lm.y * h)
                    self.landmarks.append((lm_id, x, y))

    def draw_landmarks(self):
        """Draw circles for each landmark"""
        pen = QPen(QColor("red"))
        for lm_id, x, y in self.landmarks:
            ellipse = QGraphicsEllipseItem(x-2, y-2, 4, 4)
            ellipse.setPen(pen)
            ellipse.setToolTip(f"ID: {lm_id}")
            self.scene.addItem(ellipse)
            self.landmark_items.append(ellipse)

    def on_face_scale_changed(self, value):
        scale_factor = value / 100.0
        if self.face_group is not None:
            self.face_group.setScale(scale_factor)

    def on_filter_scale_changed(self, value):
        scale_factor = value / 100.0
        self.filter_item.setScale(scale_factor)

    def save_mapping(self):
        # Build CSV rows for points that fall inside the filter image bounds
        rows = []
        pixmap = self.filter_item.pixmap()
        fw, fh = pixmap.width(), pixmap.height()
        # Get the top-left position of the filter pixmap in scene coordinates
        filter_top_left = self.filter_item.mapToScene(0, 0)
        fx, fy = filter_top_left.x(), filter_top_left.y()
        print(f"{fx:.2f} {fy:.2f} {fw:.2f} {fh:.2f}")
        for (lm_id, _x, _y), ellipse in zip(self.landmarks, self.landmark_items):
            # Use the ellipse center in scene coordinates
            scene_pt = ellipse.sceneBoundingRect().center()
            # Map to filter's local coordinates (accounts for translation and scaling)
            local_pt = self.filter_item.mapFromScene(scene_pt)
            lx, ly = local_pt.x(), local_pt.y()
            print(f"Converted from {scene_pt.x()},{scene_pt.y()} to {lx:.2f},{ly:.2f} from {fx:.2f} {fy:.2f}")
            if 0 <= lx <= fw and 0 <= ly <= fh:
                rows.append((lm_id, int(round(lx)), int(round(ly))))

        # Write CSV (id,x,y), one row per point inside filter
        with open(self.annotation_file, "w", newline="") as f:
            writer = csv.writer(f)
            for row in rows:
                writer.writerow(row)

        # # Keep JSON for reference if needed (position only)
        # pos = self.filter_item.scenePos()
        # mapping = {
        #     "filter": self.filter_img,
        #     "position": {"x": pos.x(), "y": pos.y()},
        #     "num_points_in_filter": len(rows)
        # }
        # with open("mapping.json", "w") as f:
        #     json.dump(mapping, f, indent=2)
        # print(f"Saved {self.annotation_file} with {len(rows)} rows and mapping.json")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    # tool = FilterTool("pexels-pixabay-415829.jpg", "filters/Squid-Game-Front-Man-Mask.png",annotation_file="filters/Squid-Game-Front-Man-Mask_annotation.csv")
    tool = FilterTool("filters/face.jpg", "filters/Squid-Game-Front-Man-Mask.png",annotation_file="filters/Squid-Game-Front-Man-Mask_annotation.csv")
    tool.resize(800, 600)
    tool.show()
    sys.exit(app.exec())