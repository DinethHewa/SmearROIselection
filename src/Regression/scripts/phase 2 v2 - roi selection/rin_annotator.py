# NOTE: This GUI requires a working X/Wayland display; use rin_annotator_web.py on headless/WSL.
import argparse
import os
import sys

import cv2
import numpy as np
from PyQt6.QtCore import Qt, QRect, QLibraryInfo
from PyQt6.QtGui import QImage, QPainter, QPen, QColor
from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QLabel,
    QFileDialog,
    QMessageBox,
    QVBoxLayout,
)

# =========================
# Configuration (DO NOT CHANGE mid-project)
# =========================
GRID_ROWS = 10
GRID_COLS = 10
TILE_SIZE = 200  # pixels
CANVAS_SIZE = GRID_ROWS * TILE_SIZE  # 2000 x 2000
LABEL_DIR = "labels"


def ensure_qt_plugin_path():
    if os.environ.get("QT_QPA_PLATFORM_PLUGIN_PATH"):
        return
    try:
        plugins_path = QLibraryInfo.path(QLibraryInfo.LibraryPath.PluginsPath)
    except Exception:
        return
    if plugins_path:
        os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = plugins_path


def parse_args():
    ap = argparse.ArgumentParser(
        description="RIN patch annotator (10x10 grid, 1/0 keyboard labels)."
    )
    ap.add_argument(
        "--image-dir",
        default=None,
        help="Directory of images to annotate (skips file dialog).",
    )
    ap.add_argument(
        "--labels-dir",
        default=None,
        help="Directory to save label .npy files (default: ./labels).",
    )
    return ap.parse_args()


def normalize_dir(path):
    if path is None:
        return None
    return os.path.abspath(os.path.expanduser(path))


# =========================
# Image Display Widget
# =========================
class ImageCanvas(QLabel):
    def __init__(self):
        super().__init__()
        self.setFixedSize(CANVAS_SIZE, CANVAS_SIZE)
        self.image = None
        self.labels = None
        self.active_r = 0
        self.active_c = 0

    def set_image(self, img, labels):
        self.image = img
        self.labels = labels
        self.active_r = 0
        self.active_c = 0
        self.update()

    def next_cell(self):
        self.active_c += 1
        if self.active_c >= GRID_COLS:
            self.active_c = 0
            self.active_r += 1
        self.update()

    def is_done(self):
        return self.active_r >= GRID_ROWS

    def paintEvent(self, event):
        if self.image is None:
            return

        painter = QPainter(self)
        painter.drawImage(0, 0, self.image)

        pen_grid = QPen(QColor(200, 200, 200))
        pen_grid.setWidth(1)
        painter.setPen(pen_grid)

        # Draw grid
        for r in range(GRID_ROWS + 1):
            painter.drawLine(
                0,
                r * TILE_SIZE,
                CANVAS_SIZE,
                r * TILE_SIZE,
            )
        for c in range(GRID_COLS + 1):
            painter.drawLine(
                c * TILE_SIZE,
                0,
                c * TILE_SIZE,
                CANVAS_SIZE,
            )

        # Draw labels
        for r in range(GRID_ROWS):
            for c in range(GRID_COLS):
                if self.labels[r, c] == 1:
                    painter.fillRect(
                        QRect(
                            c * TILE_SIZE,
                            r * TILE_SIZE,
                            TILE_SIZE,
                            TILE_SIZE,
                        ),
                        QColor(0, 255, 0, 60),
                    )

        # Highlight active cell
        if not self.is_done():
            pen_active = QPen(QColor(255, 0, 0))
            pen_active.setWidth(3)
            painter.setPen(pen_active)
            painter.drawRect(
                QRect(
                    self.active_c * TILE_SIZE,
                    self.active_r * TILE_SIZE,
                    TILE_SIZE,
                    TILE_SIZE,
                )
            )


# =========================
# Main Window
# =========================
class Annotator(QMainWindow):
    def __init__(self, image_dir, label_dir=LABEL_DIR):
        super().__init__()
        self.setWindowTitle("RIN Patch Annotation Tool (10x10)")

        self.label_dir = label_dir
        os.makedirs(self.label_dir, exist_ok=True)

        self.image_paths = sorted(
            [
                os.path.join(image_dir, name)
                for name in os.listdir(image_dir)
                if name.lower().endswith((".png", ".jpg", ".tif", ".tiff"))
            ]
        )

        if not self.image_paths:
            QMessageBox.critical(self, "Error", "No images found.")
            sys.exit(1)

        self.idx = 0
        self.labels = np.zeros((GRID_ROWS, GRID_COLS), dtype=np.uint8)

        self.canvas = ImageCanvas()
        layout = QVBoxLayout()
        layout.addWidget(self.canvas)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        self.load_image()

    def load_image(self):
        path = self.image_paths[self.idx]
        img = cv2.imread(path, cv2.IMREAD_COLOR)

        if img is None:
            QMessageBox.critical(self, "Error", f"Failed to load {path}")
            return

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (CANVAS_SIZE, CANVAS_SIZE))

        h, w, _ = img.shape
        qimg = QImage(img.data, w, h, 3 * w, QImage.Format.Format_RGB888)

        self.labels[:] = 0
        self.canvas.set_image(qimg, self.labels)
        self.setWindowTitle(f"Annotating: {os.path.basename(path)}")

    def save_and_next(self):
        img_name = os.path.splitext(os.path.basename(self.image_paths[self.idx]))[0]
        np.save(os.path.join(self.label_dir, img_name + ".npy"), self.labels)

        self.idx += 1
        if self.idx >= len(self.image_paths):
            QMessageBox.information(self, "Done", "All images annotated.")
            QApplication.quit()
        else:
            self.load_image()

    def keyPressEvent(self, event):
        if self.canvas.is_done():
            self.save_and_next()
            return

        if event.key() == Qt.Key.Key_1:
            self.labels[self.canvas.active_r, self.canvas.active_c] = 1
            self.canvas.next_cell()
        elif event.key() == Qt.Key.Key_0:
            self.labels[self.canvas.active_r, self.canvas.active_c] = 0
            self.canvas.next_cell()
        elif event.key() == Qt.Key.Key_Escape:
            QApplication.quit()


# =========================
# Entry Point
# =========================
if __name__ == "__main__":
    args = parse_args()
    ensure_qt_plugin_path()
    app = QApplication(sys.argv)

    img_dir = normalize_dir(args.image_dir)
    if img_dir is None:
        img_dir = QFileDialog.getExistingDirectory(
            None,
            "Select Image Directory",
        )
    if img_dir and not os.path.isdir(img_dir):
        print(f"[ERROR] Image directory not found: {img_dir}")
        sys.exit(1)
    if not img_dir:
        sys.exit(0)

    label_dir = normalize_dir(args.labels_dir) if args.labels_dir else LABEL_DIR
    window = Annotator(img_dir, label_dir=label_dir)
    window.show()
    sys.exit(app.exec())
