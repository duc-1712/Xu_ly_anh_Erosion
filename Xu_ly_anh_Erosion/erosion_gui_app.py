# erosion_gui_app.py - Ứng dụng Erosion GUI đẹp lung linh - Nhóm 11
import sys
import os
import cv2
import numpy as np
from PyQt5.QtWidgets import *
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt
from utils import read_image, save_image
from erosion import binary_erosion, grayscale_erosion


class ErosionGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Erosion - Xử Lý Ảnh Số - Nhóm 11 - Đề tài 09")
        self.resize(1400, 900)
        self.current_img = None
        self.results = []           # Lưu danh sách (tên, ảnh kết quả)
        self.current_idx = -1
        self.mode = "binary"        # hoặc "grayscale"
        self.init_ui()

    def init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        # Tiêu đề
        title = QLabel("PHÉP ĂN MÒN (EROSION) TRONG XỬ LÝ ẢNH")
        title.setStyleSheet("font: bold 20px; color: #2c3e50; padding: 15px; background: #ecf0f1;")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        # Nút chọn ảnh
        btn_layout = QHBoxLayout()
        btn_binary = QPushButton("Chọn ảnh NHỊ PHÂN (có nhiễu)")
        btn_binary.clicked.connect(lambda: self.load_image("binary"))
        btn_grayscale = QPushButton("Chọn ảnh ĐA MỨC XÁM")
        btn_grayscale.clicked.connect(lambda: self.load_image("grayscale"))

        btn_layout.addWidget(btn_binary)
        btn_layout.addWidget(btn_grayscale)
        layout.addLayout(btn_layout)

        # Cấu hình kernel
        config = QHBoxLayout()
        config.addWidget(QLabel("Kernel:"))
        self.cb_size = QComboBox()
        self.cb_size.addItems(["3x3", "5x5", "7x7"])
        config.addWidget(self.cb_size)

        self.cb_shape = QComboBox()
        self.cb_shape.addItems(["Vuông", "Chữ thập", "Tròn"])
        config.addWidget(self.cb_shape)

        config.addWidget(QLabel("Lặp:"))
        self.spin_iter = QSpinBox()
        self.spin_iter.setRange(1, 10)
        self.spin_iter.setValue(1)
        config.addWidget(self.spin_iter)

        btn_run = QPushButton("THỰC HIỆN EROSION")
        btn_run.setStyleSheet("background: #e74c3c; color: white; font-weight: bold; padding: 10px;")
        btn_run.clicked.connect(self.run_erosion)
        config.addWidget(btn_run)
        layout.addLayout(config)

        # Hiển thị ảnh
        display = QHBoxLayout()
        self.label_orig = QLabel("Ảnh gốc sẽ hiện ở đây")
        self.label_orig.setAlignment(Qt.AlignCenter)
        self.label_orig.setStyleSheet("border: 2px dashed #95a5a6; background: #f8f9fa; min-height: 500px; font-size: 18px;")
        
        self.label_result = QLabel("Kết quả sẽ hiện ở đây")
        self.label_result.setAlignment(Qt.AlignCenter)
        self.label_result.setStyleSheet("border: 2px dashed #3498db; background: #f8f9fa; min-height: 500px; font-size: 18px;")

        display.addWidget(self.label_orig)
        display.addWidget(self.label_result)
        layout.addLayout(display)

        # Nút điều hướng kết quả
        nav = QHBoxLayout()
        self.btn_prev = QPushButton("Previous")
        self.btn_prev.clicked.connect(self.show_prev)
        self.btn_next = QPushButton("Next")
        self.btn_next.clicked.connect(self.show_next)
        self.label_info = QLabel("Chưa có kết quả")
        self.label_info.setAlignment(Qt.AlignCenter)

        nav.addWidget(self.btn_prev)
        nav.addWidget(self.label_info)
        nav.addWidget(self.btn_next)
        layout.addLayout(nav)

        # Tắt nút điều hướng lúc đầu
        self.btn_prev.setEnabled(False)
        self.btn_next.setEnabled(False)

    def load_image(self, mode):
        self.mode = mode
        path, _ = QFileDialog.getOpenFileName(self, f"Chọn ảnh {mode}", "", "Images (*.png *.jpg *.jpeg *.bmp *.webp)")
        if path:
            img = read_image(path, grayscale=True)
            if img is not None:
                self.current_img = img
                self.display_image(img, self.label_orig)
                self.setWindowTitle(f"Erosion - Nhóm 11 | {os.path.basename(path)}")
                self.results = []
                self.current_idx = -1
                self.label_result.clear()
                self.label_result.setText("Kết quả sẽ hiện ở đây")
                self.label_info.setText("Chưa có kết quả")
                self.btn_prev.setEnabled(False)
                self.btn_next.setEnabled(False)

    def display_image(self, cv_img, label):
        if cv_img is None:
            return
        h, w = cv_img.shape
        qimg = QImage(cv_img.data, w, h, w, QImage.Format_Grayscale8)
        pixmap = QPixmap.fromImage(qimg).scaled(650, 650, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        label.setPixmap(pixmap)

    def run_erosion(self):
        if self.current_img is None:
            QMessageBox.warning(self, "Lỗi", "Chưa chọn ảnh!")
            return

        self.results = []
        sizes = [3, 5]
        shapes = {"Vuông": cv2.MORPH_RECT, "Chữ thập": cv2.MORPH_CROSS, "Tròn": cv2.MORPH_ELLIPSE}
        iterations = self.spin_iter.value()

        size_selected = int(self.cb_size.currentText()[0])
        shape_selected = self.cb_shape.currentText()

        # Chỉ chạy 1 kernel được chọn (để dễ xem)
        kernel = cv2.getStructuringElement(shapes[shape_selected], (size_selected, size_selected))
        kernel_name = f"{size_selected}x{size_selected}_{shape_selected.lower()}"

        if self.mode == "binary":
            result = binary_erosion(self.current_img, kernel=kernel, iterations=iterations)
        else:
            result = grayscale_erosion(self.current_img, kernel=kernel, iterations=iterations)

        title = f"{self.mode.upper()} - Erosion {kernel_name} (lặp {iterations})"
        self.results.append((title, result))

        # Lưu ảnh
        folder = f"results/{self.mode}"
        os.makedirs(folder, exist_ok=True)
        save_path = f"{folder}/erosion_{kernel_name}_iter{iterations}.png"
        save_image(save_path, result)

        # Hiển thị kết quả đầu tiên
        self.current_idx = 0
        self.display_image(result, self.label_result)
        self.label_info.setText(f"Kết quả 1/1: {title}")
        self.btn_prev.setEnabled(False)
        self.btn_next.setEnabled(False)
        QMessageBox.information(self, "Thành công!", f"Đã lưu:\n{save_path}")

    def show_prev(self):
        if self.current_idx > 0:
            self.current_idx -= 1
            title, img = self.results[self.current_idx]
            self.display_image(img, self.label_result)
            self.label_info.setText(f"Kết quả {self.current_idx + 1}/{len(self.results)}: {title}")

    def show_next(self):
        if self.current_idx < len(self.results) - 1:
            self.current_idx += 1
            title, img = self.results[self.current_idx]
            self.display_image(img, self.label_result)
            self.label_info.setText(f"Kết quả {self.current_idx + 1}/{len(self.results)}: {title}")


# === CHẠY CHƯƠNG TRÌNH ===
if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = ErosionGUI()
    window.show()
    sys.exit(app.exec_())