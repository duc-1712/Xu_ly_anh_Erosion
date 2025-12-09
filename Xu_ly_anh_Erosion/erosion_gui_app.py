import sys
import os
import cv2
import numpy as np
from PyQt5.QtWidgets import *
from PyQt5.QtGui import QPixmap, QImage, QPainter, QColor
from PyQt5.QtCore import Qt

# === HÀM ĐỌC ẢNH HỖ TRỢ TIẾNG VIỆT ===
def read_image_unicode(path, grayscale=True):
    try:
        # Đọc file bằng numpy từ đường dẫn unicode
        stream = np.fromfile(path, dtype=np.uint8)
        # Giải mã thành ảnh OpenCV
        flags = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
        img = cv2.imdecode(stream, flags)
        return img
    except Exception as e:
        print(f"Lỗi đọc ảnh: {e}")
        return None

def save_image_unicode(path, img):
    try:
        # Mã hóa ảnh thành định dạng file (ví dụ .jpg, .png) trong bộ nhớ
        ext = os.path.splitext(path)[1]
        result, img_data = cv2.imencode(ext, img)
        if result:
            with open(path, "wb") as f:
                img_data.tofile(f)
    except Exception as e:
        print(f"Lỗi lưu ảnh: {e}")

try:
    from erosion import binary_erosion, grayscale_erosion
    read_image = read_image_unicode
    save_image = save_image_unicode
except ImportError:
    read_image = read_image_unicode
    save_image = save_image_unicode
    
    def binary_erosion(img, kernel, iterations):
        return cv2.erode(img, kernel, iterations=iterations)
    def grayscale_erosion(img, kernel, iterations):
        return cv2.erode(img, kernel, iterations=iterations)

# === CLASS 1: KHUNG HIỂN THỊ ẢNH ===
class ImageViewer(QGraphicsView):
    def __init__(self, title_placeholder):
        super().__init__()
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)
        self.pixmap_item = None
        
        self.setRenderHint(QPainter.Antialiasing)
        self.setDragMode(QGraphicsView.ScrollHandDrag) 
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setBackgroundBrush(QColor("#ecf0f1"))
        
        self.placeholder_text = self.scene.addText(title_placeholder)
        self.placeholder_text.setScale(1.5)
        
    def set_image(self, cv_img):
        self.scene.clear()
        self.pixmap_item = None
        
        if cv_img is None:
            return

        if len(cv_img.shape) == 2:
            h, w = cv_img.shape
            qimg = QImage(cv_img.data, w, h, w, QImage.Format_Grayscale8)
        else:
            h, w, ch = cv_img.shape
            bytes_per_line = ch * w
            qimg = QImage(cv_img.data, w, h, bytes_per_line, QImage.Format_RGB888)

        pixmap = QPixmap.fromImage(qimg)
        self.pixmap_item = self.scene.addPixmap(pixmap)
        self.fitInView(self.pixmap_item, Qt.KeepAspectRatio) 

    def wheelEvent(self, event):
        if self.pixmap_item is None:
            return
        zoom_in = 1.15
        zoom_out = 1 / zoom_in
        if event.angleDelta().y() > 0:
            self.scale(zoom_in, zoom_in)
        else:
            self.scale(zoom_out, zoom_out)

# === CLASS 2: GIAO DIỆN CHÍNH ===
class ErosionGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Erosion App - Nhóm 11 (Fixed Unicode Path)")
        self.resize(1500, 900)
        
        self.current_img = None
        self.processed_img = None
        self.results = []
        self.current_idx = -1
        self.mode = "binary" 

        self.init_ui()

    def init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        # 1. TIÊU ĐỀ
        title = QLabel("PHÉP ĂN MÒN (EROSION) & TRÍCH XUẤT BIÊN")
        title.setStyleSheet("font: bold 24px; color: white; padding: 15px; background: #2c3e50; border-radius: 5px;")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        # 2. KHU VỰC ĐIỀU KHIỂN
        control_layout = QHBoxLayout()
        
        # --- INPUT ---
        gb_input = QGroupBox("1. Chọn Ảnh Đầu Vào")
        gb_layout = QVBoxLayout()
        btn_binary = QPushButton("Ảnh Nhị Phân")
        btn_binary.clicked.connect(lambda: self.load_image("binary"))
        btn_grayscale = QPushButton("Ảnh Đa Mức Xám")
        btn_grayscale.clicked.connect(lambda: self.load_image("grayscale"))
        gb_layout.addWidget(btn_binary)
        gb_layout.addWidget(btn_grayscale)
        gb_input.setLayout(gb_layout)
        control_layout.addWidget(gb_input)

        # --- CONFIG ---
        gb_config = QGroupBox("2. Cấu Hình Kernel")
        config_layout = QGridLayout()
        
        config_layout.addWidget(QLabel("Kích thước:"), 0, 0)
        self.cb_size = QComboBox()
        self.cb_size.addItems(["3x3", "5x5", "7x7", "9x9", "11x11"])
        config_layout.addWidget(self.cb_size, 0, 1)

        config_layout.addWidget(QLabel("Hình dáng:"), 1, 0)
        self.cb_shape = QComboBox()
        self.cb_shape.addItems(["Vuông (Rect)", "Chữ thập (Cross)", "Tròn (Ellipse)"])
        config_layout.addWidget(self.cb_shape, 1, 1)

        config_layout.addWidget(QLabel("Số lần lặp:"), 2, 0)
        self.spin_iter = QSpinBox()
        self.spin_iter.setRange(1, 20)
        self.spin_iter.setValue(1)
        config_layout.addWidget(self.spin_iter, 2, 1)
        gb_config.setLayout(config_layout)
        control_layout.addWidget(gb_config)

        # --- ACTION ---
        gb_action = QGroupBox("3. Thực Thi")
        action_layout = QVBoxLayout()
        
        btn_run = QPushButton("CHẠY EROSION")
        btn_run.setStyleSheet("background: #e74c3c; color: white; font-weight: bold; font-size: 14px; padding: 8px;")
        btn_run.clicked.connect(self.run_erosion)
        action_layout.addWidget(btn_run)
        
        self.btn_boundary = QPushButton("TRÍCH XUẤT BIÊN")
        self.btn_boundary.setStyleSheet("background: #8e44ad; color: white; font-weight: bold; font-size: 14px; padding: 8px;")
        self.btn_boundary.setToolTip("Lấy Ảnh gốc - Ảnh Erosion để hiện biên vật thể")
        self.btn_boundary.clicked.connect(self.run_boundary_extraction)
        action_layout.addWidget(self.btn_boundary)
        
        gb_action.setLayout(action_layout)
        control_layout.addWidget(gb_action)
        
        layout.addLayout(control_layout)

        # 3. VIEWER
        display_layout = QHBoxLayout()
        
        v_orig = QVBoxLayout()
        v_orig.addWidget(QLabel("<b>ẢNH GỐC</b>", alignment=Qt.AlignCenter))
        self.view_orig = ImageViewer("Chưa có ảnh")
        v_orig.addWidget(self.view_orig)
        
        v_res = QVBoxLayout()
        res_header = QHBoxLayout()
        res_header.addStretch()
        self.lbl_res_title = QLabel("<b>KẾT QUẢ</b>")
        res_header.addWidget(self.lbl_res_title)
        res_header.addStretch()
        
        self.chk_overlay = QCheckBox("Chế độ so sánh (Overlay)")
        self.chk_overlay.setToolTip("Hiển thị ảnh kết quả đè lên ảnh gốc")
        self.chk_overlay.setStyleSheet("font-weight: bold; color: #d35400;")
        self.chk_overlay.stateChanged.connect(self.update_overlay_view)
        res_header.addWidget(self.chk_overlay)
        
        v_res.addLayout(res_header)
        self.view_result = ImageViewer("Kết quả sẽ hiện ở đây")
        v_res.addWidget(self.view_result)

        display_layout.addLayout(v_orig)
        display_layout.addLayout(v_res)
        layout.addLayout(display_layout, stretch=1)

        # 4. FOOTER
        footer = QHBoxLayout()
        
        self.btn_save = QPushButton("Lưu ảnh kết quả")
        self.btn_save.setIcon(self.style().standardIcon(QStyle.SP_DialogSaveButton))
        self.btn_save.clicked.connect(self.save_current_result)
        self.btn_save.setEnabled(False)
        
        self.btn_prev = QPushButton("<< Trước")
        self.btn_prev.clicked.connect(self.show_prev)
        self.btn_next = QPushButton("Sau >>")
        self.btn_next.clicked.connect(self.show_next)
        
        self.lbl_status = QLabel("Sẵn sàng")
        self.lbl_status.setStyleSheet("color: gray; font-style: italic; margin-left: 10px;")

        footer.addWidget(self.btn_save)
        footer.addStretch()
        footer.addWidget(self.btn_prev)
        footer.addWidget(self.lbl_status)
        footer.addWidget(self.btn_next)
        
        layout.addLayout(footer)

        self.btn_prev.setEnabled(False)
        self.btn_next.setEnabled(False)
        self.btn_boundary.setEnabled(False)

    def load_image(self, mode):
        self.mode = mode
        # Dùng QFileDialog để lấy đường dẫn
        path, _ = QFileDialog.getOpenFileName(self, f"Chọn ảnh {mode}", "", "Images (*.png *.jpg *.jpeg *.bmp)")
        
        if path:
            # path ở đây có thể chứa tiếng Việt
            # read_image đã được thay bằng read_image_unicode
            img = read_image(path, grayscale=True)
            
            if img is not None:
                self.current_img = img
                self.view_orig.set_image(img)
                self.setWindowTitle(f"Erosion App | {os.path.basename(path)}")
                
                self.results = []
                self.current_idx = -1
                self.processed_img = None
                self.view_result.set_image(None)
                self.btn_boundary.setEnabled(True)
                self.chk_overlay.setChecked(False)
                self.lbl_status.setText(f"Đã tải ảnh: {path}") # Hiển thị full path
                self.btn_save.setEnabled(False)
            else:
                QMessageBox.critical(self, "Lỗi", "Không thể đọc ảnh. Vui lòng kiểm tra lại file!")

    def get_kernel(self):
        size = int(self.cb_size.currentText().split('x')[0])
        shape_text = self.cb_shape.currentText()
        shapes = {"Vuông (Rect)": cv2.MORPH_RECT, 
                  "Chữ thập (Cross)": cv2.MORPH_CROSS, 
                  "Tròn (Ellipse)": cv2.MORPH_ELLIPSE}
        return cv2.getStructuringElement(shapes[shape_text], (size, size)), size, shape_text

    def run_erosion(self):
        if self.current_img is None:
            QMessageBox.warning(self, "Chưa có ảnh", "Vui lòng chọn ảnh đầu vào trước!")
            return

        kernel, size, shape_name = self.get_kernel()
        iterations = self.spin_iter.value()
        
        if self.mode == "binary":
            res = binary_erosion(self.current_img, kernel, iterations)
        else:
            res = grayscale_erosion(self.current_img, kernel, iterations)
            
        title = f"Erosion ({shape_name.split()[0]} {size}x{size}, iter={iterations})"
        self.add_result(title, res)

    def run_boundary_extraction(self):
        if self.current_img is None:
            return
            
        kernel, size, shape_name = self.get_kernel()
        eroded = cv2.erode(self.current_img, kernel, iterations=1)
        boundary = cv2.subtract(self.current_img, eroded)
        
        title = f"Boundary (Biên) - Kernel {size}x{size}"
        self.add_result(title, boundary)

    def add_result(self, title, img):
        self.results.append((title, img))
        self.current_idx = len(self.results) - 1
        self.processed_img = img 
        
        self.btn_prev.setEnabled(self.current_idx > 0)
        self.btn_next.setEnabled(False)
        self.btn_save.setEnabled(True)
        
        self.update_overlay_view()

    def update_overlay_view(self):
        if self.processed_img is None:
            return
            
        title, img_res = self.results[self.current_idx]
        
        if self.chk_overlay.isChecked():
            img_orig_color = cv2.cvtColor(self.current_img, cv2.COLOR_GRAY2RGB)
            img_res_color = cv2.cvtColor(img_res, cv2.COLOR_GRAY2RGB)
            
            img_res_tint = img_res_color.copy()
            img_res_tint[:, :, 1] = 0 
            img_res_tint[:, :, 2] = 0 
            
            blended = cv2.addWeighted(img_orig_color, 0.6, img_res_tint, 0.4, 0)
            
            self.view_result.set_image(blended)
            self.lbl_status.setText(f"Đang hiển thị: {title} (Chế độ Overlay)")
        else:
            self.view_result.set_image(img_res)
            self.lbl_status.setText(f"Kết quả {self.current_idx + 1}/{len(self.results)}: {title}")

    def show_prev(self):
        if self.current_idx > 0:
            self.current_idx -= 1
            self.processed_img = self.results[self.current_idx][1]
            self.update_overlay_view()
            self.btn_next.setEnabled(True)
            if self.current_idx == 0: self.btn_prev.setEnabled(False)

    def show_next(self):
        if self.current_idx < len(self.results) - 1:
            self.current_idx += 1
            self.processed_img = self.results[self.current_idx][1]
            self.update_overlay_view()
            self.btn_prev.setEnabled(True)
            if self.current_idx == len(self.results) - 1: self.btn_next.setEnabled(False)

    def save_current_result(self):
        if self.processed_img is None: return
        path, _ = QFileDialog.getSaveFileName(self, "Lưu ảnh", "", "PNG (*.png);;JPG (*.jpg)")
        if path:
            # Sử dụng save_image_unicode để lưu file có dấu
            save_image(path, self.processed_img) 
            QMessageBox.information(self, "Thành công", f"Đã lưu ảnh tại:\n{path}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion") 
    
    font = app.font()
    font.setPointSize(10)
    app.setFont(font)
    
    window = ErosionGUI()
    window.show()
    sys.exit(app.exec_())