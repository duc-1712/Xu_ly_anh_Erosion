# utils.py - Bộ công cụ xử lý ảnh chuyên nghiệp - Nhóm 11
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image  # Thêm để hỗ trợ .webp

# Tùy chọn: thêm màu cho console (rất đẹp khi demo)
try:
    from colorama import init, Fore, Style
    init(autoreset=True)
    COLOR_SUCCESS = Fore.GREEN
    COLOR_ERROR = Fore.RED
    COLOR_INFO = Fore.CYAN
    COLOR_RESET = Style.RESET_ALL
except ImportError:
    COLOR_SUCCESS = COLOR_ERROR = COLOR_INFO = COLOR_RESET = ""


def read_image(path, grayscale=False):
    """
    Đọc ảnh từ đường dẫn, hỗ trợ mọi định dạng phổ biến (.png, .jpg, .webp, .bmp, .tiff...)
    
    Args:
        path (str): Đường dẫn ảnh
        grayscale (bool): Đọc ảnh xám hay màu
    
    Returns:
        numpy.ndarray hoặc None nếu lỗi
    """
    if not os.path.exists(path):
        print(f"{COLOR_ERROR}Không tìm thấy file: {path}{COLOR_RESET}")
        return None

    # Hỗ trợ .webp bằng Pillow (rất quan trọng!)
    if path.lower().endswith('.webp'):
        try:
            pil_img = Image.open(path)
            if grayscale:
                pil_img = pil_img.convert('L')
            else:
                pil_img = pil_img.convert('RGB')
            img = np.array(pil_img)
            if grayscale:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if len(img.shape) == 2 else img
            print(f"{COLOR_SUCCESS}Đọc ảnh WebP thành công: {os.path.basename(path)}{COLOR_RESET}")
            return img.astype(np.uint8) if grayscale else img
        except Exception as e:
            print(f"{COLOR_ERROR}Lỗi đọc file WebP: {e}{COLOR_RESET}")
            return None

    # Các định dạng khác dùng OpenCV
    flag = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
    img = cv2.imread(path, flag)
    
    if img is None:
        print(f"{COLOR_ERROR}Không thể đọc ảnh: {path}{COLOR_RESET}")
        return None
    
    shape_info = f"Kích thước: {img.shape}" if len(img.shape) == 3 else f"Kích thước: {img.shape} (ảnh xám)"
    print(f"{COLOR_SUCCESS}Đọc ảnh thành công: {os.path.basename(path)} | {shape_info} | dtype: {img.dtype}{COLOR_RESET}")
    return img


def show_image(title="Ảnh", img=None, cmap='gray'):
    """
    Hiển thị ảnh đẹp bằng matplotlib
    """
    if img is None:
        print(f"{COLOR_ERROR}Không thể hiển thị: {title} - ảnh bị None!{COLOR_RESET}")
        return
    
    plt.figure(figsize=(10, 8))
    if len(img.shape) == 3 and img.shape[2] == 3:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.imshow(img_rgb)
    else:
        plt.imshow(img, cmap=cmap)
    
    plt.title(title, fontsize=16, fontweight='bold', color='darkblue')
    plt.axis("off")
    plt.tight_layout()
    plt.show()


def save_image(path, img):
    """
    Lưu ảnh với thông báo đẹp và tạo thư mục tự động
    """
    if img is None:
        print(f"{COLOR_ERROR}Không thể lưu ảnh vì img = None → {path}{COLOR_RESET}")
        return False
    
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    # Đảm bảo ảnh là uint8
    if img.dtype != np.uint8:
        img = img.astype(np.uint8)
    
    success = cv2.imwrite(path, img)
    if success:
        print(f"{COLOR_SUCCESS}Đã lưu ảnh: {path}{COLOR_RESET}")
    else:
        print(f"{COLOR_ERROR}Lưu ảnh thất bại: {path}{COLOR_RESET}")
    return success