import os
import cv2
from tkinter import filedialog, Tk
from utils import read_image, show_image, save_image
from erosion import binary_erosion, grayscale_erosion


def chon_anh(title="Chọn ảnh để thực hiện Erosion"):
    root = Tk()
    root.withdraw()
    root.update()
    
    file_path = filedialog.askopenfilename(
        title=title,
        filetypes=[
            ("Ảnh thông dụng", "*.png *.jpg *.jpeg *.bmp *.tiff *.webp"),
            ("Tất cả file", "*.*")
        ]
    )
    root.destroy()
    
    if not file_path:
        print("Bạn chưa chọn ảnh → thoát chương trình")
        return None
    print(f"Đã chọn: {os.path.basename(file_path)}")
    return file_path


def chay_erosion_voi_nhieu_kernel(img, ten_loai, kernel_sizes=[3, 5], iterations=1):
    """Chạy erosion với nhiều loại kernel và lưu kết quả"""
    if img is None:
        print("Ảnh bị None → bỏ qua!")
        return
    
    # Danh sách kernel
    kernel_types = {
        "vuong": cv2.MORPH_RECT,
        "chu_thap": cv2.MORPH_CROSS,
        "tron": cv2.MORPH_ELLIPSE
    }
    
    for size in kernel_sizes:
        for ten_k, k_type in kernel_types.items():
            kernel = cv2.getStructuringElement(k_type, (size, size))
            kernel_name = f"{size}x{size}_{ten_k}"
            
            # Dùng hàm phù hợp: binary hoặc grayscale
            if ten_loai == "binary":
                eroded = binary_erosion(img, kernel=kernel, iterations=iterations)
            else:  # grayscale
                eroded = grayscale_erosion(img, kernel=kernel, iterations=iterations)
            
            # Hiển thị
            show_image(f"{ten_loai.upper()} - Erosion {kernel_name}", eroded)
            
            # Lưu ảnh
            folder = f"results/{ten_loai}"
            os.makedirs(folder, exist_ok=True)
            save_path = f"{folder}/{ten_loai}_erosion_{kernel_name}.png"
            save_image(save_path, eroded)


def main():
    print("="*60)
    print("     CHƯƠNG TRÌNH THỰC HIỆN EROSION - NHÓM 11")
    print("="*60)
    
    # Tạo thư mục kết quả
    os.makedirs("results/binary", exist_ok=True)
    os.makedirs("results/grayscale", exist_ok=True)
    
    print("\nBước 1: Chọn ảnh NHỊ PHÂN (binary) để thử nghiệm")
    path1 = chon_anh("Chọn ảnh NHỊ PHÂN (đen trắng, có nhiễu)")
    if path1:
        img_binary = read_image(path1, grayscale=True)
        show_image("Ảnh nhị phân gốc", img_binary)
        chay_erosion_voi_nhieu_kernel(img_binary, "binary")
        print("Hoàn tất Erosion ảnh nhị phân!\n")
    
    print("Bước 2: Chọn ảnh XÁM ĐA MỨC (grayscale) để thử nghiệm")
    path2 = chon_anh("Chọn ảnh XÁM ĐA MỨC (có chi tiết, bóng đổ...)")
    if path2:
        img_gray = read_image(path2, grayscale=True)
        show_image("Ảnh đa mức xám gốc", img_gray)
        chay_erosion_voi_nhieu_kernel(img_gray, "grayscale")
        print("Hoàn tất Erosion ảnh đa mức xám!\n")
    
    print("="*60)
    print("HOÀN TẤT! Tất cả kết quả đã được lưu trong thư mục:")
    print("   → results/binary/")
    print("   → results/grayscale/")
    print("Mở thư mục 'results' để xem ảnh đẹp nhất cho slide bảo vệ!")
    print("="*60)
    
    # Giữ cửa sổ matplotlib mở đến khi nhấn phím bất kỳ
    input("\nNhấn Enter để thoát chương trình...")


if __name__ == "__main__":
    main()