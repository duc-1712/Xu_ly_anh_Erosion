# erosion.py - ĐÃ SỬA ĐỂ TƯƠNG THÍCH VỚI main.py
import cv2
import numpy as np

def binary_erosion(image, kernel=None, kernel_size=3, iterations=1):
    """
    Erosion cho ảnh nhị phân - hỗ trợ kernel tùy chỉnh
    """
    if kernel is None:
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
    
    eroded = cv2.erode(image, kernel, iterations=iterations)
    return eroded


def grayscale_erosion(image, kernel=None, kernel_size=3, iterations=1):
    """
    Erosion cho ảnh xám đa mức - thủ công bằng min
    """
    if kernel is None:
        ksize = kernel_size
        if ksize % 2 == 0:
            ksize += 1  # phải lẻ
        kernel = np.ones((ksize, ksize), dtype=np.uint8)

    h, w = image.shape
    pad = kernel.shape[0] // 2
    padded = cv2.copyMakeBorder(image, pad, pad, pad, pad, cv2.BORDER_REFLECT)
    output = np.zeros_like(image)

    for i in range(h):
        for j in range(w):
            region = padded[i:i+kernel.shape[0], j:j+kernel.shape[1]]
            # Chỉ lấy vùng có kernel = 1
            output[i, j] = np.min(region[kernel == 1]) if np.any(kernel == 1) else image[i, j]

    # Áp dụng iterations
    result = output
    for _ in range(iterations - 1):
        result = grayscale_erosion(result, kernel=kernel)  # gọi đệ quy đơn giản

    return result