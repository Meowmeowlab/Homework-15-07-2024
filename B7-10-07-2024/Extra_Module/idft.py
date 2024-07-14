import numpy as np
import cv2
from matplotlib import pyplot as plt


def butterworth_lowpass_filter(d, shape, n):
    """Tạo bộ lọc Butterworth Lowpass."""
    P, Q = shape
    H = np.zeros((P, Q), dtype=np.float32)
    for u in range(P):
        for v in range(Q):
            D = np.sqrt((u - P/2)**2 + (v - Q/2)**2)
            H[u, v] = 1 / (1 + (D / d)**(2 * n))
    return H


def apply_fourier_smoothing(channel, d, n):
    """Áp dụng Biến đổi Fourier, lọc và Biến đổi ngược cho một kênh ảnh."""
    f = np.fft.fft2(channel)
    fshift = np.fft.fftshift(f)

    H = butterworth_lowpass_filter(d, fshift.shape, n)
    fshift_filtered = fshift * H

    f_ishift = np.fft.ifftshift(fshift_filtered)
    channel_filtered = np.fft.ifft2(f_ishift)
    return np.abs(channel_filtered)


def idft_filter(img):
    image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    r, g, b = cv2.split(image_rgb)

    # Định nghĩa tham số bộ lọc
    d = 30  # Ngưỡng cắt
    n = 2   # Bậc của bộ lọc

    # Áp dụng lọc Fourier cho từng kênh
    r_filtered = apply_fourier_smoothing(r, d, n)
    g_filtered = apply_fourier_smoothing(g, d, n)
    b_filtered = apply_fourier_smoothing(b, d, n)

    image_filtered = cv2.merge([r_filtered, g_filtered, b_filtered])

    # cv.imshow('Original', cv.cvtColor(
    #     image_rgb, cv.COLOR_RGB2BGR))  # image_rgb)
    # cv2.imshow('Filtered', cv2.cvtColor(
    #     image_filtered.astype(np.uint8), cv2.COLOR_RGB2BGR))
    return image_filtered
