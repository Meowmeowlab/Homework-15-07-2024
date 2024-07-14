import cv2
import numpy as np
from scipy import fftpack
import matplotlib.pyplot as plt


def butterworth_lowpass_filter(d0, n, shape):
    rows, cols = shape
    u = np.arange(rows)
    v = np.arange(cols)
    u = u - rows / 2
    v = v - cols / 2
    u, v = np.meshgrid(u, v, sparse=False, indexing='ij')
    d = np.sqrt(u**2 + v**2)
    h = 1 / (1 + (d / d0)**(2 * n))
    return h


def butterworth_highpass_filter(d0, n, shape):
    lowpass = butterworth_lowpass_filter(d0, n, shape)
    return 1 - lowpass


def apply_filter(channel, filter_func, d0, n):
    channel_float = channel.astype(np.float32)
    dft = cv2.dft(channel_float, flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)

    filter_mask = filter_func(d0, n, channel.shape)
    filter_mask = np.repeat(filter_mask[:, :, np.newaxis], 2, axis=2)

    filtered_dft_shift = dft_shift * filter_mask
    filtered_dft = np.fft.ifftshift(filtered_dft_shift)
    filtered_img = cv2.idft(filtered_dft)
    filtered_img = cv2.magnitude(filtered_img[:, :, 0], filtered_img[:, :, 1])

    cv2.normalize(filtered_img, filtered_img, 0, 1, cv2.NORM_MINMAX)
    return filtered_img


def process_image(img_path, d0, n):
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    channels = cv2.split(img)
    lowpass_channels = [apply_filter(
        channel, butterworth_lowpass_filter, d0, n) for channel in channels]
    highpass_channels = [apply_filter(
        channel, butterworth_highpass_filter, d0, n) for channel in channels]

    lowpass_img = cv2.merge(lowpass_channels)
    highpass_img = cv2.merge(highpass_channels)

    return img, lowpass_img, highpass_img
