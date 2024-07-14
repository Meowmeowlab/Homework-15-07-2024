import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft2, fftshift


img = cv2.imread('./testimg/meow_400p.jpg', cv2.IMREAD_GRAYSCALE)

if img is None:
    raise ValueError(
        "Unable to read the image. Please verify if the path is accurate.")

# FFT - Fourier
dft = fft2(img)
dft_shift = fftshift(dft)

# Calc frequency spectrum
magnitude_spectrum = 20 * np.log(np.abs(dft_shift))

# Display
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.imshow(img, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(magnitude_spectrum, cmap='gray')
plt.title('Frequency Spectrum')
plt.axis('off')

plt.show()
