import cv2
import numpy as np
from scipy.ndimage import median_filter


def unsharp(image, sigma, strength):

    # Median filtering
    image_mf = median_filter(image, sigma)

    # Calculate the Laplacian
    lap = cv2.Laplacian(image_mf, cv2.CV_64F)

    # Calculate the sharpened image
    sharp = image-strength*lap

    # Saturate the pixels in either direction
    sharp[sharp > 255] = 255
    sharp[sharp < 0] = 0

    return sharp


# original_image = cv2.imread('./testimg/meow_400p.jpg')
# original_image = cv2.imread('./testimg/city_500p.jpg')
# blurry_image = cv2.GaussianBlur(original_image, (5, 5), 0)


def sharpen_filter(img, sigma=3, strength=3):
    sharp1 = np.zeros_like(img)
    for i in range(3):
        sharp1[:, :, i] = unsharp(img[:, :, i], sigma, strength)
        pass
    return sharp1


# cv2.imshow('Original', original_image)
# cv2.imshow("Blurry", blurry_image)
# cv2.imshow('Sharp1', sharpen_filter(original_image, 3, 3))
# cv2.waitKey(0)
