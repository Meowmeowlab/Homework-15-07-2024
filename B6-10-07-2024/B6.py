import numpy as np
import cv2 as cv
import Extra_Module.bwfilter as bwf
import matplotlib.pyplot as plt


def display_images(original_img, lowpass_img, highpass_img):
    fig, axes = plt.subplots(1, 3, figsize=(12, 7), sharex=True, sharey=True)

    axes[0].imshow(original_img)
    axes[0].set_title('Original Image')
    axes[0].axis('off')

    axes[1].imshow(lowpass_img)
    axes[1].set_title('Butterworth Lowpass Filtered Image')
    axes[1].axis('off')

    axes[2].imshow(highpass_img)
    axes[2].set_title('Butterworth Highpass Filtered Image')
    axes[2].axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    img_path = './testimg/meow_400p.jpg'
    d0 = 30  # Cutoff frequency
    n = 2    # Order of the filter

    original_img, lowpass_img, highpass_img = bwf.process_image(
        img_path, d0, n)
    display_images(original_img, lowpass_img, highpass_img)
