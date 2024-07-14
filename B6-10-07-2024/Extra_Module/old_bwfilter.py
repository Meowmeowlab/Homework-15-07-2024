import numpy as np
import cv2 as cv


def butterworth_lowpass_filter(shape, cutoff, order):
    P, Q = shape
    H = np.zeros((P, Q))
    center_x, center_y = P // 2, Q // 2

    # D = np.sqrt((np.arange(P) - center_x)
    #             [:, np.newaxis]**2 + (np.arange(Q) - center_y)**2)
    # H = 1 / (1 + (D / cutoff)**(2 * order))

    for u in range(P):
        for v in range(Q):
            D = np.sqrt((u - center_x)**2 + (v - center_y)**2)
            H[u, v] = 1 / (1 + (D / cutoff)**(2 * order))

    return H


def butterworth_highpass_filter(shape, cutoff, order):
    P, Q = shape
    H = np.zeros((P, Q))
    center_x, center_y = P // 2, Q // 2

    # D = np.sqrt((np.arange(P) - center_x)
    #             [:, np.newaxis]**2 + (np.arange(Q) - center_y)**2)
    # H = np.where(D == 0, 0, 1 / (1 + (cutoff / D)**(2 * order)))

    for u in range(P):
        for v in range(Q):
            D = np.sqrt((u - center_x)**2 + (v - center_y)**2)
            if D == 0:
                H[u, v] = 0
            else:
                H[u, v] = 1 / (1 + (cutoff / D)**(2 * order))

    return H


def bw_highpass_filter(img):
    # Perform the Discrete Fourier Transform (DFT)
    dft = cv.dft(np.float32(img), flags=cv.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)

    # Create a Butterworth high-pass filter
    cutoff = 30  # Adjust the cutoff frequency as needed
    order = 2    # Adjust the order of the filter as needed
    H = butterworth_highpass_filter(img.shape, cutoff, order)

    # Apply the filter on the DFT
    filtered_dft_shift = dft_shift * H[:, :, np.newaxis]

    # Shift back and inverse DFT
    filtered_dft = np.fft.ifftshift(filtered_dft_shift)
    filtered_image = cv.idft(filtered_dft)
    filtered_image = cv.magnitude(
        filtered_image[:, :, 0], filtered_image[:, :, 1])

    return filtered_image


def bw_lowpass_filter(img):
    # Perform the Discrete Fourier Transform (DFT)
    dft = cv.dft(np.float32(img), flags=cv.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)

    # Create a Butterworth low-pass filter
    cutoff = 30  # Adjust the cutoff frequency as needed
    order = 2    # Adjust the order of the filter as needed
    H = butterworth_lowpass_filter(img.shape, cutoff, order)

    # Apply the filter on the DFT
    filtered_dft_shift = dft_shift * H[:, :, np.newaxis]

    # Shift back and inverse DFT
    filtered_dft = np.fft.ifftshift(filtered_dft_shift)
    filtered_image = cv.idft(filtered_dft)
    filtered_image = cv.magnitude(
        filtered_image[:, :, 0], filtered_image[:, :, 1])

    return filtered_image
