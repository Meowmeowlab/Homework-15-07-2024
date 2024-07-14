import cv2
import numpy


def high_pass_filter(img, sigma=31):
    # img = cv2.imread('./testimg/meow_400p.jpg')
    blur = cv2.GaussianBlur(img, (sigma, sigma), 0)
    filtered = img - blur
    neg_frame = cv2.bitwise_not(filtered)
    filtered = filtered + 127*numpy.ones(neg_frame.shape, numpy.uint8)
    return filtered


def low_pass_filter(img, sigma=3, mode=0):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if (mode == 0):
        varresult = cv2.boxFilter(img, -1, (sigma, sigma))  # linear filter
    if (mode == 1):
        varresult = cv2.GaussianBlur(img, (sigma, sigma), 0)
    if (mode == 2):
        varresult = cv2.blur(img, (sigma, sigma))

    varresult = cv2.cvtColor(varresult, cv2.COLOR_BGR2RGB)
    return varresult


def laplacian_filter(img, ksize=5, scale=1):
    # lap = cv2.Laplacian(img, cv2.CV_64F)
    lap = cv2.Laplacian(img, cv2.CV_8U, None, ksize, scale)
    return lap
