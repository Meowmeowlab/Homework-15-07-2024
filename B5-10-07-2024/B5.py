import cv2 as cv
import numpy as np
import Extra_Module.filter as fl
import Extra_Module.unsharpen as us

img = cv.imread('./testimg/meow_400p.jpg')
blurImg = cv.GaussianBlur(img, (5, 5), 0)

global srcImg
srcImg = img
srctracker = 0

cv.namedWindow("High Pass")
cv.namedWindow("Laplacian")
cv.namedWindow("Sharpen")

hpvalue = 31
lapksizevalue = 3
sharpen_strength = 3
sharpen_sigma = 3


def showimage():
    cv.imshow("Source Image", srcImg)
    cv.imshow("High Pass", fl.high_pass_filter(srcImg, hpvalue))
    # cv.imshow("Laplacian",  cv.cvtColor(
    #     fl.laplacian_filter(srcImg, lapksizevalue), cv.COLOR_BGR2GRAY))
    cv.imshow("Laplacian", fl.laplacian_filter(
        cv.cvtColor(srcImg, cv.COLOR_BGR2GRAY), lapksizevalue))
    cv.imshow("Sharpen", us.sharpen_filter(
        srcImg, sharpen_sigma, sharpen_strength))  # 37-13


def on_high_pass_sigma_change_trackbar(pos):
    global hpvalue
    hpvalue = pos
    if (hpvalue == 0):
        hpvalue = 1
    if (hpvalue % 2 == 0):
        hpvalue = hpvalue + 1
    cv.setTrackbarPos('Sigma', 'High Pass', hpvalue)
    showimage()


def on_laplacian_ksize_pass_change_trackbar(pos):
    global lapksizevalue
    lapksizevalue = pos
    if (lapksizevalue == 0):
        lapksizevalue = 1
    if (lapksizevalue % 2 == 0):
        lapksizevalue = lapksizevalue + 1
    cv.setTrackbarPos('KSize', 'Laplacian', lapksizevalue)
    showimage()


def on_sharpen_strength_change_trackbar(pos):
    global sharpen_strength
    sharpen_strength = pos
    # cv.setTrackbarPos('Strength', 'Sharpen', sharpen_strength)
    showimage()


def on_sharpen_sigma_change_trackbar(pos):
    global sharpen_sigma
    sharpen_sigma = pos
    # cv.setTrackbarPos('Sigma', 'Sharpen', sharpen_sigma)
    showimage()


cv.createTrackbar('Sigma', 'High Pass',
                  hpvalue, 200, on_high_pass_sigma_change_trackbar)
cv.createTrackbar('KSize', 'Laplacian',
                  lapksizevalue, 31, on_laplacian_ksize_pass_change_trackbar)

cv.createTrackbar('Strength', "Sharpen",
                  sharpen_strength, 100, on_sharpen_strength_change_trackbar)
cv.createTrackbar('Sigma', "Sharpen",
                  sharpen_sigma, 100, on_sharpen_strength_change_trackbar)


def on_src_toggle_trackbar(pos):
    global srctracker, srcImg
    srctracker = pos
    if (srctracker == 0):
        srcImg = img
        showimage()
    else:
        srcImg = blurImg
        showimage()


cv.namedWindow('Source Image')
cv.createTrackbar('Source Toggle', 'Source Image',
                  srctracker, 1, on_src_toggle_trackbar)

cv.imshow("Original Image", img)
cv.imshow("Source Image", srcImg)


showimage()

cv.waitKey(0)
