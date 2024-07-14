import cv2 as cv
import numpy as np
import Extra_Module.filter as fl
import Extra_Module.idft as idft
import Extra_Module.dft as dft

img = cv.imread('./testimg/meow_400p.jpg')
blurImg = cv.GaussianBlur(img, (5, 5), 0)

global srcImg
srcImg = img
srctracker = 0

cv.namedWindow("Low Pass")
cv.namedWindow("ORIGINAL DFT/IFT ROUND TRIP")

lpvalue = 3
lpmode = 0
radius = 32
gblur = 19


def showlowpass():
    cv.imshow("Source Image", srcImg)
    cv.imshow("Low Pass", fl.low_pass_filter(srcImg, lpvalue, lpmode))


def showimgifilter():
    img_ifiltered = idft.idft_filter(srcImg)
    cv.imshow('IDFT Filtered', cv.cvtColor(
        img_ifiltered.astype(np.uint8), cv.COLOR_RGB2BGR))


def showimgfilter():
    img_filtered = dft.fft_filter(srcImg, radius, gblur)
    cv.imshow('ORIGINAL DFT/IFT ROUND TRIP',
              img_filtered[0])
    cv.imshow('FILTERED DFT/IFT ROUND TRIP',
              img_filtered[1])
    cv.imshow('FILTERED DFT/IFT ROUND TRIP Cleanup',
              img_filtered[2])


def on_low_pass_sigma_change_trackbar(pos):
    global lpvalue
    lpvalue = pos
    if (lpvalue == 0):
        lpvalue = 1
    if (lpvalue % 2 == 0):
        lpvalue = lpvalue + 1
    cv.setTrackbarPos('Sigma', 'Low Pass', lpvalue)
    showlowpass()


def on_low_pass_mode_change_trackbar(pos):
    global lpmode
    lpmode = pos
    # cv.setTrackbarPos('Mode', 'Low Pass', lpmode)
    showlowpass()


def on_radius_change_trackbar(pos):
    global radius
    radius = pos
    # cv.setTrackbarPos('Mode', 'Low Pass', lpmode)
    showimgfilter()


def on_gblur_change_trackbar(pos):
    global gblur
    gblur = pos
    if (gblur == 0):
        gblur = 1
    if (gblur % 2 == 0):
        gblur = gblur + 1
    cv.setTrackbarPos('Blur Size', 'ORIGINAL DFT/IFT ROUND TRIP', gblur)
    showimgfilter()


cv.createTrackbar('Sigma', 'Low Pass',
                  lpvalue, 200, on_low_pass_sigma_change_trackbar)
cv.createTrackbar('Mode', 'Low Pass',
                  lpmode, 2, on_low_pass_mode_change_trackbar)
cv.createTrackbar('Radius', 'ORIGINAL DFT/IFT ROUND TRIP',
                  radius, 100, on_radius_change_trackbar)
cv.createTrackbar('Blur Size', 'ORIGINAL DFT/IFT ROUND TRIP',
                  gblur, 100, on_gblur_change_trackbar)

cv.imshow("Original Image", img)
showimgifilter()
showimgfilter()
showlowpass()

cv.waitKey(0)
cv.destroyAllWindows()
