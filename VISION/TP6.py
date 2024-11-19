import cv2 as cv
import numpy as np

img = cv.imread('incredible.webp', cv.IMREAD_GRAYSCALE)

cv.threshold(img, 128, 255, cv.THRESH_BINARY, img)

cv.namedWindow('erode')
sizeErode = 1


def erode_func():
    size = sizeErode * 2 + 1
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (size, size))

    img_erode = cv.erode(img, kernel)
    cv.imshow('erode', img_erode)


def changeESize(x):
    global sizeErode
    sizeErode = x
    print(sizeErode)
    erode_func()


cv.createTrackbar('erodSize', "erode", sizeErode, 17, changeESize)


cv.namedWindow('dilate')
sizeDrode = 1


def dilate_func():
    size = sizeDrode * 2 + 1
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (size, size))

    img_dilate = cv.dilate(img, kernel)
    cv.imshow('dilate', img_dilate)


def changeDSize(x):
    global sizeDrode
    sizeDrode = x
    print(sizeDrode)
    dilate_func()


cv.createTrackbar('dilateSize', "dilate", sizeDrode, 17, changeDSize)

cv.namedWindow('morph')
sizeMrode = 1


def morph_func():
    size = sizeMrode * 2 + 1
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (size, size))
    img_morph = cv.morphologyEx(img, cv.MORPH_GRADIENT, kernel)
    cv.imshow('morph', img_morph)


def changeMSize(x):
    global sizeMrode
    sizeMrode = x
    morph_func()


cv.createTrackbar('morphSize', "morph", sizeMrode, 17, changeMSize)


# cv.imshow('image', img)
cv.waitKey(0)
cv.destroyAllWindows()
