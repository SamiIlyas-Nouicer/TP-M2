import cv2 as cv
import numpy as np
img = cv.imread('pic2.jpeg', cv.IMREAD_COLOR)
img_b = np.zeros(img.shape, img.dtype)
img_g = np.zeros(img.shape, img.dtype)
img_r = np.zeros(img.shape, img.dtype)
print("img shape is", img.shape)
h, w, c = img.shape
# for i in range(h):
#     for j in range(w):
#         img_b[i, j, 0] = img[i, j, 0]
#         img_g[i, j, 1] = img[i, j, 1]
#         img_r[i, j, 2] = img[i, j, 2]


img_b[:, :, 0], img_g[:, :, 1], img_r[:, :,2] = img[:,:,0] ,img[:,:,1],img[:,:,2]

img_gray = img[:, :, 0] / 3 + img[:, :, 1] / 3 + img[:, :, 2] / 3

img_hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
img_float32 = np.float32(img)/255
cv.imshow('image float32', img_float32)
# cv.imshow('image U16', np.uint16(img_float32)*(2**16-1))
cv.imshow('image U16-2', np.uint16(img_float32*(2**16-1)))
# cv.imshow('blue, img_b)
# cv.imshow('green', img_g)
# cv.imshow('red', img_r)
# cv.imshow('moyenne', np.uint8(img_gray))
cv.imshow('hsv', img_hsv)
cv.waitKey(0)
cv.destroyAllWindows()
