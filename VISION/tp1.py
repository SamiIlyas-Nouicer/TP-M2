import cv2 as cv
import numpy as np

img = cv.imread("pic3.jpg")
if img is None:
    print("Could not open or find the image")
    exit(0)
else:
    cv.imshow("the pic3", img)
    cv.waitKey(0)
    cv.destroyAllWindows()
