import numpy as np
import cv2
from IPython import embed

im = cv2.imread('image/10ft.jpg')
imgray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
ret,thresh = cv2.threshold(imgray,127,255,0)
im2, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
#cv2.drawContours(im2, contours, -1, (0,255,0), 3)

# black outlines
cv2.imwrite('10.jpg', im2)

