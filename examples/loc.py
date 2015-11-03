# import the necessary packages
#from skimage import exposure
import numpy as np
import cv2

# load the query image, compute the ratio of the old height
# to the new height, clone it, and resize it

def get_contours(image, out):
    image = cv2.imread(image)
     
    # convert the image to grayscale, blur it, and find edges
    # in the image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 11, 17, 17)
    edged = cv2.Canny(gray, 30, 200)

    # find contours in the edged image, keep only the largest
    # ones, and initialize our screen contour
    (img, cnts, _) = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:100]
    screenCnt = None

    # loop over our contours
    for c in cnts:
        # approximate the contour
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.01 * peri, True)

        # if our approximated contour has four points, then
        # we can assume that we have found our screen
        if len(approx) == 4:
            cv2.drawContours(image, [approx], -1, (0, 255, 0), 3)
    #cv2.imshow("Image", image)
    cv2.imwrite(out, image)

def get_bw_cont(image, out):
    imgcolor = cv2.imread(src)
    img = cv2.imread(src, 0)
    #ret,thresh = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
    ret,thresh = cv2.threshold(img,0,255,cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    #img = cv2.imread(src, 0)
    #ret, threshold = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    #cv2.imshow('hello', thresh)
    #cv2.waitKey(0)
    #continue
    #edged = cv2.Canny(thresh, 30, 200)

    (image, cnts, _) = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:100]
    screenCnt = None

    # loop over our contours
    for c in cnts:
        # approximate the contour
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.01 * peri, True)

        # if our approximated contour has four points, then
        # we can assume that we have found our screen
        #print approx
        if len(approx) == 4:
            cv2.drawContours(imgcolor, [approx], -1, (0, 255, 0), 3)
    #cv2.imshow("Image", image)
    #cv2.waitKey(0)
    cv2.imwrite(out, imgcolor)

for i in [2,3,4,6,8,10,12]:
    #get_contours(src, str(i) + '.jpg')
    src = 'image/' + str(i) + 'ft.jpg'
    out = "out-bw/" + str(i) + '.jpg'
    get_bw_cont(src, out)
