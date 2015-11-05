import numpy as np
import cv2
import time

# TODO - camera calibration:
# http://opencv-python-tutroals.readthedocs.org/en/latest/py_tutorials/py_calib3d/py_calibration/py_calibration.html
# http://www.pyimagesearch.com/2015/01/19/find-distance-camera-objectmarker-using-python-opencv/
# http://www.pyimagesearch.com/2014/09/01/build-kick-ass-mobile-document-scanner-just-5-minutes/

# calculating contour sizes
MIN_THRESHOLD = 500
MAX_THRESHOLD = 100000

# square config
SQUARE_N = 5

# linear gradient config
#GRADIENT = []
#GRADIENT_KNOWN_WIDTH = 12*12

#(7.5)*((240/sin(20.705 deg)))
#PPI = 5091.1244048

# get hardlight contours of an image
# returns a tuple of (contours, hardlight_filtered_image)
def hardlight_contours(image_src, take_first=100):
    img = cv2.imread(image_src, 0) # 0 means grayscale
    # force hard black and white
    # hardlight with grayscale
    ret, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # find contours, make sure to copy image otherwise it gets distorted
    (image, cnts, _) = cv2.findContours(img.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:take_first]

    return (cnts, img)

# returns (array of tuples (object_code, size (w,h), rectangle), image)
def identify_objects(image_src):
    (cnts, img) = hardlight_contours(image_src)
    # loop over our contours
    objects = []
    for c in cnts:
        # approximate the contour
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.01 * peri, True)

        # if our approximated contour has four points and area is within our range
        #area = cv2.contourArea(approx)
        c_rect = x,y,w,h = cv2.boundingRect(c)
        area = w*h
        if len(approx) == 4 and area > MIN_THRESHOLD and area <= MAX_THRESHOLD:
            norm_img = four_point_transform(img, approx)
            # hardlight
            ret, norm_img = cv2.threshold(norm_img, 0, 255, cv2.THRESH_BINARY)
            rect = cv2.boundingRect(norm_img)
            code = get_object_code(norm_img, rect)
            if code > 0:
                objects.append((code, (rect[2], rect[3]), c_rect))

    return (objects, img)

# known(pixels, dist)
def locate_xy(image_src, known, out):
    # debug stuff
    img_color = cv2.imread(image_src)

    current_x = -1
    current_y = -1

    # find qr codes in image
    objs, img = identify_objects(image_src)

    # get center of camera
    rect = cv2.boundingRect(img)
    camera_cx = rect[2] / 2
    camera_cy = rect[3] / 2

    # find closest qr code to origin
    closest = None
    dist = -1
    d = 0
    for o in objs:
        code, size, pos_rect = o

        d = pr_dist_sq(camera_cx, camera_cy, pos_rect)
        if (closest is None or d > dist):
            closest = o
            dist = d
        x,y,w,h = pos_rect

        # debug stuff
        cv2.rectangle(img_color, (x, y), (x + w, y + h), (0,0,255), 3)
        cv2.putText(img_color, str(code), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,0,0))
        cv2.imwrite(out, img_color)

        current_y = float(known[0])/float(size[0]) * float(known[1])
        # 7.5 * (240 / sin(20.705)) / (S_pixel)
        #current_y = 5091.1244048 / float(size[0])
    
    print(closest)
    dx = np.fabs(rect_center(closest[2])[0] - camera_cx)
    # current_y * dx / (240 / sin(20.705))

    return (current_x, current_y)
#checkyourdongle

# x = 0, y = 1, w = 2, h = 3
def rect_center(rect):
    return (rect[0] + rect[2] / 2, rect[1] + rect[3] / 2)

def pr_dist_sq(x,y,rect):
    dx = x - (rect[0] + rect[2] / 2)
    dy = y - (rect[1] + rect[3] / 2)
    return x*x + y*y

# find an object's code from a full-sized image object
# TODO: to make this better, instead of sampling one pixel average a small area within each shape
def get_object_code(image, rect):
    x,y,w,h = rect
    cx = x + w/2
    cy = y + h/2
    dx = w/SQUARE_N
    dy = h/SQUARE_N
    val = 511 - (binary(image[cy - dy, cx - dx]) * 1 + binary(image[cy - dy, cx]) * 2 + binary(image[cy - dy, cx + dx]) * 4 + \
           binary(image[cy, cx - dx]) * 8 + binary(image[cy, cx]) * 16 + binary(image[cy, cx + dx]) * 32 + \
           binary(image[cy + dy, cx - dx]) * 64 + binary(image[cy + dy, cx]) * 128 + binary(image[cy + dy, cx + dx]) * 256)
    # TODO: verify all 4 corners are black
    return val if val >= 0 else 0

# returns either 0 or 1 if val is less than mid of max_val or greater
def binary(val, max_val=255):
    return 0 if val < max_val / 2 else 1

# normalizes an object with coordinates pts on image
def four_point_transform(image, pts):
    # obtain a consistent order of the points and unpack them individually
    rect = order_points(pts.reshape(4,2))
    (tl, tr, br, bl) = rect

    # compute the width of the new image, which will be the
    # maximum distance between bottom-right and bottom-left
    # x-coordiates or the top-right and top-left x-coordinates
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
 
    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    # now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left
    # order
    dst = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]], dtype = "float32")

    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    # return the warped image
    return warped

def order_points(pts):
    # initialzie a list of coordinates that will be ordered
    # such that the first entry in the list is the top-left,
    # the second entry is the top-right, the third is the
    # bottom-right, and the fourth is the bottom-left
    rect = np.zeros((4, 2), dtype = "float32")

    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = pts.sum(axis = 1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis = 1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    # return the ordered coordinates
    return rect

# compute and retrn the distance from the maker to the camera
def distance_to_camera(knownWidth, focalLength, perWidth):
    return (knownWidth * focalLength) / perWidth

# calibrate
objects, image = identify_objects('image/12ft.jpg')
known_dist = 12
w,h = objects[0][1]
print("Focal length at", known_dist, "feet is", w, "px")
#for i in [2,3,4,6,8,10,12,'6-rot']:
"""
for i in ['setup']:
    src = 'image/' + str(i) + 'ft.jpg'
    out = "out/" + str(i) + '.jpg'
    #print "locating", i
    print locate_xy(src, (known_dist, w), out)
"""

# inches
HEIGHT = 61.25
object_points = {
# negative
    '322': (-60.875, 0, 60.625),
    '257': (-63.875, 0, 23.5),
    '195': (-39.375, 0, 31.875),
    '170': (-34.25,  0, 60.25),
    '16':  (-27.875, 0, 45.25),
    '381': (-18.75,  0, 19.75),
    '204': (-21,     0, 73.125),
    '14':  (-15.25,  0, 60.125),
    '58':  (-4.375,  0, 36.625),
    '273': (0,       0, 60.375),
    '493': (6.25,    0, 75.375),
    '295': (10.75,   0, 60.25),
    '427': (17.5,    0, 44.75),
    '490': (26.5,    0, 74.5),
    '56':  (30.25,   0, 59.875),
    '23':  (44.625,  0, 70),
    '432': (44.25,   0, 35),
    '337': (57.375,  0, 60.125)
}

camera_matrix = np.eye(3)
camera_matrix[0,0] = 706.64790925
camera_matrix[0,2] = 391.97561153
camera_matrix[1,1] = 738.94091402
camera_matrix[1,2] = 285.82410445

dist_coefs = np.zeros((5,1))
dist_coefs[0] = 0.07060063
dist_coefs[1] = -0.23243439
dist_coefs[2] = -0.02683919
dist_coefs[3] = -0.00194744
dist_coefs[4] = 0.24594352

#for i in [2,3,4,6,8,10,12,'6-rot']:
for i in [8]:
    src = 'image/' + str(i) + 'ft.jpg'
    out = "out/" + str(i) + '.jpg'
    print("locating", i)
    #locate_xyz(src, (known_dist, w), out)

    objects, image = identify_objects(src)
    objs = []
    imgs = []
    for obj in objects:
        code, size, rect = obj
        center = rect_center(rect)

        point = object_points[str(code)]
        imgs.append([center])
        objs.append([point])

    objs = np.array(objs, dtype=np.float)
    imgs = np.array(imgs, dtype=np.float)

    solve = cv2.solvePnPRansac(objs, imgs, camera_matrix, dist_coefs)
    print(solve)
    break

