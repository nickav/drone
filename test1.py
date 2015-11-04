import cv2
import numpy as np

objectPoints = np.random.random((10,3,1))

imagePoints = np.random.random((10,2,1))

print imagePoints

cameraMatrix = np.eye(3)

distCoeffs = np.zeros((5,2))

print cv2.solvePnP(objectPoints, imagePoints, cameraMatrix, distCoeffs)
