


import sys
import mmcv
import cv2

from epipolar import findFundamentalMatrix
from utils import loadDetections

def calibrate():
    # open 2 images
    im1 = cv2.imread('./data/input_1.jpg')
    im2 = cv2.imread('./data/input_2.jpg')
    
    dets1 = loadDetections('./data/det_im1.txt')
    dets2 = loadDetections('./data/det_im2.txt')

    # find fundamental matrix between 2 views
    findFundamentalMatrix(im1, im2, dets1, dets2)

if __name__ == "__main__":
    calibrate()