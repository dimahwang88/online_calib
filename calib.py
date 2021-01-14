


import sys
import mmcv
import cv2

from epipolar import findFundamentalMatrix
from epipolar import findEssentialMatrixFromF
from epipolar import findRTfromE

from utils import loadDetections
from utils import loadCameraParameters
from utils import dumpPairFromVideo

import os

def calibrate(path):

    intpath = os.path.join(path, "intrinsic.yaml")
    extpath = os.path.join(path, "extrinsic.yml")

    K, d, Ri, Ti = loadCameraParameters(intpath, extpath)

    impath1 = os.path.join(path, "input_1.jpg")
    impath2 = os.path.join(path, "input_2.jpg")
    
    im1 = cv2.imread(impath1,0)
    im2 = cv2.imread(impath2,0)
    
    detpath1 = os.path.join(path, "det_im1.txt")
    detpath2 = os.path.join(path, "det_im2.txt")

    dets1 = loadDetections(detpath1)
    dets2 = loadDetections(detpath2)

    # find fundamental matrix between 2 views
    F = findFundamentalMatrix(im1, im2, dets1, dets2)
    E = findEssentialMatrixFromF(F, K)
    R,t = findRTfromE(E)

if __name__ == "__main__":
    calibrate(sys.argv[1])
    # dumpPairFromVideo(sys.argv[1])
