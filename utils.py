import cv2 as cv
import numpy as np
import mmcv
import copy 

def loadDetections(txt_path):
    dets = []
    with open(txt_path) as f:

        for line in f:
            line = line.rstrip('\n')
            split = line.split(',')
            dets.append((int(split[1]), int(split[2]), int(split[3]), int(split[4])))

    return dets

def isInsideBox(pt, boxes):
    x, y = pt[0], pt[1]

    for box in boxes:
        x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
        if x >= x1 and x <= x2 and y >= y1 and y <= y2:
            return True

    return False

def removePointFromRoi(boxes, points, descriptors):
    points_filt = []
    descr_filt = []
    indices = []

    for i,p in enumerate(points):
        if not isInsideBox((p.pt[0], p.pt[1]), boxes):
            points_filt.append(p)
            indices.append(i)

    for i,d in enumerate(descriptors):
        if i in indices:
            descr_filt.append(d)

    return points_filt, descr_filt

def drawlines(img1,img2,lines,pts1,pts2):
    ''' img1 - image on which we draw the epilines for the points in img2
        lines - corresponding epilines '''
    
    r , c = img1.shape[0], img1.shape[1]

    img1 = cv.cvtColor(img1, cv.COLOR_GRAY2BGR)
    img2 = cv.cvtColor(img2, cv.COLOR_GRAY2BGR)

    for r,pt1,pt2 in zip(lines,pts1,pts2):
        color = tuple(np.random.randint(0,255,3).tolist())
        x0,y0 = map(int, [0, -r[2]/r[1] ])
        x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
        img1 = cv.line(img1, (x0,y0), (x1,y1), color,1)
        img1 = cv.circle(img1,tuple(pt1),5,color,-1)
        img2 = cv.circle(img2,tuple(pt2),5,color,-1)
    return img1,img2

def loadCameraParameters(intpath, extpath):
    fs1 = cv.FileStorage(intpath, cv.FILE_STORAGE_READ)
    fn_k = fs1.getNode("camera_matrix")
    fn_d = fs1.getNode("distortion_coefficients")
    
    K = fn_k.mat()
    d = fn_d.mat()

    fs2 = cv.FileStorage(extpath, cv.FILE_STORAGE_READ)
    fn_r = fs2.getNode("Camera 0").getNode("R")
    fn_t = fs2.getNode("Camera 0").getNode("t")

    R = fn_r.mat()
    t = fn_t.mat()

    return K, d, R, t


def dumpPairFromVideo(path):

    video = mmcv.VideoReader(path)

    for index in range(0, 8002):
        frame = video[index]

        if index+1 >= 6100 and index+1 <= 7000:
            cv.imwrite('./tmp/input_%d.jpg' % (index+1), frame)


