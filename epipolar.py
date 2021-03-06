import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

from utils import removePointFromRoi
from utils import drawlines

def findFundamentalMatrix(img1, img2, dets1, dets2):
    sift = cv.SIFT_create()
    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)

    # for p in kp1:
    #     cv.circle(img1, (int(p.pt[0]), int(p.pt[1])), 2, (0,255,255), 2)

    # for p in kp2:
    #     cv.circle(img2, (int(p.pt[0]), int(p.pt[1])), 2, (255,0,255), 2)

    # cv.imwrite("./data/sift_1.jpg", img1)
    # cv.imwrite("./data/sift_2.jpg", img2)

    kp1_f, des1_f = removePointFromRoi(dets1, kp1, des1)
    kp2_f, des2_f = removePointFromRoi(dets2, kp2, des2)        

    # for p in kp1_f:
    #     cv.circle(img1, (int(p.pt[0]), int(p.pt[1])), 2, (0,255,255), 2)

    # for p in kp2_f:
    #     cv.circle(img2, (int(p.pt[0]), int(p.pt[1])), 2, (255,0,255), 2)

    # cv.imwrite("./data/sift_1_filter.jpg", img1)
    # cv.imwrite("./data/sift_2_filter.jpg", img2)

    # FLANN parameters
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=50)
    flann = cv.FlannBasedMatcher(index_params,search_params)
    
    # matches = flann.knnMatch(np.asarray(des1,np.float32), np.asarray(des2,np.float32), k=2)
    matches = flann.knnMatch(np.asarray(des1_f,np.float32),np.asarray(des2_f,np.float32),k=2)

    pts1 = []
    pts2 = []
    # ratio test as per Lowe's paper
    for i,(m,n) in enumerate(matches):
        if m.distance < 0.8*n.distance:
            pts2.append(kp2_f[m.trainIdx].pt)
            pts1.append(kp1_f[m.queryIdx].pt)
    
    pts1 = np.int32(pts1)
    pts2 = np.int32(pts2)
    F, mask = cv.findFundamentalMat(pts1,pts2,cv.FM_LMEDS)
    # We select only inlier points
    pts1 = pts1[mask.ravel()==1]
    pts2 = pts2[mask.ravel()==1]

    # Find epilines corresponding to points in right image (second image) and
    # drawing its lines on left image
    lines1 = cv.computeCorrespondEpilines(pts2.reshape(-1,1,2), 2,F)
    lines1 = lines1.reshape(-1,3)

    img5,img6 = drawlines(img1,img2,lines1,pts1,pts2)
    
    # Find epilines corresponding to points in left image (first image) and
    # drawing its lines on right image
    lines2 = cv.computeCorrespondEpilines(pts1.reshape(-1,1,2), 1,F)
    lines2 = lines2.reshape(-1,3)
    img3,img4 = drawlines(img2,img1,lines2,pts2,pts1)

    plt.figure(figsize=(20,6))
    plt.subplot(121),plt.imshow(img5)
    plt.subplot(122),plt.imshow(img3)

    plt.savefig('./data/epipolar.jpg')
    
    return F

def findEssentialMatrixFromF(F, K):
    print ("Fundamental Matrix")
    print (F)

    print ("Camera Matrix")
    print (K)

    E = K.transpose() * F * K

    print ('Essential Matrix')    
    print (E)

    return E

def findRTfromE(E):
    print ('SVD of E:')
    U,S,Vt = np.linalg.svd(E)

    print (U)
    print (S)
    print (Vt)

    W = np.array([[0,-1,0],[1,0,0],[0,0,1]])
    Z = np.array([[0,1,0],[-1,0,0],[0,0,0]])

    R = U * np.linalg.inv(W) * Vt
    t_cp = U * Z * U.transpose()
    
    t1 = t_cp[2][1]
    t2 = t_cp[0][2]
    t3 = t_cp[1][0]
    T = np.array([[t1],[t2],[t3]])

    print (R)
    print (T)

    return R,T