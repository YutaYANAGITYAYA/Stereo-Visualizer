# # -*- coding: utf-8 -*-
import numpy as np
import cv2
import glob

imDir = "chesspattern" # キャリブレーション画像のあるディレクトリ
square_size = 31.5     # 格子サイズ
pattern_size = (9, 6)  # 格子数


pattern_points = np.zeros((np.prod(pattern_size), 3), np.float32)
pattern_points[:,:2] = np.indices(pattern_size).T.reshape(-1, 2)
pattern_points *= square_size
term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 1000, 0.0001)
im = cv2.imread(glob.glob("%s\\*.jpg"%imDir)[0])
imSize = im.shape[:2][::-1]

objPoints_l = []
objPoints_r = []
objPoints_stereo = []
imPoints_l = []
imPoints_r = []
imPoints_stereo1 = []
imPoints_stereo2 = []

imlst = glob.glob("%s\\left*.jpg"%imDir)
for fn in imlst:
    print ("loading image... %s/%d"%(fn[len(imDir)+5:-4],len(imlst)))
    im_l = cv2.imread(fn)
    im_r = cv2.imread(fn.replace("left","right"))
    gray_l = cv2.cvtColor(im_l,cv2.COLOR_BGR2GRAY)
    gray_r = cv2.cvtColor(im_r,cv2.COLOR_BGR2GRAY)

    ret_l, corner_l = cv2.findChessboardCorners(gray_l, pattern_size)
    if ret_l:
        print ("found corners in left image")
        cv2.cornerSubPix(gray_l, corner_l, (5,5), (-1,-1), term)
        cv2.drawChessboardCorners(im_l, pattern_size, corner_l, ret_l)
        imPoints_l.append(corner_l.reshape(-1, 2))
        objPoints_l.append(pattern_points)

    ret_r, corner_r = cv2.findChessboardCorners(gray_r, pattern_size)
    if ret_r:
        print ("found corners in right image")
        cv2.cornerSubPix(gray_r, corner_r, (5,5), (-1,-1), term)
        cv2.drawChessboardCorners(im_r, pattern_size, corner_r, ret_r)
        imPoints_r.append(corner_r.reshape(-1, 2))
        objPoints_r.append(pattern_points)

    if ret_l and ret_r:
        imPoints_stereo1.append(corner_l.reshape(-1, 2))
        imPoints_stereo2.append(corner_r.reshape(-1, 2))
        objPoints_stereo.append(pattern_points)

    cv2.destroyAllWindows()
    imRes = cv2.hconcat([im_l, im_r])
    cv2.imshow("result", imRes)
    cv2.waitKey(100)

cv2.destroyAllWindows()


retval, K_l, d_l, r, t = cv2.calibrateCamera(objPoints_l,imPoints_l,imSize,None,None)
print ("\nLeft Camera\nRMS error = ", retval)
print ("cameraMatrix =\n", K_l)
print ("distCoeffs =\n", d_l)

retval, K_r, d_r, r, t = cv2.calibrateCamera(objPoints_r,imPoints_r,imSize,None,None)
print ("\nRight Camera\nRMS error = ", retval)
print ("cameraMatrix =\n", K_r)
print ("Left distCoeffs =\n", d_r)

retval, cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, R, T, E, F = cv2.stereoCalibrate(objPoints_stereo, imPoints_stereo1, imPoints_stereo2, K_l, d_l, K_r, d_r, imSize)
print ("\nStereo Camera System\nRMS error = ", retval)
print ("Rotation Matrix =\n", R)
print ("Translation Vector =\n", T)


np.savetxt("cameraMatrix1.csv", cameraMatrix1, delimiter =',',fmt="%0.14f")
np.savetxt("cameraMatrix2.csv", cameraMatrix2, delimiter =',',fmt="%0.14f") 
np.savetxt("distCoeffs1.csv", distCoeffs1, delimiter =',',fmt="%0.14f")
np.savetxt("distCoeffs2.csv", distCoeffs2, delimiter =',',fmt="%0.14f")
np.savetxt("R.csv", R, delimiter =',',fmt="%0.14f")
np.savetxt("T.csv", T, delimiter =',',fmt="%0.14f")
