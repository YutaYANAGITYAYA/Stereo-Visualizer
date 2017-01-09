# -*- coding: utf-8 -*-
import numpy as np
import cv2

im1,im2 = "ts_L.jpg", "ts_R.jpg"
TgtImg_l = cv2.imread(im1)
TgtImg_r = cv2.imread(im2)
for i in range(2):
    TgtImg_l = cv2.bilateralFilter(TgtImg_l,7,15,15)
    TgtImg_r = cv2.bilateralFilter(TgtImg_r,7,15,15)
cameraMatrix1 = np.loadtxt('cameraMatrix1.csv',delimiter = ',')
cameraMatrix2 = np.loadtxt('cameraMatrix2.csv',delimiter = ',')
distCoeffs1 = np.loadtxt('distCoeffs1.csv',delimiter = ',')
distCoeffs2 = np.loadtxt('distCoeffs2.csv',delimiter = ',')
R = np.loadtxt('R.csv',delimiter = ',')
T = np.loadtxt('T.csv',delimiter = ',')
imageSize = TgtImg_l.shape[:2][::-1]


# 平行化変換のためのRとPおよび3次元変換行列Qを求める
R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = cv2.stereoRectify(cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, imageSize, R, T, flags=0, alpha=1)

# 平行化変換マップを求める
m1type = cv2.CV_32FC1
map1_l, map2_l = cv2.initUndistortRectifyMap(cameraMatrix1, distCoeffs1, R1, P1, imageSize, m1type)
map1_r, map2_r = cv2.initUndistortRectifyMap(cameraMatrix2, distCoeffs2, R2, P2, imageSize, m1type)

# ReMapにより平行化を行う
interpolation = cv2.INTER_NEAREST
Re_TgtImg_l = cv2.remap(TgtImg_l, map1_l, map2_l, interpolation)
Re_TgtImg_r = cv2.remap(TgtImg_r, map1_r, map2_r, interpolation)

imResL = Re_TgtImg_l.copy()
imResR = Re_TgtImg_r.copy()
cv2.rectangle(imResL, tuple(validPixROI1[:2]), tuple(validPixROI1[2:]), (0,255,0), 2)
cv2.rectangle(imResR, tuple(validPixROI2[:2]), tuple(validPixROI2[2:]), (0,255,0), 2)
imRes = cv2.hconcat([imResL, imResR])
cv2.imshow('Rectified Image', imRes)
cv2.waitKey(10)


# セミグローバルブロックマッチング(SGBM)
window_size = 3
min_disp = 24
num_disp = 16*8
stereo = cv2.StereoSGBM_create(
    minDisparity = min_disp,
    numDisparities = num_disp,
    blockSize = window_size,
    P1 =  8*3*window_size**2,
    P2 = 32*3*window_size**2,
    disp12MaxDiff = 4,
    uniquenessRatio = 10,
    speckleWindowSize = 150,
    speckleRange = 16,
    mode = True
    )


# 視差を求める
print ('computing disparity...')
disp = stereo.compute(Re_TgtImg_l, Re_TgtImg_r).astype(np.float32) / 16.0
disp = disp[validPixROI1[1]:validPixROI1[3],validPixROI1[0]:validPixROI1[2]]

# 3次元座標への変換
import pylab as plt
ply_header = '''ply
format ascii 1.0
element vertex %(vert_num)d
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header'''

# ply形式の3Dモデルファイルを生成
def write_ply(fn, verts, colors):
    verts = verts.reshape(-1, 3)
    colors = colors.reshape(-1, 3)
    verts = np.hstack([verts, colors])
    h = ply_header % dict(vert_num=len(verts))
    np.savetxt(fn, verts, fmt="%f %f %f %d %d %d", header=h, comments='')


# 視差画像からx,y,z座標を取得
print ('generating 3d point cloud...')
points = cv2.reprojectImageTo3D(disp, Q)
colors = cv2.cvtColor(Re_TgtImg_l[validPixROI1[1]:validPixROI1[3],validPixROI1[0]:validPixROI1[2]],cv2.COLOR_BGR2RGB)
mask = disp > disp.min() + min_disp
out_points = points[mask]
out_colors = colors[mask]
write_ply("out.ply", out_points, out_colors)

# 結果表示
im_disp = (disp-min_disp)/(num_disp-min_disp)
cv2.imshow("result",im_disp)
cv2.waitKey(0)
cv2.destroyAllWindows()
