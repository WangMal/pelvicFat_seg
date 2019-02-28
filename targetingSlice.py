# -*- coding: UTF-8 -*-

import numpy as np
import nrrd
import cv2
import scipy
from scipy import misc

def nrrd2np(Nrrd):
    nrrd_data, nrrd_option = nrrd.read(Nrrd)
    pieces = len(nrrd_data[0, 0, :])
    nrrd_array = np.array(nrrd_data, dtype=float)

    # 改变矩阵的维度
    resize = np.zeros((512, 512, pieces), dtype=float)
    output_array = np.zeros((512, 512, pieces), dtype=float)
    for i in range(min(624, nrrd_array.shape[2])):
        resize[:, :, i] = nrrd_array[:, :, i]
    for i in range(resize.shape[2]):
        output_array[:, :, i] = cv2.resize(resize[:, :, i], (512, 512), interpolation=cv2.INTER_CUBIC)
    return output_array, pieces


def findTargetRegion(path):
    patient, pieces = nrrd2np(path+'/org.nrrd')
    bone, piecesBone = nrrd2np(path+'/bone.nrrd')
    # muscle = nrrd2np('./MuscleClosing3.nrrd')

    # 从pic0开始（底部），左右为1最大距离小于200cut
    for t in range(pieces):
        # 从左开始一列一列取，有1时记录列号 break 开始从右往左
            # 从右往左一列一列取，有1时记录列号break
        pic = cv2.resize(np.transpose(bone[:, :, t]), (512, 512), interpolation=cv2.INTER_CUBIC)

        for left in range(512):
            line = pic[:, left]
            if len(set(line)) == 2:  # line里只能是0和1
                break
        for right in range(511, 0, -1):
            line = pic[:, right]
            if len(set(line)) == 2:  # line里只能是0和1
                break

        if right - left < 200:
            break
    t = t-1
    print('top = ', t)

    flag = 'None'
    # 从底部开始找耻骨联合位置
    for b in range(pieces):
         # print('pic: ', b)
        pic = cv2.resize(np.transpose(bone[:, :, b]), (512, 512), interpolation=cv2.INTER_CUBIC)
        center = 255
        if len(set(pic[:, center])) == 1:
            toleft = center
            toright = center
            while True:
                toleft -= 1
                line = pic[:, toleft]
                if len(set(line)) == 2:
                    left = toleft
                    break
            while True:
                toright += 1
                line = pic[:, toright]
                if len(set(line)) == 2:
                    right = toright
                    break
            if right - left < 7:         # 耻骨联合
                break
        elif len(set(pic[:, center])) == 2:
            break
    print('bottom = ', b)


     # 存储pic区域
    region_pic_bone = np.zeros([512, 512, t-b+1])
    region_pic_patient = np.zeros([512, 512, t-b+1])
    # region_pic_muscle = np.zeros([512, 512, t-b+1])
    for p in range(t-b+1):
        region_pic_bone[:, :, p] = cv2.resize(bone[:, :, b+p], (512, 512), interpolation=cv2.INTER_CUBIC)
        region_pic_patient[:, :, p] = cv2.resize(patient[:, :, b+p], (512, 512), interpolation=cv2.INTER_CUBIC)
        # region_pic_muscle[:, :, p] = cv2.resize(muscle[:, :, b+p], (512, 512), interpolation=cv2.INTER_CUBIC)
        # scipy.misc.imsave('./targetJpg/P1/pic%d.jpg' % p, cv2.resize(patient[:, :, b+p], (512, 512), interpolation=cv2.INTER_CUBIC))

    nrrd.write(path+'/target_bone.nrrd', region_pic_bone)
    nrrd.write(path+'/target.nrrd', region_pic_patient)
    # nrrd.write('./region_pic_muscle.nrrd', region_pic_muscle)


for i in range(11, 43):
    path = './nrrd/p%d' % i
    print(path)
    findTargetRegion(path)
