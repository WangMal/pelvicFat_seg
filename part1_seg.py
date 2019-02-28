# -*- coding: UTF-8 -*-

import numpy as np
import nrrd
import cv2
import imageio



class SegFat:
    def __init__(self):
        # ========  读part1  ========
        path = './result/nrrd'
        self.target = self.nrrd2np(path + '/data_part1.nrrd')
        self.bone = self.nrrd2np(path + '/bone_part1.nrrd')
        self.fat = self.nrrd2np(path + '/fat_part1.nrrd')
        self.pieces = len(self.target[0, 0, :])
        print('切片数：piece = ', self.pieces)

    def nrrd2np(self, nrrd_path):
        nrrd_data, nrrd_option = nrrd.read(nrrd_path)
        pieces = len(nrrd_data[0, 0, :])
        nrrd_array = np.array(nrrd_data, dtype=float)
        # 改变矩阵的维度
        resize = np.zeros((512, 512, pieces), dtype=float)
        output_array = np.zeros((512, 512, pieces), dtype=float)
        for i in range(nrrd_array.shape[2]):
            resize[:, :, i] = nrrd_array[:, :, i]
        for i in range(resize.shape[2]):
            output_array[:, :, i] = cv2.resize(resize[:, :, i], (512, 512), interpolation=cv2.INTER_CUBIC)
        return output_array.transpose((1, 0, 2))

    def fatRemoveRight(self, pic_fat, pic_bone):
        # ========  去除骨头右的点  ========
        fat_set = []
        for c in range(512):
            col_fat = pic_fat[c, :]  # 脂肪的每一列
            col_bone = pic_bone[c, :]  # 骨头的每一列
            fat_1 = np.where(col_fat == 1)[0]  # 脂肪是1的部分[fa,fb]
            if len(fat_1):
                ma_f = max(fat_1)
                ma_f = max(ma_f, 260)

                # 找到骨头最下 [bone_ma]
                bone_1 = np.where(col_bone == 1)[0]
                if len(bone_1):
                    ma_b = max(bone_1)
                    max_f = min(ma_b, ma_f)
                else:
                    max_f = ma_f
                # print('max_f ', max_f)
                for f in fat_1:
                    if f < max_f:
                        # print('[c, f]=', [c, f])
                        fat_set.append([c, f])
        # print(np.transpose(fat_set)[0])
        print('去除骨头右边，脂肪像素点个数：', len(fat_set))
        return fat_set

    def fatRemoveLeft(self, pic_fat, pic_bone):
        # ========  去除骨头左的点  ========
        fat_set = []
        for c in range(512):
            col_fat = pic_fat[c, :]  # 脂肪的每一行
            col_bone = pic_bone[c, :]  # 骨头的每一行
            fat_1 = np.where(col_fat == 1)[0]  # 脂肪是1的部分[fa,fb]
            if len(fat_1):
                mi_f = min(fat_1)
                mi_f = min(mi_f, 260)
                # 找到骨头最下 [bone_ma]
                bone_1 = np.where(col_bone == 1)[0]
                if len(bone_1):
                    mi_b = min(bone_1)
                    min_f = max(mi_b, mi_f)
                else:
                    min_f = mi_f
                # print('max_f ', max_f)
                for f in fat_1:
                    if f > min_f:
                        # print('[c, f]=', [c, f])
                        fat_set.append([c, f])
        # print(np.transpose(fat_set)[0])
        print('去除骨头左边，脂肪像素点个数：', len(fat_set))
        return fat_set

    def fatRemoveUpBone(self, pic_fat, pic_bone):
        for i in range(512):
            row = pic_bone[i, :]
            if 1 in row:
                topBone = i
                break
        pic_fat[0:topBone, :] = 0
        return pic_fat

    def fatRemoveDownBone(self, pic_fat, pic_bone):
        for i in range(511, 0, -1):
            row = pic_bone[i, :]
            if 1 in row:
                bottomBone = i
                break
        pic_fat[bottomBone:511, :] = 0
        return pic_fat

    def change2dotset(self, pic):
        dataset = []
        for i in range(512):
            for j in range(512):
                if pic[i, j] == 1:
                    dataset.append([i, j])
        return dataset


s1 = SegFat()
pieces = s1.pieces
fat_part1 = np.zeros([512, 512, pieces], dtype=float)
for p in range(pieces):
    # for p in range(0, 10):
    print('pieces: ', p)
    target_p = s1.change2dotset(s1.target[:, :, p])

    fat_p = s1.fat[:, :, p]
    bone_p = s1.bone[:, :, p]
    data_p = s1.target[:, :, p]

    # 去除骨头左右
    fat_p_coordinate = s1.fatRemoveRight(fat_p, bone_p)
    fat_p = np.zeros([512, 512], dtype='int')
    for f in fat_p_coordinate:
        fat_p[f[0], f[1]] = 1
    fat_p_coordinate = s1.fatRemoveLeft(fat_p, bone_p)

    fat_p = np.zeros([512, 512], dtype='int')
    for f in fat_p_coordinate:
        fat_p[f[0], f[1]] = 1

    fat_p = s1.fatRemoveUpBone(fat_p, bone_p)
    # fat_p = s1.fatRemoveDownBone(fat_p, bone_p)

    fat_p_coordinate = s1.change2dotset(fat_p)

    for f in fat_p_coordinate:
        # print('f= （%d，%d)'% (f[0], f[1]))
        # inx = 0.6*(1-0.1*p/pieces)  # 距离指数 应该随p增大减小
        inx = 0.6 * (1 - 0.1 * p / pieces)
        index = inx*abs(265-f[0])+(1-inx)*abs(data_p[f[0], f[1]])  # 距离-p越小对应像素的值越重要
        if index > 70:
            # print('index=', index)
            fat_p[f[0], f[1]] = 0

    fat_part1[:, :, p] = np.transpose(fat_p)
    imageio.imwrite('./result/fat/pic/part1-2/pic%d.jpg' % p,
                        cv2.resize(fat_p, (512, 512), interpolation=cv2.INTER_CUBIC))

    # scipy.misc.imsave('./fat1/fat%d.jpg' % p, fat_p)
nrrd.write('./result/fat/part1-2.nrrd', fat_part1)





