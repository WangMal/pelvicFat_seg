# -*- coding: UTF-8 -*-

import numpy as np
import nrrd
import cv2
import matplotlib.pyplot as plt
from part1_fat import FatPart1
import imageio


class SegFat:
    def __init__(self):
        # ========  读part2  ========
        path = './result/nrrd'
        self.target = self.nrrd2np(path + '/data_part2.nrrd')
        self.bone = self.nrrd2np(path + '/bone_part2.nrrd')
        self.fat = self.nrrd2np(path + '/fat_part2.nrrd')
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

    def get_edge(self, pic):
        #  ========  获取边缘点集坐标   ========
        edge_set = []
        for c in range(512):  # 每一行找左右
            flagl = 0
            flagr = 0
            for l in range(256):
                if pic[c, l] > 0:
                    flagl = 1
                    left = l
                    break
            for r in range(511, 256, -1):
                if pic[c, r] > 0:
                    flagr = 1
                    right = r
                    break
            if flagl == 1:
                edge_set.append([c, l])
            if flagr == 1:
                edge_set.append([c, r])
        print('目标边缘点集个数：', len(edge_set))
        # print('edge = ', edge_set)
        return edge_set

    def fatRemoveDown(self, pic_fat, pic_bone):
        # ========  去除骨头下方的点  ========
        fat_set = []
        for c in range(512):
            col_fat = pic_fat[:, c]  # 脂肪的每一列
            col_bone = pic_bone[:, c]  # 骨头的每一列
            fat_1 = np.where(col_fat == 1)[0]  # 脂肪是1的部分[fa,fb]
            if len(fat_1):
                ma_f = max(fat_1)
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
                        fat_set.append([f, c])
        # print(np.transpose(fat_set)[0])
        print('去除骨头下方后，脂肪像素点个数：', len(fat_set))
        return fat_set

    def fatRemoveRight(self, pic_fat, pic_bone):
        # ========  去除骨头右的点  ========
        fat_set = []
        for c in range(512):
            col_fat = pic_fat[c, :]  # 脂肪的每一行
            col_bone = pic_bone[c, :]  # 骨头的每一行

            fat_1 = np.where(col_fat == 1)[0]  # 脂肪是1的部分[fa,fb]
            bone_1 = np.where(col_bone == 1)[0]

            if len(fat_1):
                if len(bone_1) > 0 and max(bone_1) > 265:
                    for f in fat_1:
                        if f < max(bone_1):
                            fat_set.append([c, f])
                else:
                    for f in fat_1:
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
            bone_1 = np.where(col_bone == 1)[0]

            if len(fat_1):
                if len(bone_1) > 0 and min(bone_1) < 265:
                    for f in fat_1:
                        if f > min(bone_1):
                            fat_set.append([c, f])
                else:
                    for f in fat_1:
                        fat_set.append([c, f])

        # print(np.transpose(fat_set)[0])
        print('去除骨头左边，脂肪像素点个数：', len(fat_set))
        return fat_set

    def create_fig(self, fat_set, p):
        plt.rcParams['figure.figsize'] = (5.0, 5.0)
        plt.xlim(0, 512)
        plt.ylim(0, 512)
        plt.title('piece%d' % p)
        plt.scatter(np.transpose(fat_set)[0], np.transpose(fat_set)[1])
        # plt.plot(fat_set, 'ro')
        plt.show()

    def change2dotset(self, pic):
        dataset = []
        for i in range(512):
            for j in range(512):
                if pic[i, j] == 1:
                    dataset.append([i, j])
        return dataset


s2 = SegFat()
pieces = s2.pieces
fat_part2 = np.zeros([512, 512, pieces], dtype=float)

for p in range(51, pieces):
    # for p in range(0, 10):
    print('pieces: ', p)
    # ===================             对每一切片上脂肪建立 dataSet            ===================
    # ---------------------------------------------------------------------------------------
    # |          | location | 是否有bone？| 离边缘距离 | 最短距离点连接线ε近邻带脂肪 | pelvic fat？ |
    # |          |  左 ？右  | 上 下 左 右 |          |                        |              |
    # ---------------------------------------------------------------------------------------
    # | [x1, y1] |          |1 |  |  |  |           |                        |      0       |
    # | [x2, y2] |          |0 |  |0 |1 |           |                        |      0       |
    # | [x3, y3] |   left   |0 |  |1 |0 |           |                        |      0       |
    # | [x4, y4] |   right  |  |  |  |  |   > dis   |                        |      1       |
    # | [x5, y5] |          |  |  |  |  |   < dis   |       > percent        |      0       |
    # | [x6, y6] |          |  |  |  |  |           |       > percent        |      1       |
    # ---------------------------------------------------------------------------------------
    target_p = s2.change2dotset(s2.target[:, :, p])

    fat_p = s2.fat[:, :, p]
    bone_p = s2.bone[:, :, p]
    data_p = s2.target[:, :, p]
    # fat_p_coordinate = s2.change2dotset(fat_p)
    fat_p_coordinate = s2.fatRemoveDown(fat_p, bone_p)  # 去除骨头下方的脂肪
    fat_p = np.zeros([512, 512], dtype='int')
    for f in fat_p_coordinate:
        fat_p[f[0], f[1]] = 1

    fat_p_coordinate = s2.fatRemoveRight(fat_p, bone_p)
    fat_p = np.zeros([512, 512], dtype='int')
    for f in fat_p_coordinate:
        fat_p[f[0], f[1]] = 1

    fat_p_coordinate = s2.fatRemoveLeft(fat_p, bone_p)
    fat_p = np.zeros([512, 512], dtype='int')
    for f in fat_p_coordinate:
        fat_p[f[0], f[1]] = 1

    bone_p_coordinate = s2.change2dotset(bone_p)
    edge_p_coordinate = s2.get_edge(data_p)    # 获取边缘点坐标集

    edge = np.zeros([512, 512], dtype='int')
    for e in edge_p_coordinate:
        edge[e[0], e[1]] = 1

    pelvic_fat = []
    fat_fig = np.zeros([512, 512], dtype='int')

    for f in fat_p_coordinate:
        label = 0
        fat = FatPart1(f[0], f[1], bone_p, edge_p_coordinate)  # class Fat
        [dis2Edge, nearestDot2Edge] = fat.__dot2edge__(edge_p_coordinate)
        if dis2Edge > 25:
            fat_fig[f[0], f[1]] = 1
        elif 25 < dis2Edge < 40:
            neardot = fat.neardot(fat.x, fat.y, nearestDot2Edge)  # 寻找到fat和边界点连线距离小于d的点
            nearFatRate = fat.nearfatrate(neardot, fat_p_coordinate)  # 最短距离点连接线ε近邻带脂肪
            if nearFatRate < 0.35:
                fat_fig[f[0], f[1]] = 1

    fat_part2[:, :, p] = np.transpose(fat_fig)

    imageio.imwrite('./result/fat/pic/part2-1/pic%d.jpg' % p,
                    cv2.resize(fat_fig, (512, 512), interpolation=cv2.INTER_CUBIC))

nrrd.write('./result/fat/part1.nrrd', fat_part2)


