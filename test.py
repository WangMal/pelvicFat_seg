# -*- coding: UTF-8 -*-

import numpy as np
import nrrd
import cv2
import scipy
from scipy import misc
import matplotlib.pyplot as plt
from part1_fat import FatPart1


class SegFat:
    def __init__(self):
        # ========  读part1  ========
        path = './p1'
        # self.target = self.nrrd2np(path + '/target_part1.nrrd')
        # self.bone = self.nrrd2np(path + '/bone_part1.nrrd')
        self.fat = self.nrrd2np(path + '/fat_part1.nrrd')
        # self.pieces = len(self.target[0, 0, :])
        # print('切片数：piece = ', self.pieces)

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


s = SegFat()
data = s.nrrd2np('./part1.nrrd')
pic = np.zeros([512, 512, 113], dtype=float)
for p in range(113):
    pic[:, :, p] = data[:, :, p]
nrrd.write('./part1_1.nrrd', pic)