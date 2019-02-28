# -*- coding: UTF-8 -*-
import numpy as np


class Piece:
    def __init__(self, org, fat, bone, p):
        self.p = p  # 切片数
        self.data = org[:, :, p]
        self.fat = self.change2dotset(fat[:, :, p])
        self.bone = self.change2dotset(bone[:, :, p])
        self.edge = self.get_edge(self.data)
        print('切片：', p)

    def change2dotset(self, pic):
        dataset = []
        for i in range(512):
            for j in range(512):
                if pic[i, j] == 1:
                    dataset.append([i, j])
        return dataset

    def get_edge(self, pic):
        #  ========  获取边缘点集   ========
        edge_set = []
        for r in range(512):
            row = pic[:, r]
            a = np.where(row > 0)[0]
            if len(a):
                ma = max(a)
                mi = min(a)
                edge_set.append([r, ma])
                edge_set.append([r, mi])
        print('目标边缘点个数：', len(edge_set))
        return edge_set

