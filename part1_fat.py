# -*- coding: UTF-8 -*-

import numpy as np


class FatPart1:

    # def __init__(self, x, y, bone_p, edge_p_coordinate,fat_p_coordinate):
    def __init__(self, x, y, bone_p, edge_p_coordinate):
        self.x = x
        self.y = y
        # [self.leftBone, self.rightBone, self.upBone, self.downBone] = self.__is_bone__(bone_p)
        # [self.dis2Edge, self.nearestDot2Edge] = self.__dot2edge__(edge_p_coordinate)
        # self.neardot = self.neardot(self.x, self.y, self.nearestDot2Edge)  # 寻找到fat和边界点连线距离小于d的点
        # self.nearFatRate = self.nearfatrate(self.neardot, fat_p_coordinate)  # 最短距离点连接线ε近邻带脂肪
        # self.isPelvicFat

    def __is_bone__(self, bone):
        left = 0
        right = 0
        up = 0
        down = 0

        if 1 in bone[0:self.x, self.y]:
            left = 1
        if 1 in bone[self.x:-1, self.y]:
            right = 1
        if 1 in bone[self.x, self.y:-1]:
            up = 1
        if 1 in bone[self.x, 0:self.y]:
            down = 1
        return left, right, up, down

    def __dot2edge__(self, edge):
        dis = []
        for e in edge:
            dis.append(np.sqrt((self.x-e[0])**2) + np.sqrt((self.y-e[1])**2))
        mindis = min(dis)
        minloc = np.where(dis == mindis)[0]
        return mindis, edge[minloc[0]]

    def change2dotset(self, pic):
        dataset = []
        for i in range(512):
            for j in range(512):
                if pic[i, j] == 1:
                    dataset.append([i, j])
        return dataset

    def neardot(self, x, y, dot_edge):
        # 遍历min[x, dot_edge[0]]-1,max[x, dot_edge[0]]+1
        # min[y, dot_edge[[1]]] - 1, max[x, dot_edge[1]] + 1
        # 寻找到fat和边界点连线距离小于d的点
        min_x = min(x, dot_edge[0])
        max_x = max(x, dot_edge[0])
        min_y = min(y, dot_edge[1])
        max_y = max(y, dot_edge[1])
        near_dot = []
        for x0 in range(min_x-1, max_x+1):
            for y0 in range(min_y - 1, max_y + 1):
                dis = self.disdot2line([x, y], dot_edge, [x0, y0])
                if dis < 1:
                    near_dot.append([x0, y0])
        return near_dot

    def nearfatrate(self, near_dot, fat_p):
        count_all = len(near_dot)
        count_fat = 0
        for d in near_dot:
            if d in fat_p:
                count_fat += 1
        fat_rate = count_fat/count_all
        return fat_rate

    def disdot2line(self, dot1_online, dot2_online, dot):
        # 计算点dot到直线的距离，dot1_online,dot2_online 是线上的点
        lineVector = np.array([dot1_online[0] - dot2_online[0], dot1_online[1] - dot2_online[-1]])
        v = np.array([dot[0] - dot1_online[0], dot[1] - dot1_online[1]])
        h = np.linalg.norm(np.cross(lineVector, v) / np.linalg.norm(v))
        return h







