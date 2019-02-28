# -*- coding: UTF-8 -*-
import numpy as np
import nrrd
import cv2
import imageio

# 目标切片区域分为两部分
# * 顶部到闭孔开始到耻骨联合开始部分为第一部分；
# * 耻骨联合开始到最后为第二部分；


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
        output_array[:, :, i] = cv2.resize(resize[:, :, i], (512, 512),
                                           interpolation=cv2.INTER_CUBIC)
    return output_array, pieces


def closeholebegin(bone, pieces):
    # 确定闭孔位置
    #   每个切片中骨头最上和最下中间有超过 10 像素的连续空洞，认为是闭孔？
    flag = True
    for p in range(pieces - 1, 0, -1):
        bone_p = bone[:, :, p]
        count = 0
        #     从切片顶部开始 连续没有骨头的行超过10pixel时算闭孔开始
        for t in range(512):  # 有骨头开始
            row = bone_p[:, t]
            if len(set(row)) != 1:
                top = t
                break

        for b in range(510, t, -1):
            row = bone_p[:, b]
            if len(set(row)) != 1:
                bottom = b
                break

        # 连续10行都为0
        if bottom - top > 150:

            for x in range(top, bottom):
                row = bone_p[:, x]
                if len(set(row[10:500])) == 1:
                    count += 1
                else:
                    count = 0
                if count >= 10:
                    flag = False
                    break
            if flag is False:  # 闭孔开始
                break
    close_begin = p
    return close_begin


def symphysis(bone, close_begin):
    # 确定耻骨开始联合位置（上）
    for b in range(1, close_begin):
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
            if right - left > 100:
                break
    symphysis_begin = b
    return symphysis_begin


# 读入盆骨区域数据
path = './p1'
target, pieces = nrrd2np(path + '/target.nrrd')
bone, pieces = nrrd2np(path + '/target_bone.nrrd')
fat, pieces = nrrd2np(path + '/target_fat.nrrd')
midLoc = int(np.mean(np.where(bone == 1)[0]))
print(midLoc)
# close_begin = closeholebegin(bone, pieces)  # 闭孔开始
# symphysis_begin = symphysis(bone, close_begin)  # 耻骨联合开始
symphysis_begin = symphysis(bone, pieces)  # 耻骨联合开始

# symphysis_begin = 39
print("耻骨联合开始切片：", symphysis_begin)

data_part1 = target[:, :, 0:symphysis_begin]
data_part2 = target[:, :, symphysis_begin:pieces]
bone_part1 = bone[:, :, 0:symphysis_begin]
bone_part2 = bone[:, :, symphysis_begin:pieces]
fat_part1 = fat[:, :, 0:symphysis_begin]
fat_part2 = fat[:, :, symphysis_begin:pieces]
#
# for p in range(symphysis_begin):
#     imageio.imwrite('./result/pic/part1/pic%d.jpg' % p,
#                     cv2.resize(data_part1[:, :, p], (512, 512), interpolation=cv2.INTER_CUBIC))
# for p in range(np.shape(data_part2)[2]):
#     piece = p+symphysis_begin
#     imageio.imwrite('./result/pic/part2/pic%d.jpg' % piece,
#                     cv2.resize(data_part2[:, :, p], (512, 512), interpolation=cv2.INTER_CUBIC))

nrrd.write("./result/nrrd1/data_part1.nrrd", data_part1)
nrrd.write("./result/nrrd1/data_part2.nrrd", data_part2)

nrrd.write("./result/nrrd1/bone_part1.nrrd", bone_part1)
nrrd.write("./result/nrrd1/bone_part2.nrrd", bone_part2)

nrrd.write("./result/nrrd1/fat_part1.nrrd", fat_part1)
nrrd.write("./result/nrrd1/fat_part2.nrrd", fat_part2)
