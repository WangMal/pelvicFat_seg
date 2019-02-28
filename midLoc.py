import nrrd
import numpy as np
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


path = './p1'
bone, pieces = nrrd2np(path + '/target_bone.nrrd')
midLoc = int(np.mean(np.where(bone == 1)[0]))

print(midLoc)
