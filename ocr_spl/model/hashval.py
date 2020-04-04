# encoding=utf8

import cv2 as cv
import numpy as np


SCALE_SIZE = (16, 16)


class ImageHashCal(object):
    def __init__(self):
        self.scale_size = SCALE_SIZE
        self.hash_func = {'dHash': self.dHash,
                          'aHash': self.aHash,
                          'pHash': self.pHash,
                          'hHash': self.hHash,
                          'd2Hash': self.d2Hash,
                          }

    def dHash(self, img):
        import pdb; pdb.set_trace()
        # 差值感知算法
        img = cv.resize(img, self.scale_size, interpolation=cv.INTER_CUBIC)
        #gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        gray = img
        hash_str = ''

        # 每行前一个像素大于后一个像素为1，相反为0，生成哈希
        for i in range(img.shape[0]):
            for j in range(img.shape[1]-1):
                if gray[i, j] > gray[i, j + 1]:
                    hash_str = hash_str + '1'
                else:
                    hash_str = hash_str + '0'
        hashval = int(hash_str, 2)
        return hashval

    def aHash(self, img):
        # 均值哈希算法
        img = cv.resize(img, self.scale_size, interpolation=cv.INTER_CUBIC)
        #gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        gray = img
        s = 0
        hash_str = ''
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                s = s + gray[i, j]
        avg = s / 64

        # 灰度大于平均值为1相反为0生成图片的hash值
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                if gray[i, j] > avg:
                    hash_str = hash_str + '1'
                else:
                    hash_str = hash_str + '0'

        # using numpy

        hashval = int(hash_str, 2)
        return hashval

    def pHash(self, img):
        # 加载并调整图片为32x32灰度图片
        img = cv.resize(img, (64, 64), interpolation=cv.INTER_CUBIC)

        h, w = img.shape[:2]
        vis0 = np.zeros((h, w), np.float32)
        vis0[:h, :w] = img

        # 二维DCT变换
        vis1 = cv.dct(cv.dct(vis0))
        # cv.SaveImage('/tmp/a.jpg', cv.fromarray(vis1)) #保存图片

        vis1.resize(32, 32)
        # 把二维list变成一维list
        img_list = vis1.flatten()

        # 计算均值
        avg = sum(img_list) * 1. / len(img_list)
        avg_list = ['0' if i < avg else '1' for i in img_list]

        # 得到哈希值
        hash_str = ''.join(['%x' % int(''.join(avg_list[x:x + 4]), 2) for x in range(0, 32 * 32, 4)])

        hashval = int(hash_str, 16)
        return hashval

    def hHash(self, img):
        # 直方图hash
        img = cv.resize(img, self.scale_size, interpolation=cv.INTER_CUBIC)
        hist = cv.calcHist([img], [0], None, [64], [0.0, 255.0])
        hashval = (hist>hist.mean()).transpose()[0].astype(int).astype(str)
        hashval = ''.join(hashval.tolist())
        hashval = int(hashval, 2)
        return hashval

    @staticmethod
    def hanmming_distance(x, y):
        return bin(x ^ y).count('1')


    def d2Hash(self, img):
        """尝试改良版 difference hash"""
        # 差值感知算法
        img = cv.resize(img, self.scale_size, interpolation=cv.INTER_CUBIC)
        #gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        gray = img
        hash_str = ''

        # 每行前一个像素大于后一个像素为1，相反为0，生成哈希
        for i in range(img.shape[0]):
            for j in range(img.shape[1]-1):
                if gray[i, j] > (gray[i, j + 1] + 50):
                    hash_str = hash_str + '1'
                else:
                    hash_str = hash_str + '0'

        for i in range(img.shape[0]):
            for j in range(img.shape[1]-1):
                if (gray[i, j] + 50) < gray[i, j + 1]:
                    hash_str = hash_str + '1'
                else:
                    hash_str = hash_str + '0'

        for i in range(img.shape[1]):
            for j in range(img.shape[0]-1):
                if gray[j, i] > (gray[j + 1, i] + 40):
                    hash_str = hash_str + '1'
                else:
                    hash_str = hash_str + '0'

        for i in range(img.shape[1]):
            for j in range(img.shape[0]-1):
                if (gray[j, i] + 40) < gray[j + 1, i]:
                    hash_str = hash_str + '1'
                else:
                    hash_str = hash_str + '0'

        hashval = int(hash_str, 2)
        return hashval
