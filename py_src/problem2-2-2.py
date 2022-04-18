# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/4/18 20:48
# @Author  : zhangmin
# @File    : problem2-2-2.py
# @Brief   :
# @Version : 0.1

import cv2
import numpy as np

'''
函数名：    gaussian_filter
参数：     image          原图像
          K_size         高斯核尺寸，size*size
          sigma          高斯核方差
返回值：    gaussian_img   高斯滤波后的图像
功能：     对图像进行高斯滤波
**************************************
思路：
1.准备工作：获取图像长，宽，通道数
2.图像填充，补零操作
3.高斯核计算
4.进行高斯滤波
  4-1.注意对矩阵元素值的限定
  4-2.将矩阵元素类型转换为uint8
5.返回高斯滤波后的图像
'''


def gaussian_filter(img, K_size=3, sigma=2):
    # [1]获取图像长，宽，通道数
    H, W, C = img.shape

    # [2]图像填充，补零操作
    # [2-1]计算需要添加的层数，同时也方便后面的卷积操作
    center = K_size // 2
    # [2-2]创建一个三维矩阵，存放高斯滤波后的图像
    gaussian_img = np.zeros((H + center * 2, W + center * 2, C), dtype=np.float)
    # [2-3]将原图像copy到该矩阵中
    gaussian_img[center: center + H, center: center + W] = img.copy().astype(np.float)

    # [3]高斯核计算
    # [3-1]初始化高斯核，大小为size*size的二维矩阵
    gaussian_kernel = np.zeros((K_size, K_size), dtype=np.float)
    # [3-2]将坐标代入，初步计算出高斯核
    for x in range(-center, center):
        for y in range(-center, center):
            gaussian_kernel[y + center, x + center] = np.exp(-(x ** 2 + y ** 2) / (2 * (sigma ** 2)))
    gaussian_kernel /= (2 * np.pi * sigma * sigma)
    # [3-3]归一化
    gaussian_kernel /= gaussian_kernel.sum()

    # [4]进行高斯滤波
    # [4-1]将gaussian_img复制一份，方便卷积操作
    tmp = gaussian_img.copy()
    # [4-2]遍历所有像素点(三通道)，卷积操作
    for x in range(H):
        for y in range(W):
            for c in range(C):
                gaussian_img[center + x, center + y, c] = np.sum(gaussian_kernel * tmp[x: x + K_size, y: y + K_size, c])
    # [4-3]对矩阵的元素值进行范围限定，0~255
    gaussian_img = np.clip(gaussian_img, 0, 255)
    # [4-4]将矩阵元素类型转换为uint8
    gaussian_img = gaussian_img[center: center + H, center: center + W].astype(np.uint8)

    # [5]返回高斯滤波后的图像
    return gaussian_img


if __name__ == '__main__':
    # 1.读取图像
    img = cv2.imread("D:/Picture/lena.jpg")

    # 2.高斯滤波
    gaussian_img = gaussian_filter(img, K_size=3, sigma=2)

    # 3.展示图像
    cv2.imshow("RGB", img)
    cv2.imshow("Gaussian", gaussian_img)

    # 4.等待按键操作
    cv2.waitKey(0)
