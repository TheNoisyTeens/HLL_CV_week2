# -*- coding: utf-8 -*-
# @Time     : 2022-04-17
# @Author   : zhangmin
# @File     : problem2_2_1.py
# @Brief    : 将RGB彩色图转换为灰度图、HSV图
# @version  : 0.1

import numpy as np
import cv2

# *******************************将RGB彩色图转换为灰度图****************************************************
'''
函数名：  color2gray
参数：    color_img，读取到的RGB彩色图
返回值：  gray_img，最终得出的灰度图
功能：    将RGB彩色图转换为灰度图
*********************************************
思路：
    1.将图像三维矩阵“分离”为3个二维矩阵R、G、B
    2.二维矩阵每个元素储存的是该通道相应位置的亮度值
    3.套公式求出灰度图中的亮度值
'''


def color2gray(color_img):
    size_h = color_img.shape[0]  # 获取图像宽像素
    size_w = color_img.shape[1]  # 获取图像长像素

    B = color_img[:, :, 0]  # 获取R通道--二维矩阵
    G = color_img[:, :, 1]  # 获取G通道--二维矩阵
    R = color_img[:, :, 2]  # 获取B通道--二维矩阵

    # 创建一个二维矩阵存储灰度图的亮度值
    gray_img = np.zeros((size_h, size_w), dtype=np.uint8)

    # 遍历所有像素点，套公式将R、G、B三通道的亮度值转换为灰度图的亮度值
    for i in range(size_h):
        for j in range(size_w):
            gray_img[i, j] = round((R[i, j] * 30 + G[i, j] * 59 + \
                                    B[i, j] * 11) / 100)

    return gray_img  # 返回灰度图


# *****************************************************************************************************


# **********************************将RGB彩色图转换为HSV图*************************************************
'''
函数名：  color2hsv
参数：    color_img，读取到的RGB彩色图
返回值：  hsv_img，最终得出的HSV图
功能：    将RGB彩色图转换为HSV图
*********************************************
思路：
    1.通道分离
    2.创建三个二维矩阵存储H、S、V三通道的值
    3.遍历像素点，套公式求出HSV图中的三个通道对应的值
'''


def color2hsv(img):
    h = img.shape[0]  # 获取图像宽像素
    w = img.shape[1]  # 获取图像长像素

    # 1.通道分离
    r, g, b = cv2.split(img)
    r, g, b = r / 255.0, g / 255.0, b / 255.0

    # 2.创建三个二维矩阵存储H、S、V三通道的值
    H = np.zeros((h, w), np.float32)
    S = np.zeros((h, w), np.float32)
    V = np.zeros((h, w), np.float32)

    # 3.遍历像素点，进行通道计算
    for i in range(0, h):
        for j in range(0, w):
            mx = max((b[i, j], g[i, j], r[i, j]))  # r,g,b三通道最大值
            mn = min((b[i, j], g[i, j], r[i, j]))  # r,g,b三通道最大值

            # V 通道值
            V[i, j] = mx

            # S 通道值
            if V[i, j] == 0:
                S[i, j] = 0
            else:
                S[i, j] = (V[i, j] - mn) / V[i, j]

            # H 通道值
            if mx == mn:
                H[i, j] = 0
            elif V[i, j] == r[i, j]:
                if g[i, j] >= b[i, j]:
                    H[i, j] = (60 * ((g[i, j]) - b[i, j]) / (V[i, j] - mn))
                else:
                    H[i, j] = (60 * ((g[i, j]) - b[i, j]) / (V[i, j] - mn)) + 360
            elif V[i, j] == g[i, j]:
                H[i, j] = 60 * ((b[i, j]) - r[i, j]) / (V[i, j] - mn) + 120
            elif V[i, j] == b[i, j]:
                H[i, j] = 60 * ((r[i, j]) - g[i, j]) / (V[i, j] - mn) + 240
            H[i, j] = H[i, j] / 2

    return H, S, V  # 返回H，S，V三通道


# ******************************************************************************************************


if __name__ == '__main__':
    # 1.读取图像
    color_img = cv2.imread("./lena.jpg")
    # D:/Picture/y.jpg
    # 2.调用color2gray函数，将图像转换为灰度图
    gray_img = color2gray(color_img)

    # 3.调用color2hsv函数，将图像转换为HSV图
    h, s, v = color2hsv(color_img)
    hsv_img = cv2.merge([h, s, v])  # 前面分离出来的三个通道

    # 4.显示原图、灰度图、HSV图
    cv2.imshow("color image", color_img)
    cv2.imshow("gray image", gray_img)
    cv2.imshow("hsv image", hsv_img)

    cv2.waitKey(0)

'''
# 原代码--rgb2hsv
def color2hsv(color_img):
    size_h = color_img.shape[0]  # 获取图像宽像素
    size_w = color_img.shape[1]  # 获取图像长像素

    B = color_img[:, :, 0]  # 获取R通道--二维矩阵
    G = color_img[:, :, 1]  # 获取G通道--二维矩阵
    R = color_img[:, :, 2]  # 获取B通道--二维矩阵

    # 创建三个二维矩阵存储H、S、V三通道的值
    H = np.zeros((size_h, size_w), dtype=np.uint8)
    S = np.zeros((size_h, size_w), dtype=np.uint8)
    V = np.zeros((size_h, size_w), dtype=np.uint8)

    # H通道计算
    for i in range(size_h):
        for j in range(size_w):
            rgb_max = max(R[i, j], R[i, j], B[i, j])
            rgb_min = min(R[i, j], R[i, j], B[i, j])

            # V 通道值
            V[i, j] = rgb_max

            # S 通道值
            if max == 0:
                S[i, j] = 0
            else:
                S[i, j] = 1 - rgb_min / rgb_max

            # H 通道值
            if rgb_max == rgb_min:
                H[i, j] = 0
            elif V[i, j] == R[i, j]:
                if G[i, j] >= B[i, j]:
                    H[i, j] = (60 * ((G[i, j]) - B[i, j]) / (V[i, j] - rgb_min))
                else:
                    H[i, j] = (60 * ((G[i, j]) - B[i, j]) / (V[i, j] - rgb_min)) + 360
            elif V[i, j] == G[i, j]:
                H[i, j] = 60 * ((B[i, j]) - R[i, j]) / (V[i, j] - rgb_min) + 120
            elif V[i, j] == B[i, j]:
                H[i, j] = 60 * ((R[i, j]) - G[i, j]) / (V[i, j] - rgb_min) + 240
            H[i, j] = H[i, j] / 2

            # 将H、S、V二维矩阵转换为三维矩阵
            H_t = H.reshape((size_h, size_w, 1))
            S_t = S.reshape((size_h, size_w, 1))
            V_t = V.reshape((size_h, size_w, 1))

            # 合并三个矩阵
            hsv_img = np.concatenate((H_t, S_t, V_t), 2)

            return hsv_img #返回HSV图


   # main中调用
   hsv_img = color2hsv(color_img)
   cv2.imshow("hsv image", hsv_img)
'''
