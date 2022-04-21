# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/4/21 21:55
# @Author  : zhangmin
# @File    : prob2-2-3.py
# @Brief   : 霍夫直线检测
# @Version : 0.4

# -*- coding: utf-8 -*-
import numpy as np
import cv2
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

'''
函数名：  color2gray
参数：    color_img，读取到的RGB彩色图
返回值：  gray_img，最终得出的灰度图
功能：    将RGB彩色图转换为灰度图
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


'''
函数名：    gaussian_filter
参数：     image          原图像
          K_size         高斯核尺寸，size*size
          sigma          高斯核方差
返回值：    gaussian_img   高斯滤波后的图像
功能：     对图像进行高斯滤波
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


'''
函数名：    edgeDetect
参数：     src          原图像(单通道)   
          bmin=100     双阈值进行边界点判断 
          bmax=220     双阈值进行边界点判断
返回值：    dst         边缘检测后的图像(单通道)
功能：     对图像进行边缘检测
**************************************
思路：
    1.准备工作：
      (1)获取图像长，宽
      (2)将原图像copy一份并填充0
      (3)创建一个dst存储边缘检测后的图像
      (4)sobel_x,sobel_y sobel算子
      (5)gx，gy二维矩阵，存储梯度值

    2.遍历像素点，卷积处理
      (1)计算出x方向和y方向的梯度值
      (2)得到幅值
      (3)存入图像矩阵

    3.非极大值抑制
      (1)遍历像素点
      (2)与梯度方向进行比较

    4.边缘连接
      主要使用双阈值，对介于较大值与较小值的点查看其8邻域内是否有边
'''


def edgeDetect(src):
    # [1] 准备工作
    # [1-1] 获取图像长，宽
    img_h = src.shape[0]
    img_w = src.shape[1]
    # [1-2] 将原图像copy一份，并在四周填充一层0
    Newsrc = np.zeros((img_h + 2, img_w + 2), dtype=np.float)
    Newsrc[1:img_h + 1, 1:img_w + 1] = src.copy().astype(np.float)  # 利用切片src.copy().astype(np.float)
    # [1-3] 创建一个dst存储边缘检测后的图像
    dst = src.copy().astype(np.float)
    # [1-4] sobel_x,sobel_y
    sobel_x = [[1, 0, -1], [2, 0, -2], [1, 0, -1]]
    sobel_y = [[1, 2, 1], [0, 0, 0], [-1, -2, -1]]

    # [2] 遍历每一个像素点，进行卷积处理
    for i in range(1, img_h + 1):
        for j in range(1, img_w + 1):
            # [2-1] 算出gx、gy
            gx = np.sum(Newsrc[i - 1:i + 2, j - 1: j + 2] * sobel_x)  # gx
            gy = np.sum(Newsrc[i - 1:i + 2, j - 1: j + 2] * sobel_y)  # gy
            # [2-2] 算出g
            g = abs(gx) + abs(gy)
            # [2-3] 将求出的值存入dst
            dst[i - 1, j - 1] = g
    # [2-4] 注意对矩阵元素值的限定
    dst = np.clip(dst, 0, 255)
    # [2-5] 将矩阵元素类型转换为uint8
    dst = dst.astype(np.uint8)

    # [3] 返回
    return dst


# 函数功能：将src图像值小于comp_val的置为0
def gray2(src, comp_val):
    img_h = src.shape[0]
    img_w = src.shape[1]

    dst = np.zeros((img_h, img_w), dtype=np.uint8)

    for i in range(img_h):
        for j in range(img_w):
            if src[i, j] <= comp_val:
                dst[i, j] = 0
            else:
                dst[i, j] = src[i, j]

    return dst


# 霍夫极坐标变换：直线检测
'''
函数名：    hough_line_detection
参数：     src          原图像,绘制直线的图像   
          edge_img     边缘检测后的图像
          voteThresh   是经过某一点曲线的数量的阈值，超过这个阈值，就表示这个交点所代表的参数对(rho, theta)在原图像中为一条直线
          stepTheta =1 角度步长，默认为1
返回值：   无
功能：     对边缘图像进行直线检测
**************************************
思路：
    1.准备工作：      
    2.投票计数      
    3.阈值处理
    4.画线    
    5.展示图像
'''
def hough_line_detection(image,voteThresh = 60, stepTheta=1, stepRho=1):
    # [1] 准备工作
    # [1-1] 获取图像长，宽
    rows, cols = image.shape
    # [1-2] 图像中可能出现的最大垂线的长度
    L = round(math.sqrt(pow(rows - 1, 2.0) + pow(cols - 1, 2.0))) + 1
    # [1-3] 初始化投票器
    numtheta = int(180.0 / stepTheta)
    numRho = int(2 * L / stepRho + 1)
    accumulator = np.zeros((numRho, numtheta), np.int32)
    # []-4建立字典
    accuDict = {}
    for k1 in range(numRho):
        for k2 in range(numtheta):
            accuDict[(k1, k2)] = []

    # [2] 投票计数
    # [2-1] 遍历像素点
    for y in range(rows):
        for x in range(cols):
            if (image[y][x] ==255):  # 不对值为0的点做霍夫变换
                for m in range(numtheta):
                    # [2-2] 对每一个角度，计算对应的 rho 值
                    rho = x * math.cos(stepTheta * m / 180.0 * math.pi) + y * math.sin(stepTheta * m / 180.0 * math.pi)
                    # [2-3] 计算投票哪一个区域
                    n = int(round(rho + L) / stepRho)
                    # [2-4] 投票加 1
                    accumulator[n, m] += 1
                    # [2-5] 记录该点
                    accuDict[(n, m)].append((x, y))

    # 计数器大小
    rows, cols = accumulator.shape

    # [3] 阈值处理
    for r in range(rows):
        for c in range(cols):
            if accumulator[r][c] > voteThresh:
                points = accuDict[(r, c)]
                # [4] 画线
                cv2.line(img, points[0], points[len(points) - 1], (255), 2)


# 主函数
if __name__ == "__main__":
    # 1.读取图像
    img = cv2.imread("D:/Picture/hough_line_detection.png")

    # 2.高斯滤波
    gaussian_img = gaussian_filter(img, K_size=3, sigma=2)
    # 3.灰度图
    gray_img = color2gray(gaussian_img)
    # 4.边缘检测图
    edge_img = edgeDetect(gray_img)
    # 5.阈值后的图
    bw_img = gray2(edge_img, 30)

    # 6.霍夫直线检测
    hough_line_detection(bw_img, 1, 1)

    # 7.显示原图
    cv2.imshow("image", img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
