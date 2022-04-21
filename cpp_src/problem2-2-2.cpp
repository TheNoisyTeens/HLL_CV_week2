/**
 * @file      problem2-2-2.cpp
 * @author    zhangmin(3273749257@qq.com)
 * @brief     对图像进行高斯滤波
 * @version   0.2
 * @date      2022-04-18
 * @copyright Copyright (c) 2022
 */



#include<iostream>
#include<opencv2/opencv.hpp>
#include<cmath>

using namespace std;
using namespace cv;




//创建一个类进行高斯滤波
/**********************class myGaussianFilter********************************/
/**
* 类名：myGaussianFilter
* 类成员属性：Mat src，要进行高斯滤波的图像
*           int img_h，图像 像素长
*           int img_w，图像 像素宽
*           int gaussianSize_h，高斯核长
*           int gaussianSize_w，高斯核宽
*           double gaussianSigma，高斯核方差
* 类成员函数：
*           myGaussianFilter(Mat& _img,int gaussianSize_h,int gauaasianSize_w,double _sigma)，构造函数，初始化类成员属性
*           void getgaussianArray()，获取高斯核
*           Mat& gaussianMerge()，对图像进行高斯滤波
*/
class myGaussianFilter
{
private:
    /* 图像 */
    Mat src;//待处理的图像
    int img_h;//图像 像素长
    int img_w;//图像 像素长

    /* 高斯核 */
    Mat gaussianKernel;//高斯核
    int gaussianSize_h;//高斯核长
    int gaussianSize_w;//高斯核宽
    double gaussianSigma;//高斯核方差

public:
    //构造函数
    myGaussianFilter(Mat& _img,int h,int w,double _sigma):
        gaussianSigma(_sigma),gaussianSize_h(h),gaussianSize_w(w)
    {
        /* 图像 */
        src=_img;
        //获取图像长和宽
        img_h=src.rows;
        img_w=src.cols;

        /* 初始化高斯核 */
        //(1)获取高斯核的Size
        Size gaussianSize;
        gaussianSize.height=gaussianSize_h;
        gaussianSize.width=gaussianSize_w;
        //(2)创建高斯核
        gaussianKernel.create(gaussianSize,CV_64FC1);
        //(3)初始化高斯核(填充0)
        gaussianKernel=cv::Mat::zeros(gaussianSize,gaussianKernel.type());
    }
    Mat GaussianFilter();

private:
    void getgaussianKernel();
};


/************************获取高斯核*************************/
/**
 * 函数名：  getgaussianArray
 * 参数：   无
 * 返回值： 无
 * 功能：   获取高斯核
 * ***************************************
 * 思路：
 * 1.取gaussianArray[1][1]的位置为中心，坐标（0，0）；
 * 2.坐标为(i-center_i,j-center_j)，用x、y取代，将坐标代入二维高斯函数获取初步的高斯核
 * 3.将高斯核归一化
 */
void myGaussianFilter::getgaussianKernel()
{
    // [1] 准备工作
    // [1-1] 获取中心坐标值
    int center_i= gaussianSize_h / 2;
    int center_j= gaussianSize_w / 2;
    // [1-2] 存储初步得出的高斯核的和
    double gaussianSum=0.0f;
    // [1-3] 存储坐标值
    double x, y;

    // [2] 初步计算高斯核
    // [2-1] 遍历每一个坐标
    for (int i = 0; i < gaussianSize_h; ++i)
    {
        y = pow(i - center_i, 2);
        for (int j = 0; j < gaussianSize_w; ++j)
        {
            // [2-2] 套公式计算每一个坐标的值
            x = pow(j - center_j, 2);
            //因为最后都要归一化的，常数部分可以不计算，也减少了运算量
            double g = exp(-(x + y) / (2 * gaussianSigma*gaussianSigma));
            gaussianKernel.at<double>(i, j) = g;
            gaussianSum += g;//求高斯核的和
        }
    }

    // [3] 归一化
    gaussianKernel = gaussianKernel / gaussianSum;
}



/************************对单通道高斯滤波*************************/
/**
 * 函数名：  GaussianFilter
 * 参数：   无
 * 返回值： Mat dst，高斯模糊后的图像
 * 功能：   对图像进行高斯滤波
 * 补充：   可以是单通道，也可以是三通道
 * ***************************************
 * 思路：
 * 1.主要准备工作
 *  (1)获取高斯核
 *  (2)初始化一个图像，存放高斯滤波后的结果，用到zeros()函数
 *  (3)边界填充src，存储到新图像Newsrc中，用到copyMakeBorder
 *
 * 2.遍历每一个像素点,进行卷积处理
 *  要判断单通道还是三通道
 *  (1)对九宫格进项操作
 *  (1)对求出的和进行范围限定
 *  (2)将求出的和放入图像对应通道中
 *  这里单通道对数值操作，三通道对向量操作
 *
 * 3.返回处理后的图像
 */
Mat myGaussianFilter::GaussianFilter()
{
    // [1] 准备工作
    // [1-1] 获取高斯核
    getgaussianKernel();
    // [1-2] 获取中心坐标值,为高斯滤波做准备
    int center_i = (gaussianKernel.rows - 1) / 2;
    int center_j = (gaussianKernel.cols - 1) / 2;
    // [1-3] 初始化一个图像，存放高斯滤波后的结果
    Mat dst = cv::Mat::zeros(src.size(),src.type());
    // [1-4] 边界填充src，存储到Newsrc中
    Mat Newsrc;
    copyMakeBorder(src, Newsrc, center_i, center_i, center_j, center_j, cv::BORDER_REPLICATE);//边界复制

    // [2] 高斯滤波
    // [2-1] 遍历每一个像素点
    for (int i = center_i; i < src.rows + center_i;++i)
    {
        for (int j = center_j; j < src.cols + center_j; ++j)
        {
           // [2-2] 卷积处理
           double sum[3] = { 0 };//存放卷积核与图像九宫格相乘的和(单通道或三通道)
           // 嵌套两层for循环遍历高斯掩膜
           for (int r = -center_i; r <= center_i; ++r)
           {
               for (int c = -center_j; c <= center_j; ++c)
               {
                   /* 通道数为1 */
                   if (src.channels() == 1)
                       sum[0] = sum[0] + Newsrc.at<uchar>(i + r, j + c) * gaussianKernel.at<double>(r + center_i, c + center_j);

                   /* 通道数为3 */
                   else if (src.channels() == 3)
                   {
                       // 利用Vec3b向量对三通道进行处理
                       Vec3b rgb = Newsrc.at<cv::Vec3b>(i+r,j + c);
                       sum[0] = sum[0] + rgb[0] * gaussianKernel.at<double>(r + center_i, c + center_j);//B
                       sum[1] = sum[1] + rgb[1] * gaussianKernel.at<double>(r + center_i, c + center_j);//G
                       sum[2] = sum[2] + rgb[2] * gaussianKernel.at<double>(r + center_i, c + center_j);//R
                   }
               }
           }

           // [2-3] 对值进行限制0-255
           for (int k = 0; k < src.channels(); ++k)
           {
               if (sum[k] < 0)
                   sum[k] = 0;
               else if (sum[k]>255)
                   sum[k] = 255;
           }


           // [2-4] 将处理完后的值放入对应通道
           /* 通道数为1 */
           if (src.channels() == 1)
               dst.at<uchar>(i - center_i, j - center_j) = static_cast<uchar>(sum[0]);

           /* 通道数为3 */
           else if (src.channels() == 3)
           {
               Vec3b rgb = { static_cast<uchar>(sum[0]), static_cast<uchar>(sum[1]), static_cast<uchar>(sum[2]) };//这里进行了类型转换
               dst.at<Vec3b>(i-center_i, j-center_j) = rgb;
           }
        }
    }

    // [3] 返回高斯模糊后的图像
    return dst;
}




int main()
{
    // 1.读取图像
    Mat img=imread("D:/Picture/lena.jpg");
    if(img.empty())//未获取到图像
    {
        cout << "can't read this image!" << endl;
        return 0;
    }

    // 2.实现高斯滤波
    myGaussianFilter test(img,3,3,sqrt(2));
    Mat temp=test.GaussianFilter();

    // 3.显示图像
    imshow("src",img);
    imshow("GaussianFilter",temp);

    // 4.等待按键操作
    waitKey(0);
    return 0;
}
