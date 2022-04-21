/**
 * @file      problem2-2-3.cpp
 * @author    zhangmin(3273749257@qq.com)
 * @brief     霍夫直线检测
 * @version   0.4
 * @date      2022-04-21
 * @copyright Copyright (c) 2022
 */



#include<iostream>
#include<opencv2/opencv.hpp>
#include<cmath>

using namespace std;
using namespace cv;



/**
* 函数名：     color2hsv
* 参数：       const Mat& color_img，需要转换为HSV图的原图像
* 返回值：     const Mat hsv_img，转换后的灰度图
* 函数功能：   将RGB彩色图转换为HSV图
*/
const Mat color2gray(const Mat& color_img)
{
    int size_h=color_img.rows;//获取像素长
    int size_w=color_img.cols;//获取像素宽

    //1.彩色图片通道分离
    vector<Mat> channels;
    split(color_img, channels);//B,G,R 11 59  30


    //2.构造一个单通道图
    Mat gray_img(size_h,size_w,CV_8UC1);

    //3.遍历每一个像素点
    for(int i=0;i<size_h;i++)
        for(int j=0;j<size_w;j++)
            //将rgb对应亮度值转化为灰度图对应亮度值
            gray_img.at<uchar>(i,j)=(uchar)((channels[0].at<uchar>(i,j)*11+channels[1].at<uchar>(i,j)*59+channels[2].at<uchar>(i,j)*30)/100);
    return gray_img;
}


//创建一个类进行高斯滤波
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

/************************对图像进行边缘检测*************************/
/**
 * 函数名：  edgeDetection
 * 参数：   Mat src，需要进行边缘检测的图像
 * 返回值： Mat dst，边缘检测后的图像
 * 功能：   对图像进行边缘检测
 * 补充：   对单通道图像
 * ***************************************
 * 思路：
 * 1.主要准备工作
 *  (1)获取图像长、宽
 *  (2)创建一个图像，存储填充补零后的图像
 *  (3)创建一个图像，存储边缘处理后的图像
 *  (4)sobel算子，x方向、y方向
 *  (5)gx数组，gy数组，g
 *
 * 2.遍历每一个像素点,进行卷积处理（sobel算子）
 *  (1)
 *  (2)
 *  (3)
 *
 * 3.因为图像边缘不可能为边缘，故进行边缘置0
 *
 *
 * 3.返回边缘检测后的图像
 */

Mat edgeDetection(Mat src)
{
    // [1]
    // [1-1]
    int img_h=src.rows;
    int img_w=src.cols;
    // [1-2] 边界填充src，存储到Newsrc中
    Mat Newsrc;
    copyMakeBorder(src, Newsrc, 1, 1, 1, 1, cv::BORDER_REPLICATE);//边界复制一层
    // [1-3]
    Mat dst(img_h,img_w,CV_8UC1);
    // [1-4]
    int sobel_x[][3]={{-1,0,1},{-2,0,2},{-1,0,1}};
    int sobel_y[][3]={{-1,-2,-1},{0,0,0},{1,2,1}};
    // [1-5] gx、gy、g
    float gx[img_h][img_w],gy[img_h][img_w],g;

    // [2] 卷积处理(sobel算子)
    // [2-1] 遍历每一个像素点
    for(int i=1;i<img_h+1;i++)
        for(int j=1;j<img_w+1;j++)
        {
            for (int r = -1; r <= 1; ++r)
            {
                for (int c = -1; c <= 1; ++c)
                {
                     gx[i-1][j-1] += Newsrc.at<uchar>(i + r, j + c) * sobel_x[r+1][c+1];
                     gy[i-1][j-1] += Newsrc.at<uchar>(i + r, j + c) * sobel_y[r+1][c+1];
                }
                g=sqrt(gx[i-1][j-1]*gx[i-1][j-1]+gy[i-1][j-1]*gy[i-1][j-1]);

                if(g>255)
                    g=255;
                else if(g<0)
                    g=0;

                dst.at<uchar>(i-1,j-1)=g;
            }
        }


    // [3] 边缘置0
    for(int i=0;i<img_h;i++)
    {
        dst.at<uchar>(i,0)=0;
        dst.at<uchar>(i,img_w-1)=0;
    }
    for(int j=0;j<img_w;j++)
    {
        dst.at<uchar>(0,j)=0;
        dst.at<uchar>(img_h-1,j)=0;
    }

    // [4] 非极大值抑制
    // [4-1] 准备工作，变量定义
    uint8_t c,g1,g2,g3,g4;
    float weight,dTemp1,dTemp2;
    for(int i=1;i<img_h-1;i++)
    {
        for(int j=1;j<img_w-1;j++)
        {
            c=dst.at<uchar>(i,j);
            // [4-1] 判断gx、gy绝对值大小
            if(fabs(gx[i][j])<fabs(gy[i][j]))
            {
                weight= gx[i][j]/gy[i][j];
                g2=dst.at<uchar>(i-1,j);
                g4=dst.at<uchar>(i+1,j);

                //[4-2] 判断gx、gy是否同向
                if(gx[i][j]*gy[i][j]>0)
                {
                    g1=dst.at<uchar>(i-1,j-1);
                    g3=dst.at<uchar>(i+1,j+1);
                }

                else
                {
                    g1=dst.at<uchar>(i-1,j+1);
                    g3=dst.at<uchar>(i+1,j-1);
                }

                dTemp1 = weight * g1 + (1 - weight) * g2;
                dTemp2 = weight * g3 + (1 - weight) * g4;
            }

            else
            {
                weight= gy[i][j]/gx[i][j];
                g2=dst.at<uchar>(i,j-1);
                g4=dst.at<uchar>(i,j+1);

                //[4-2] 判断gx、gy是否同向
                if(gx[i][j]*gy[i][j]>0)
                {
                    g1=dst.at<uchar>(i+1,j-1);
                    g3=dst.at<uchar>(i-1,j+1);
                }

                else
                {
                    g1=dst.at<uchar>(i-1,j-1);
                    g3=dst.at<uchar>(i+1,j+1);
                }

                dTemp1 = weight * g1 + (1 - weight) * g2;
                dTemp2 = weight * g3 + (1 - weight) * g4;
            }

            if(c>=dTemp1&&c>=dTemp2);
            else
                dst.at<uchar>(i,j)=0;

        }
    }

    // [5] 边缘连接：使用双阈值
    int comp_max=200;
    int comp_min=50;
    // [5-1] 遍历像素点
    for(int i=1;i<img_h+1;i++)
        for(int j=1;j<img_w+1;j++)
        {
            // [5-2] 判断
            // [5-2-1] 大于comp_max
            if(dst.at<uchar>(i,j)>=comp_max);
            // [5-2-2] 小于comp_min
            else if(dst.at<uchar>(i,j))
                dst.at<uchar>(i,j)=0;
            // [5-2-3] 介于comp_min与comp_max之间
            else
            {

                // [5-3] 判断8邻域
                int flag=0;//设置一个标志位，来标志8邻域内是否存在边缘点，标志位为0不存在边缘点
                for(int k=-1;k<=1;k++)
                    for(int l=-1;l<=1;l++)
                        if(dst.at<uchar>(i+k,j+l)>=comp_max);
                            flag=1;
                 if(flag==0)
                 {
                     flag=0;//标志位清零
                     dst.at<uchar>(i,j)=0;
                 }
            }
        }


    return dst;
}



/**
* 函数名：     lineDetection
* 参数：       Mat src，绘画直线的图像
*             Mat edge_img，边缘检测后的图像
*             int voteThresh，是经过某一点曲线的数量的阈值，超过这个阈值，就表示这个交点所代表的参数对(rho, theta)在原图像中为一条直线
*             int stepRho = 1，距离步长，默认为1
*             int stepTheta =1，角度步长，默认为1
* 返回值：     无
* 函数功能：   对边缘图像进行直线检测
* ************************************************************
* 思路：
* 1.准备工作
*   (1)获取像素长，像素宽，图像对角线长
*   (2)霍夫变换后角度和距离的个数
*   (3)投票器
*
* 2.投票
*
* 3.阈值
*
* 4.非极大值抑制
*
* 5.输出直线
*/

void lineDetection(Mat src,Mat edge_img,int voteThresh,int stepRho = 1,int stepTheta =1)
{
    // [1] 准备工作
    // [1-1] 获取像素长，像素宽，图像对角线长
    int img_h=edge_img.rows;
    int img_w=edge_img.cols;
    int L=sqrt(img_h*img_h+img_w*img_w);
    // [1-2]
    int numRho=int(180.0 / stepTheta);
    int numTheta=int(2 * L / stepRho + 1);
    // [1-3] 投票器
    int accumulator[numRho][numTheta];
    // [1-4] 直线序列
    vector<vector<Point>> line_point(numRho*numTheta, vector<Point>(img_h*img_w));
 //   int line_total=0;//直线总数
    int pointnum=0;



    // [2] 投票
    // [2-1] 遍历除图像边界点以外的像素点
    for(int y=1;y<img_h-1;y++)
        for(int x=1;x<img_w-1;x++)
        {

            // [2-2] 遍历所有角点
            if(edge_img.at<uchar>(x,y)!=0)
            {
                for(int m=0;m<numTheta;m++)
                {
                    // [2-3] 对每一个角度，代入x、y，以及所有角点 计算对应的 rho 值
                    float rho = x *cos(stepTheta * m / 180.0 * CV_PI) + y * sin(stepTheta * m / 180.0 * CV_PI);
                    // [2-4] 计算投票哪一个区域
                    int n = int(round(rho + L) / stepRho);
                    // [2-5]投票加 1
                    accumulator[n][m]++;
                  int line_pos=m*(n+1);//m*n+m是对应位置，方便将line_pos解析W为一个m、n
                    line_point[line_pos][pointnum].x=x;//line_point[n][m],对应原空间的一条直线
                    line_point[line_pos][pointnum].y=y;

                }
                pointnum++;
            }
        }



    // [3] 阈值处理

    for(int i=1;i<numRho-1;i++)
        for(int j=1;j<numTheta-1;j++)
        {

            if(accumulator[i][j]>voteThresh&&accumulator[i][j]>=accumulator[i-1][j-1]&&accumulator[i][j]>=\
                    accumulator[i-1][j+1]&&accumulator[i][j]>=accumulator[i+1][j-1]&&accumulator[i][j]>=accumulator[i+1][j+1])
            {

                int pos=j*(i+1);


                line(src,line_point[pos][0],line_point[pos][pointnum-1],(0,0,255),2);

            }


        }



    imshow("line detection",src);
}



int main()
{
    // 1.读取图像
    Mat img=imread("D:/Picture/hough_line_detection.png");
    if(img.empty())//未获取到图像
    {
        cout << "can't read this image!" << endl;
        return 0;
    }

    // 2.实现高斯滤波
    myGaussianFilter test(img,3,3,sqrt(2));
    Mat gaus_img=test.GaussianFilter();
    // 3.灰度图
    Mat gray_img=color2gray(gaus_img);
    // 4.边缘检测
    Mat edge_img=edgeDetection(gray_img);

    lineDetection(img,edge_img,60);

    // 5.显示图像
    imshow("src",img);
    imshow("GaussianFilter",gaus_img);
    imshow("Gray",gray_img);
    imshow("Edge Detection",edge_img);

    // 6.霍夫直线检测
    lineDetection(img,edge_img,60);

    waitKey(0);


    return 0;
}
