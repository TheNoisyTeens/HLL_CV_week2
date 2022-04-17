/**
 * @file      problem2_2_2.cpp
 * @author    zhangmin(3273749257@qq.com)*
 * @brief     对图像进行高斯滤波
 * @version   0.1
 * @date      2022-04-17
 * @copyright Copyright (c) 2022
 */

/**
*不足:
* 边缘没有进行卷积操作
*/

#include<iostream>
#include<opencv2/opencv.hpp>
#include<cmath>

using namespace std;
using namespace cv;

//pi宏定义
#define pi 3.141592653589793



//创建一个类进行高斯滤波
/**********************class myGaussianFilter********************************/
/**
* 类名：myGaussianFilter
* 类成员属性：Mat img，要进行高斯滤波的图像
*           double **gaussianArray，高斯核
*           int gaussianSize，高斯核大小，size*size
*           double gaussianSigma，高斯核方差
* 类成员函数：
*           myGaussianFilter(Mat& _img,int _size,double _sigma)，构造函数，初始化类成员属性
*           void getgaussianArray()，获取高斯核
*           Mat& gaussianMerge()，对图像进行高斯滤波
*           void GaussianFilter(Mat& image)，对单通道高斯滤波
*           ~myGaussianFilter()，析构函数，释放内存
*/
class myGaussianFilter
{
private:
    Mat img;//待处理的图像
    double **gaussianArray;//高斯核
    int gaussianSize;//高斯核大小size*size
    double gaussianSigma;//高斯核方差

public:
    //构造函数
    myGaussianFilter(Mat& _img,int _size,double _sigma):gaussianSize(_size),gaussianSigma(_sigma)
    {
        //图像
        img=_img;

        //构建高斯二维数组
        gaussianArray= new double*[gaussianSize];
        for (int i = 0; i < gaussianSize; i++)
        {
               gaussianArray[i] = new double[gaussianSize];
        }

        //初始化高斯数组
        for (int i = 0; i < gaussianSize; i++ )
        {
               for (int j = 0; j < gaussianSize; j++)
               {
                   gaussianArray[i][j] = 0;
               }
        }
    }

    void getgaussianArray();
    Mat& gaussianMerge();

    //析构函数
    ~myGaussianFilter()
    {
        if(gaussianArray)
            delete []gaussianArray;
        for (int i = 0; i < gaussianSize; i++)
        {
            if(gaussianArray[i] )
                delete []gaussianArray[i];
        }
    }


private:
    void GaussianFilter(Mat& image);
};


/************************获取高斯核*************************/
/**
 * 函数名：  getgaussianArray
 * 参数：   无
 * 返回值： 无
 * 功能：   获取高斯核
 * ***************************************
 */
void myGaussianFilter::getgaussianArray()
{
    //获取中心坐标值
    int center_i, center_j;
    center_i = center_j = gaussianSize / 2;

    double gaussianSum=0.0f;//存储初步得出的高斯核的和

    //初步计算高斯核
    for (int i = 0; i < gaussianSize; i++ )
    {
           for (int j = 0; j < gaussianSize; j++)
           {
               //中心坐标(0,0)，将坐标依次代入二维高斯函数，得到高斯核
               gaussianArray[i][j] = exp( -(1.0f)* ( ((i-center_i)*(i-center_i)+(j-center_j)*(j-center_j)) /
                                                     (2.0f*gaussianSigma*gaussianSigma) ));
               gaussianSum += gaussianArray[i][j];
           }
    }
    //归一化处理
    for (int i = 0; i < gaussianSize; i++)
    {
           for (int j = 0; j < gaussianSize; j++)
           {
               gaussianArray[i][j] /= gaussianSum;
           }
    }
}


/************************对单通道高斯滤波*************************/
/**
 * 函数名：  GaussianFilter
 * 参数：   Mat& image，需要进行高斯滤波的单通道图像
 * 返回值： 无
 * 功能：   对单通道高斯滤波
 * ***************************************
 * 思路：
 * 创建一个临时Mat对象存放卷积操作后的数据
 * 遍历除边缘外的每一个像素点，进行卷积操作
 */
void myGaussianFilter::GaussianFilter(Mat& image)
{

    int img_h=image.rows;//获取像素长
    int img_w=image.cols;//获取像素宽

    //1.将传入的单通道image拷贝一份
    Mat temp=image.clone();

    //2.遍历每一个像素点
    for (int i = 0; i < img_h; i++ )
    {
           for (int j = 0; j < img_w; j++)
           {
               //边缘除外
               if (i > (gaussianSize / 2) - 1 && j > (gaussianSize / 2) - 1 &&
                              i <img_h - (gaussianSize / 2) && j < img_w - (gaussianSize / 2))
               {
                   //进行卷积操作
                   double sum=0;
                   for (int k = 0; k < gaussianSize; k++ )
                   {
                          for (int l = 0; l < gaussianSize; l++)
                          {
                              sum += image.ptr<uchar>(i-k+(gaussianSize/2))[j-l+(gaussianSize/2)] * gaussianArray[k][l];
                          }
                   }

                    //将求出的sum放入临时Mat对象中
                    temp.ptr<uchar>(i)[j] = sum;
             }
    }
    //3.将处理好的单通道temp克隆给image
    image = temp.clone();
  }

}

/************************对图像进行高斯滤波*************************/
/**
 * 函数名：  gaussianMerge
 * 参数：   无
 * 返回值： Mat& final_img，高斯滤波后的图像
 * 功能：   对图像进行高斯滤波
 * ***************************************
 * 思路：
 * 1.将RGB彩色图通道分离
 * 2.将分离出的三个通道分别进行高斯滤波
 * 3.将处理后的三个通道合并
 */
Mat& myGaussianFilter::gaussianMerge()
{

    Mat final_img;
    //1.图片通道分离
    vector<Mat> channels;
    split(img, channels);

    //2.高斯滤波处理
     for (int i = 0; i < 3; i++)
     {
          GaussianFilter(channels[i]);
      }
     //3.合并返回
     merge(channels, final_img);
     return final_img;

}



int main()
{
    Mat img=imread("D:/Picture/lena.jpg");
    if(img.empty())//未获取到图像
    {
        cout << "can't read this image!" << endl;
        return 0;
    }

    myGaussianFilter test(img,3,2);
    test.getgaussianArray();
    imshow("GaussianFilter",test.gaussianMerge());
    cout<<"test"<<endl;
    waitKey(0);
    return 0;
}
