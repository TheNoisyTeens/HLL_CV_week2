/**
 * @file      problem2-2-2.cpp
 * @author    zhangmin(3273749257@qq.com)*
 * @brief     将RGB图像转换为灰度图和HSV图
 * @version   0.1
 * @date      2022-04-18
 * @copyright Copyright (c) 2022
 */


#include<opencv2/opencv.hpp>
#include<iostream>


using namespace cv;
using namespace std;

/*************************将RGB彩色图转换为灰度图**********************************/
/**
* 函数名：     color2gray
* 参数：       const Mat& color_img，需要转换为灰度图的原图像
* 返回值：     const Mat gray_img，转换后的灰度图
* 函数功能：   将RGB彩色图转换为灰度图
* ******************************************************
* 思路：
* 1.彩色图片通道分离，并构造一个单通道图
* 2.遍历每一个像素点，将rgb对应亮度值转化为灰度图对应亮度值
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



/*************************将RGB彩色图转换为HSV图**********************************/
/**
* 函数名：     color2hsv
* 参数：       const Mat& color_img，需要转换为HSV图的原图像
* 返回值：     const Mat hsv_img，转换后的灰度图
* 函数功能：   将RGB彩色图转换为HSV图
* ******************************************************
* 思路：
* 1.先copy一份RGB彩色图，方便操作
* 2.遍历每一个像素点，套公式将R、G、B转换为对应的H、S、V求出来
*   这里主要是利用Vec3b类型的向量，得到R、G、B三通道的值
*   然后将对应求出的值以同样的方法放入H、S、V三通道中
*/
Mat color2hsv(Mat color_img)
{
  // [1] copy一份RGB彩色图，
  Mat hsv_img = Mat::zeros(color_img.size(), CV_8UC3);

  // [2] 遍历每一个像素点，套公式将R、G、B转换为对应的H、S、V求出来
  for (int i = 0; i < color_img.rows; i++)
  {
    // [2-1] 利用Vec3b类型的向量，得到R、G、B三通道的值
    Vec3b *rgb = color_img.ptr<Vec3b>(i);   //B--rgb[0]  G--rgb[1]  R--rgb[2]
    // [2-2] 利用Vec3b类型的向量，得到H、S、V三通道
    Vec3b *hsv = hsv_img.ptr<Vec3b>(i);     //B--hsv[0]  G--hsv[1]  R--hsv[2]

    // [2-3] 进行计算
    for (int j = 0; j < color_img.cols; j++)
    {
      float B = rgb[j][0] / 255.0;
      float G = rgb[j][1] / 255.0;
      float R = rgb[j][2] / 255.0;

      //V通道值
      float V = (float)max({ B, G, R });
      float vmin = (float)min({ B, G, R });
      float diff = V - vmin;


      float S, H;
      // S 通道值
      if (V == 0)
          S= 0;
      else
          S = diff / V;

      //H通道值
      if (V == B)   //V=B
      {
        H = 240.0 + (R - G) * diff;
      }
      else if (V == G)  //V=G
      {
        H = 120.0 + (B - R) * diff;
      }
      else if (V == R)   //V=R
      {
        H = (G - B) * diff;
      }


      H = (H < 0.0) ? (H + 360.0) : H;

      // [2-4] 将求出的数据放入通道中
      hsv[j][0] = (uchar)(H / 2);
      hsv[j][1] = (uchar)(S * 255);
      hsv[j][2] = (uchar)(V * 255);
    }
  }
  return hsv_img;//返回
}


int main()
{
    //1.读取图像
    Mat img=imread("D:/Picture/lena.jpg");

    //未获取到图像
    if(img.empty())
    {
        cout << "can't read this image!" << endl;
        return 0;
    }


    //2.转换为灰度图
    Mat gray_img=color2gray(img);
    //3.转换为HSV图
    Mat hsv_img=color2hsv(img);

    //4.展示图像
    imshow("RGB image",img);
    imshow("Gray image",gray_img);
    imshow("HSV image",hsv_img);

    //5.等待按键操作
    waitKey(0);


    return 0;
}

