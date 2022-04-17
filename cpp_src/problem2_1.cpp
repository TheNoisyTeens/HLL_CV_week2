/*
*@file       problem2_1.cpp
*@author     zhangmin(3273749257@qq.com)
*@brief      视频播放并用滑动条控制
*@version    0.1
*@date       2022-04-14
*
*@copyright  Copyright (c) 2022
*
*/


/*
*不足：
* 1. video_flag暂停-播放标志位，无法真正判断出视频在播放还是暂停，
*    只是因为视频一开始是播放，故初始化video_flag=0并设定video_flag=0时暂停
* 2. 视频播放完成就退出了，无法实现A键、Esc键等键的功能
* 3. A键有时候不大灵敏
* 4.空格暂停键没有实现
*/

#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

/******************全局变量定义**********************/
int frames;//视频帧数
int fps;//帧率
int pos=1;//指向滑动条当前位置
VideoCapture cap;//用于打开视频文件
int video_flag=0;//设置视频暂停、开启标志，0为暂停，1为开启



/*
* 函数名：     onTrackbarSlide
* 参数：       int pos，滑动条位置
*             void *，用户数据，这里不需要
* 返回值：     无
* 函数功能：    作为回调函数，若滑块位置移动，当前视频帧数跟着移动
*/
void onTrackbarSlide(int pos,void *){
   cap.set(CAP_PROP_POS_FRAMES,pos);
}

/*
* 函数名：     key_control
* 参数：       无
* 返回值：     无
* 函数功能：    实现键功能：按下A键回放，按下空格键继续播放或暂停
*/
void key_control(void){

    char myKey=waitKey(1000/fps);
    /****************按下A键****************/
    if(myKey=='a') {
        cap.set(CAP_PROP_POS_FRAMES,0);//回放
        cout<<"回放"<<endl;
    }

  /****************按下D键****************/
    else if(myKey=='d'){
        cout<<"快进"<<endl;
        pos=cap.get(CAP_PROP_POS_FRAMES);//获取当前帧数
        setTrackbarPos("scrollbar","video",pos+4);//实现快进(移动4帧)
    }


    /****************按下空格****************/
    else if(myKey==20){


        //video_flag==1，播放
        if(video_flag){
             cout<<"继续播放"<<endl;
            video_flag=0;
        }


        //video_flag==0，暂停
        if(!video_flag){
             cout<<"暂停"<<endl;
             //等待再次播放或按下Esc键退出
            while(waitKey(10)!=20&&waitKey(10)!=27);
            video_flag=1;
        }
    }

    return;
}

int main() {
    namedWindow("video");

    //1.创建一个VideoCapture对象，用于打开视频文件
    cap.open("C:/Users/86157/Videos/TNT/TNT.mp4");

    frames=cap.get(CAP_PROP_FRAME_COUNT);//获取当前帧数
    fps=cap.get(CAP_PROP_FPS);//获取帧率

    //创建一个滑动条
    cv::createTrackbar("scrollbar", "video", &pos, frames, onTrackbarSlide);

    //打开视频失败
    if(!cap.isOpened())
            return 0;


    while (1) {


   /***********在这里实现视频播放*************/
        Mat frame;
        cap>>frame;//取帧给frame

        imshow("video",frame);//显示这一帧
   /*****************************************/

        /**************设置滚动条跟着视频移动******************/
        pos=cap.get(CAP_PROP_POS_FRAMES);
        setTrackbarPos("scrollbar","video",pos);

        /***************如果播放完成，等待按键操作************/
        if(frame.empty()){
            cout<<"完成播放"<<endl;
            waitKey(0);
        }

        /*******实现Esc键功能，若按下Esc键则退出视频播放*******/
        char t=waitKey(10);
        if(t==27){
            cout<<"退出播放"<<endl;
            break;               
        }

        /****实现A键和空格键功能*****/
        key_control();


    }

    return 0;
}

