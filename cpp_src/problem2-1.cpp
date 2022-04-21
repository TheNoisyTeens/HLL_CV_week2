/*
*@file       problem2-1.cpp
*@author     zhangmin(3273749257@qq.com)
*@brief      视频播放并用滑动条控制
*@version    0.3
*@date       2022-04-18
*
*@copyright  Copyright (c) 2022
*
*/


/*
*不足：
* 1. video_flag暂停-播放标志位，无法真正判断出视频在播放还是暂停，
*    只是因为视频一开始是播放，故初始化video_flag=0并设定video_flag=0时暂停
*/

#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;





/******************全局变量定义**********************/
int frames;//视频总帧数
int fps;//帧率
int pos=1;//指向滑动条当前位置
VideoCapture cap;//用于打开视频文件
int video_flag=0;//设置视频暂停、开启标志，0为暂停，1为开启
int esc_flag=1;//设置退出标志位，为0则退出.该标志位实现在按下暂停时仍然可以退出
//宏定义按键等待时间，如果等待太久会导致按键不灵敏
#define time 1000/fps//一帧时间

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
* 函数功能：    实现键功能：按下A键回放，按下D键快进，按下空格键继续播放或暂停
* ****************************************************************
* 思路：
* 1.A键：回放功能
*   将视频下一帧位置设置为0
* 2.D键：快进
*   将视频下一帧位置加4
* 3.Esc键：退出视频播放
*   设置退出标志位为0
* 4.空格键：视频暂停与继续
*   设置了一个标志位判断暂停还是播放
*   若为暂停，使用while等待空格按键或Esc退出键
*/
void key_control(void){

    char myKey=waitKey(time);
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
    else if (myKey==27) {
        esc_flag=0;//设置退出标志位为0
    }

    /****************按下空格****************/
    else if(myKey==32){
        //video_flag==1，播放
        if(video_flag){
            cout<<"继续播放"<<endl;
            video_flag=0;//下一次暂停
        }

        //video_flag==0，暂停
        if(!video_flag){
            cout<<"暂停"<<endl;
            //等待再次播放或按下Esc键退出
            while (1) {
                char key_temp=waitKey(time);
                if(key_temp==32||key_temp==27){
                    //如果按下Esc键退出
                    if(key_temp==27)
                        esc_flag=0;//设置退出标志位为0
                    break;//跳出循环，继续播放或退出
            }
            video_flag=1;//下一次播放
        }
    }
}
    return;
}

int main() {
    namedWindow("video");

    // 创建一个VideoCapture对象，用于打开视频文件
    cap.open("C:/Users/86157/Videos/TNT/TNT.mp4");

    frames=cap.get(CAP_PROP_FRAME_COUNT);//获取当前帧数
    fps=cap.get(CAP_PROP_FPS);//获取帧率

    // 创建一个滑动条
    cv::createTrackbar("scrollbar", "video", &pos, frames, onTrackbarSlide);

    //打开视频失败
    if(!cap.isOpened())
            return 0;


    while (1) {

        /***********在这里实现视频播放*************/
        Mat frame;
        cap>>frame;//取帧给frame

        imshow("video",frame);//显示这一帧


        /**************设置滚动条跟着视频移动******************/
        pos=cap.get(CAP_PROP_POS_FRAMES);
        setTrackbarPos("scrollbar","video",pos);

        /***************如果播放完成，等待按键操作************/
        if(frame.empty()){
            cout<<"完成播放"<<endl;
            waitKey(0);
        }


        /*实现A键回放，D键快进，Esc键退出视频播放，空格暂停启动键功能*/
        key_control();

        /****************按下Esc键退出播放****************/
        if(esc_flag==0)
            break;

    }

    return 0;
}

