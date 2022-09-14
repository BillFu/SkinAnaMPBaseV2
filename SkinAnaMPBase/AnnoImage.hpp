//
//  AnnoImage.hpp
//
//
/******************************************************************************************
本模块的作用在于，将计算的结果以可视化的方式打印在输入影像的拷贝上，便于查看算法的效果。
 
Author: Fu Xiaoqiang
Date:   2022/9/11
******************************************************************************************/

#ifndef ANNO_IMAGE_HPP
#define ANNO_IMAGE_HPP

#include<opencv2/opencv.hpp>

using namespace std;
using namespace cv;


/******************************************************************************************
本函数的功能是，将人脸关键点提取和位姿估计的结果打印在输入影像的拷贝上。
*******************************************************************************************/
void AnnoHeadPoseEst(const Mat& srcImage, Mat& annoImage,
                     int lm_2d[468][2], float pitch, float yaw, float roll);

#endif /* end of ANNO_IMAGE_HPP */
