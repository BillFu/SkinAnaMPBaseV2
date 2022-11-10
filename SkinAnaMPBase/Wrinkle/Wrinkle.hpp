//
//  Wrinkle.hpp
//
//
/*
本模块负责检测皱纹和相关处理。
 
Author: Fu Xiaoqiang
Date:   2022/11/1
*/

#ifndef WRINKLE_HPP
#define WRINKLE_HPP

#include "opencv2/opencv.hpp"

using namespace std;
using namespace cv;

#include "../Common.hpp"

// wrkGaborRespMap: 输出，记录Gabor滤波的结果，大小和位置由Face_Rect来限定
void DetectWrinkle(const Mat& inImg, const Rect& faceRect,
                   const Mat& wrkFrgiMask,
                   WrkRegGroup& wrkRegGroup,
                   CONTOURS& deepWrkConts,
                   Mat& wrkGaborRespMap);

void PreprocGrImg(const Mat& grSrcImg, Mat& outImg);

#endif /* end of WRINKLE_HPP */
