//
//  EyebrowMask.hpp
//
//
/*
本模块用于计算Eyebow Mask。
 
Author: Fu Xiaoqiang
Date:   2022/9/18
*/

#ifndef EYEBROW_MASK_HPP
#define EYEBROW_MASK_HPP

#include "opencv2/opencv.hpp"

using namespace std;
using namespace cv;

#include "../Common.hpp"


/**********************************************************************************************

***********************************************************************************************/
void ForgeEyebrowsMask(const FaceInfo& faceInfo, Mat& outMask);


/**********************************************************************************************
only cover the two eyes
***********************************************************************************************/
void ForgeEyesMask(const FaceInfo& faceInfo, Mat& outMask);

#endif /* end of EYEBROW_MASK_HPP */
