//
//  EyebowMask.hpp
//
//
/*
本模块用于计算Eyebow Mask。
 
Author: Fu Xiaoqiang
Date:   2022/9/18
*/

#ifndef EYEBOW_MASK_HPP
#define EYEBOW_MASK_HPP

#include "opencv2/opencv.hpp"

using namespace std;
using namespace cv;

#include "../Common.hpp"


/**********************************************************************************************

***********************************************************************************************/
void ForgeTwoEyebowsMask(const FaceInfo& faceInfo, Mat& outMask);

#endif /* end of EYEBOW_MASK_HPP */
