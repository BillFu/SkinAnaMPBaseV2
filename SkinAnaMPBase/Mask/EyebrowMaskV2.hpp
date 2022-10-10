//
//  EyebrowMaskV2.hpp
//
//
/*
本模块用于计算Eyebow Mask。用分割出的眼睛CP点对Refined Eye Group中的点坐标进行纠正。
目前先考虑简单的单点校正。
 
Author: Fu Xiaoqiang
Date:   2022/10/10
*/

#ifndef EYEBROW_MASK_V2_HPP
#define EYEBROW_MASK_V2_HPP

#include "opencv2/opencv.hpp"

using namespace std;
using namespace cv;

#include "../Common.hpp"


/**********************************************************************************************

***********************************************************************************************/
void ForgeBrowsMask(const FaceInfo& faceInfo, Mat& outMask);


/**********************************************************************************************
only cover the two eyes
***********************************************************************************************/
void ForgeEyesMask(const FaceInfo& faceInfo,
                   const FaceSegResult& segResult, // input, const 
                   Mat& outMask);

#endif /* end of EYEBROW_MASK_V2_HPP */
