//
//  EyebowMask.hpp
//
//
/*
利用分割的结果来计算眼睛水平线以下的脸部轮廓和Mask，嘴部、胡子、部分眼睛在集成时再去剔除。
 
Author: Fu Xiaoqiang
Date:   2022/10/3
*/

#ifndef LOWER_FACE_MASK_HPP
#define LOWER_FACE_MASK_HPP

#include "opencv2/opencv.hpp"

using namespace std;
using namespace cv;

#include "../Common.hpp"


/**********************************************************************************************
利用分割的结果来计算眼睛水平线以下的脸部轮廓和Mask，
嘴部、胡子、部分眼睛在集成时再去剔除。
***********************************************************************************************/
void ForgeLowerFaceMask(const FaceSegResult& segResult,
                        const Mat& fbBiLab, Mat& outMask);


/**********************************************************************************************
利用分割的结果来计算眼睛水平线以下的脸部轮廓和Mask，
胡子也被剔除。
***********************************************************************************************/
void ForgeLFMaskExBeard(const FaceSegResult& segResult,
                        const Mat& fbBiLab, Mat& outMask);


#endif /* end of LOWER_FACE_MASK_HPP */
