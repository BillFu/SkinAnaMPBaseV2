//
//  SkinFeatureMask.hpp
//
//
/*
本模块目前构建皮肤各类特征的检测区域的矢量版雏形。
 
Author: Fu Xiaoqiang
Date:   2022/9/23
*/

#ifndef SKIN_FEATURE_MASK_HPP
#define SKIN_FEATURE_MASK_HPP

#include "opencv2/opencv.hpp"
#include "Common.hpp"

using namespace std;
using namespace cv;


//-------------------------------------------------------------------------------------------

/**********************************************************************************************
本函数构建Pore的矢量版Mask雏形。
***********************************************************************************************/
void ForgePoreMask(const FaceInfo& faceInfo,
                   const Mat& fullFaceMask,
                   const Mat& eyeFullMask,  // cover the eyes and eyebows and the surrounding nearby area
                   const Mat& mouthMask, // the enlarged mouth mask
                   Mat& outPoreMask);

void ForgePoreMaskV2(const FaceInfo& faceInfo,
                   const Mat& faceLowMask,  // lower than eyes
                   const Mat& foreheadMask,
                   const Mat& eyeFullMask,  // cover the eyes and eyebows and the surrounding nearby area
                   const Mat& mouthMask, // the enlarged mouth mask
                   Mat& outPoreMask);

#endif /* end of SKIN_FEATURE_MASK_HPP */
