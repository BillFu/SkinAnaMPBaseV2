//
//  SkinFeatureMask.cpp

/*******************************************************************************
Author: Fu Xiaoqiang
Date:   2022/9/23

********************************************************************************/

#include "SkinFeatureMask.hpp"
#include "FundamentalMask.hpp"

/**********************************************************************************************
本函数构建Pore的矢量版Mask雏形。
***********************************************************************************************/
void ForgePoreMask(const FaceInfo& faceInfo,
                   const Mat& fullFaceMask,
                   const Mat& eyeFullMask,  // cover the eyes and eyebows and the surrounding nearby area
                   const Mat& mouthMask, // the enlarged mouth mask
                   Mat& outPoreMask)
{
    outPoreMask = fullFaceMask & (~eyeFullMask) & (~mouthMask);
}


void ForgePoreMaskV2(const FaceInfo& faceInfo,
                   const Mat& faceLowMask,  // lower than eyes
                   const Mat& foreheadMask,
                   const Mat& eyeFullMask,  // cover the eyes and eyebows and the surrounding nearby area
                   const Mat& mouthMask, // the enlarged mouth mask
                   const Mat& noseMask,
                   Mat& outPoreMask)
{
    Mat outMask = faceLowMask | foreheadMask | noseMask ;
    outPoreMask = outMask & (~eyeFullMask) & (~mouthMask);
    
    //int expandSize = 20;
    //expanMask(outMask, expandSize, outPoreMask);
}
