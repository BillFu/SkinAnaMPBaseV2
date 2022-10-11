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


// Ip: interpolate
Point2i IpPtInERG(const Point2i eyeRefPts[NUM_PT_EYE_REFINE_GROUP],
                  int pIndex1, int pIndex2, float t);


//ERPs: eye refined points
// 用分割出的眼睛中心点来修正ERPs的坐标
// transform the eye refine points in face mesh space into the segment space
void FixERPsBySegEyeCP(const Point2i eyeRefPts[NUM_PT_EYE_REFINE_GROUP], // input
                       const Point2i& segCP, // eye center point in segment space
                       Point2i fixedEyeRefPts[NUM_PT_EYE_REFINE_GROUP]   // output
);


//ERPs: eye refined points
// 用分割出的眉毛中心点来修正ERPs的坐标
// transform the eye refine points in face mesh space into the segment space
void FixERPsBySegBrowCP(const Point2i eyeRefPts[NUM_PT_EYE_REFINE_GROUP], // input
                       const Point2i& segCP, // eye center point in segment space
                       Point2i fixedEyeRefPts[NUM_PT_EYE_REFINE_GROUP]   // output
);
/**********************************************************************************************

***********************************************************************************************/
// Note: Only forge One!!!
void ForgeBrowPg(const Point2i eyeRefPts[NUM_PT_EYE_REFINE_GROUP],
                 const Point2i& segBrowCP,POLYGON& browPg);

void ForgeBrowsMask(const FaceInfo& faceInfo,
                    const FaceSegResult& segResult, // input, const
                    Mat& outMask);

/**********************************************************************************************
only cover the two eyes
***********************************************************************************************/
void ForgeEyesMask(const FaceInfo& faceInfo,
                   const FaceSegResult& segResult, // input, const 
                   Mat& outMask);

#endif /* end of EYEBROW_MASK_V2_HPP */
