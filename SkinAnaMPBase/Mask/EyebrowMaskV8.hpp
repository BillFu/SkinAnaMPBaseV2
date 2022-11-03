/*
EyebrowMaskV8.hpp

本模块用于计算Eyebrow Mask。
基本思路：

 
Author: Fu Xiaoqiang
Date:   2022/11/3
*/

#ifndef EYEBROW_MASK_V8_HPP
#define EYEBROW_MASK_V8_HPP

#include "opencv2/opencv.hpp"

using namespace std;
using namespace cv;

#include "../Common.hpp"


// Ip: interpolate
Point2i IpPtInERG(const Point2i eyeRefPts[NUM_PT_EYE_REFINE_GROUP],
                  int pIndex1, int pIndex2, float t);


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
                    const FaceSegRst& segResult, // input, const
                    Mat& outMask);

/**********************************************************************************************/
// forge the initial polygon covering one eye
// the points in the returned initEyePg are measured in Source Space.
// in the final step, expand the polygon
void ForgeInitEyePg(const Point2i eyeRefinePts[NUM_PT_EYE_REFINE_GROUP],
                    const Point2i& eyeSegCP, POLYGON& initEyePg);

void ForgeEyesMask(const Mat& srcImage,
                   const FaceInfo& faceInfo,
                   const FaceSegRst& segResult, // input, const 
                   Mat& outMask);

// forge the polygong of one eye, only use the result of face/bg segment
void ForgeEyePg(Size srcImgS, const SegMask& eyeSegMask,
                const EyeSegFPs& eyeFPs,
                const Mat& srcImage,
                const string& outFileName);

void ForgeEyePgV2(Size srcImgS, const SegMask& eyeSegMask,
                  const SegEyeFPsNOS& eyeFPsNOS,
                const Mat& srcImage,
                  const string& outFileName);

#endif /* end of EYEBROW_MASK_V8_HPP */
