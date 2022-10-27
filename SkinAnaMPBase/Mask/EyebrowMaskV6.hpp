/*
EyebrowMaskV6.hpp

本模块用于计算Eyebrow Mask。
基本思路：

 
Author: Fu Xiaoqiang
Date:   2022/10/26
*/

#ifndef EYEBROW_MASK_V6_HPP
#define EYEBROW_MASK_V6_HPP

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
                    float expandScale, int& numPtsUC, POLYGON& initEyePg);

// eyeCP: given by face/bg segment and in source space
void ForgeEyePgBySnakeAlg(Size srcImgS,
                          const Point2i eyeRefinePts[NUM_PT_EYE_REFINE_GROUP],
                          const SegMask& eyeSegMask,
                          const EyeFPs& eyeFPs,
                          const Point2i& eyeCP,
                          POLYGON& eyePg);

void ForgeEyesMask(const Mat& srcImage,
                   const FaceInfo& faceInfo,
                   const FaceSegRst& segResult, // input, const 
                   Mat& outMask);


// 提取眼睛轮廓线的上、下弧线
void SplitEyeCt2UpLowCurves(const CONTOUR& nosEyeCont,
                            const Point2i& nosLCorPt,
                            const Point2i& nosRCorPt,
                            CONTOUR& upEyeCurve,
                            CONTOUR& lowEyeCurve);


// 移动初始眼睛轮廓线上的点，让它们跳出Mask的包围圈
// 逃出包围圈的方向：从中心点出发，到当前轮廓点连一条线，沿这条线远离中心点，最终可逃离包围圈，或到达工作区边界。
// 采用新思路，就可以不用区分上弧线、下弧线。
// eyeCPWC: center point of eye in working coordinate system
void MoveInitEyePtsOutMask(const POLYGON& eyePgWC,
                           int workRegW, int workRegH,
                           const Mat& workMask,
                           const Point2i& eyeCPWC,
                           POLYGON& adjustEyePgWC);

#endif /* end of EYEBROW_MASK_V6_HPP */
