//
//  EyebrowMaskV4.hpp
//
//
/*
本模块用于计算Eyebrow Mask。
以分割出来的brow mask为雏形，采用极坐标(r, theta)旋转扫描轮廓一周，获得轮廓点序列，再对r序列按照均匀theta间隔进行插值，
并光滑，最后加以伸长，再由极坐标返回到笛卡尔坐标，获得我们想要的轮廓。
 
Author: Fu Xiaoqiang
Date:   2022/10/12
*/

#ifndef EYEBROW_MASK_V4_HPP
#define EYEBROW_MASK_V4_HPP

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
                    const FaceSegResult& segResult, // input, const
                    Mat& outMask);

/**********************************************************************************************
only cover the two eyes
***********************************************************************************************/
// extract the primary and coarse polar sequence on the contour
void CalcPolarSeqOnCt(const CONTOUR& spCt, const Point2i& eyeCP,
                      PolarContour& polarCont);

// build a new version of polar point sequence that with even intervals of theta.
void BuildEvenPolarSeq(const PolarContour& rawPolarSeq,
                       int num_interval, // how many intervals from 0 to 2*Pi
                       PolarContour& evenPolarSeq);

void ForgeEyePgBySegRst(Size srcImgS, const SegMask& eyeSegMask, const Point2i& eyeCP,
                        POLYGON& eyePg);
//void ForgeEyePgBySegRst(const SegMask& eyeSegMask, const Point2i& eyeCP,
//                        POLYGON& eyePg);

void ForgeEyesMask(const FaceInfo& faceInfo,
                   const FaceSegResult& segResult, // input, const 
                   Mat& outMask);

#endif /* end of EYEBROW_MASK_V4_HPP */
