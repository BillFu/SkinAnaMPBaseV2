//
//  WrinkleGaborV3.h
//  MCSkinAnaLib
//
//  Created by Fu on 2022/11/1.
//  Copyright © 2022 MeiceAlg. All rights reserved.
//
/****************************************************************************
 本模块的功能是，用Gabor滤波来提取细纹理。
 作者：傅晓强
 日期：2022/11/17
*****************************************************************************/

#ifndef WRINKLE_GABOR_V3_H
#define WRINKLE_GABOR_V3_H

#include "opencv2/opencv.hpp"

#include "GaborDL.hpp"
#include "../Common.hpp"

using namespace cv;
using namespace std;


void AnnoPointsOnImg(Mat& annoImage,
                         const SPLINE& pts,
                     int ptIDs[], int numPt);

// agg: aggregated
void ApplyGaborBank(const GaborOptBank& gBank, const Mat& inGrFtImg,
                    Mat& aggGabMapFt);

//WrinkRespMap的大小和在原始影像坐标系中的位置由Face_Rect限定
void CalcGaborMap(const Mat& grSrcImg,
                  WrkRegGroup& wrkRegGroup,
                  Mat& fhGabMap8U,
                  Mat& glabGabMap8U,
                  Mat& lCirEyeGabMap8U,
                  Mat& rCirEyeGabMap8U,
                  Mat& lNagvGabMap8U,
                  Mat& rNagvGabMap8U);

// forehead，前额
Mat CcGaborMapOnFh(const Mat& grFtSrcImg, int kerSize, int sigma,
                   const Rect& fhRect);

Mat CcGaborMapOnFhV2(const Mat& grFtSrcImg, int kerSize, int sigma,
                   const Rect& fhRect);

// glabella，眉间，印堂
Mat CcGaborMapOnGlab(const Mat& grFtSrcImg, int kerSize, int sigma,
                     const Rect& glabRect);

void BuildGabOptsForNagv(int kerSize, int sigma, bool isLeft, GaborOptBank& gOptBank);

Mat CcGabMapInOneNagv(bool isLeft, const Mat& grFtSrcImg,
                      int kerSize, int sigma,
                      const Rect& nagvRect);

void BuildGabOptsForCirEye(int kerSize, int sigma, bool isLeft, GaborOptBank& gOptBank);

Mat CcGabMapInOneCirEye(bool isLeft, const Mat& grFtSrcImg,
                        int kerSize, int sigma,
                        const DetectRegion& cirEyeReg);

///////////////////////////////////////////////////////////////////////////////////////////////

// 老百姓理解的深、浅皱纹是以几何深度来划分的，但图像算法又是以颜色深浅来测量的。

Mat drawFhWrk(const Mat& canvas, const CONTOURS& LightWrkConts);

void ExtWrkFromFhGabMap(const DetectRegion& fhReg,
                        const Mat& fhGabMap8U,
                        int minWrkTh,
                        int longWrkThresh,
                        CONTOURS& DeepWrkConts,
                        CONTOURS& LongWrkConts);

void ExtWrkFromGlabGabMap(const DetectRegion& glabReg,
                        const Mat& glabGabMap8U,
                        int minWrkTh,
                        int longWrkThresh,
                        CONTOURS& DeepWrkConts,
                        CONTOURS& LongWrkConts);

void ExtWrkInNagvGabMap(bool isLeft,
                        const DetectRegion& nagvReg,
                        const Mat& nagvGabMap8U,
                        int minWrkTh,
                        int longWrkThresh,
                        CONTOURS& DeepWrkConts,
                        CONTOURS& LongWrkConts);

void ExtWrkInCirEyeGabMap(bool isLeft,
                        const DetectRegion& cirEyeReg,
                        const Mat& cirEyeGabMap8U,
                        int minWrkTh,
                        int longWrkThresh,
                        CONTOURS& lightWrkConts,
                          CONTOURS& longWrkConts);

#endif /* WRINKLE_GABOR_V3_H */
