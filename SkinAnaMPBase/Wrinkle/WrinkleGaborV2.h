//
//  WrinkleGaborV2.h
//  MCSkinAnaLib
//
//  Created by Fu on 2022/11/1.
//  Copyright © 2022 MeiceAlg. All rights reserved.
//
/****************************************************************************
 本模块的功能是，用Gabor滤波来提取细纹理。
 作者：傅晓强
 日期：2022/11/15
*****************************************************************************/

#ifndef WRINKLE_GABOR_V2_H
#define WRINKLE_GABOR_V2_H

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
                  Mat& WrinkRespMap,
                  Mat& fhGabMap8U,
                  Mat& glabGabMap8U);

void BuildGabOptsForEb(int kerSize, int sigma, bool isLeftEye, GaborOptBank& gOptBank);
Mat CcGaborMapInOneEyebag(const Mat& grFtSrcImg, int kerSize, int sigma,
                        bool isLeftEye, const Rect& ebRect);

void BuildGabOptsForCF(int kerSize, int sigma, 
                       bool isLeftEye, GaborOptBank& gOptBank);
// 鱼尾纹
Mat CcGaborMapInOneCrowFeet(const Mat& grFtSrcImg, int kerSize, int sigma,
                        bool isLeftEye, const Rect& cfRect);

// forehead，前额
Mat CcGaborMapOnFh(const Mat& grFtSrcImg, int kerSize, int sigma,
                   const Rect& fhRect);

// glabella，眉间，印堂
Mat CcGaborMapOnGlab(const Mat& grFtSrcImg, int kerSize, int sigma,
                     const Rect& glabRect);

void BuildGabOptsForCheek(int kerSize, int sigma,
                          bool isLeftEye, GaborOptBank& gOptBank);
Mat CcGaborMapInOneCheek(const Mat& grFtSrcImg, int kerSize, int sigma,
                         bool isLeft, const Rect& cheekRect);
///////////////////////////////////////////////////////////////////////////////////////////////

// 老百姓理解的深、浅皱纹是以几何深度来划分的，但图像算法又是以颜色深浅来测量的。
// 提取浅皱纹，Ext: Extract
void ExtLightWrk(const Mat& wrkGaborMap,
                     int minLenOfWrk,
                     int longWrkThresh,
                     CONTOURS& LightWrkConts,
                     CONTOURS& LongWrkConts,
                     int& totalLength);

// 提取深皱纹，Ext: Extract
void ExtDeepWrk(const Mat& wrkGaborRespMap,
                    int minLenOfWrk,
                    int longWrkThresh,
                    CONTOURS& DeepWrkConts,
                    CONTOURS& LongWrkConts);

Mat drawFhWrk(const Mat& canvas, const CONTOURS& LightWrkConts);

void ExtWrkFromFhGabMap(const Rect& fhRect,
                        const Mat& fhGabMap8U,
                        int minLenOfWrk,
                        int longWrkThresh,
                        CONTOURS& DeepWrkConts,
                        CONTOURS& LongWrkConts);

void ExtWrkFromGlabGabMap(const Rect& glabRect,
                        const Mat& glabGabMap8U,
                        int minLenOfWrk,
                        int longWrkThresh,
                        CONTOURS& DeepWrkConts,
                        CONTOURS& LongWrkConts);

#endif /* WRINKLE_GABOR_V2_H */
