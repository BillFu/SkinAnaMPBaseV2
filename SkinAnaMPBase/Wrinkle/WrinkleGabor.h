//
//  WrinkleGabor.h
//  MCSkinAnaLib
//
//  Created by Fu on 2022/11/1.
//  Copyright © 2022 MeiceAlg. All rights reserved.
//
/****************************************************************************
 本模块的功能是，用Gabor滤波来提取细纹理。
 作者：傅晓强
 日期：2022/11/1
*****************************************************************************/

#ifndef WRINKLE_GABOR_H
#define WRINKLE_GABOR_H

#include "opencv2/opencv.hpp"

#include "cvgabor.h"
#include "../Common.hpp"

using namespace cv;
using namespace std;

#ifdef TEST_RUN_WRK
extern string wrk_out_dir;
#endif

void AnnoPointsOnImg(Mat& annoImage,
                         const SPLINE& pts,
                     int ptIDs[], int numPt);

void InitLCheekGaborBank(CvGabor lcGabor[5]);

// 返回左面颊的Gabor滤波响应值
Mat CalcOneCheekGaborResp(const vector<Point2i>& wrkSpline,
                          int spPtIDs[4],
                          CvGabor lcGabor[5],
                          const Rect& faceRect,
                          const Mat& inImgInFR_Gray,
                          Rect& cheekRect);

// 返回左面颊的Gabor滤波响应值
// 返回右面颊的Gabor滤波响应值

// glabella，眉间，印堂
Mat CalcGlabellaGaborResp(const vector<Point2i>& wrinkle_spline_curve,
                  const Rect& FaceContour_Rect,
                  const Mat& allWrinkle,
                  Rect& erect, const Rect& lrect);

// forehead，前额
Mat CalcFhGaborResp(const vector<Point2i>& wrinkle_spline_curve,
                  const Rect& FaceContour_Rect,
                  Mat& allWrinkle, Rect& frect);

// 鼻梁上半部，rhinion(keystone)
Mat CalcUpperNoseGaborResp(const vector<Point2i>& wrinkle_spline_curve,
                  const Rect& FaceContour_Rect,
                  const Mat& allWrinkle, Rect& nrect);

void CalcGaborResp(const Mat& grFrImg,
                           const Rect& Face_Rect,
                           const SPLINE& wrinkle_spline,
                           Mat& WrinkRespMap //WrinkRespMap的大小和在原始影像坐标系中的位置由Face_Rect限定
                           );

// 老百姓理解的深、浅皱纹是以几何深度来划分的，但图像算法又是以颜色深浅来测量的。
// 提取浅皱纹，Ext: Extract
void ExtLightWrk(const Mat& wrkGaborRespMap,
                     const Mat& wrkMaskInFR, // wrinkle mask cropped by face rectangle
                     const Rect& faceRect,
                     int minLenOfWrk,
                     int longWrkThresh,
                     CONTOURS& LightWrkConts,
                     CONTOURS& LongWrkConts,
                     int& totalLength);

// 提取深皱纹，Ext: Extract
void ExtDeepWrk(const Mat& wrkGaborRespMap,
                    const Mat& wrkMaskInFR,
                    const Rect& faceRect,
                    int minLenOfWrk,
                    int longWrkThresh,
                    CONTOURS& DeepWrkConts,
                    CONTOURS& LongWrkConts);

void ExtWrkInFhGaborResp(const Mat& fhGaborMap,
                     const Rect& faceRect,
                     int minLenOfWrk,
                     CONTOURS& LightWrkConts);

Mat drawFhWrk(const Mat& canvas, const CONTOURS& LightWrkConts);

#endif /* WRINKLE_GABOR_H */
