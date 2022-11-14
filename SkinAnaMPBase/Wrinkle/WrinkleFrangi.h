//
//  wrinkle_frangi.h
//  MCSkinAnaLib
//
//  Created by Fu on 2022/10/20.
//  Copyright © 2022 MeiceAlg. All rights reserved.
//

/****************************************************************************
 本模块的功能是，用Frangi滤波来提取粗纹理。本模块由老版算法拆分而来。
 作者：傅晓强
 日期：2022/11/1
 ****************************************************************************/

#ifndef WRINKLE_FRANGI_H
#define WRINKLE_FRANGI_H

#include "opencv2/opencv.hpp"

#include "../Common.hpp"

using namespace cv;
using namespace std;

#ifdef TEST_RUN_WRK
extern string wrk_out_dir;
#endif

void PreprocGrImg(const Mat& grSrcImg, Mat& outImg);

//-----------------------------------------------------------------------------

// Cc: calculate
// blur --> clahe --> frangi
/*
void CcFrgiMapInFR(const Mat& imgGray,
                    const Rect& faceRect,
                    int scaleRatio,
                    Mat& frgiMapSSInFR);
*/

// return Mat is 8U type
Mat CcFrgiMapInRect(const Mat& imgGray,
                    const Rect& rect,
                    int scaleRatio);

void CcFrgiMap(const Mat& imgGray, int scaleRatio, Mat& frgiMap8U);

// 最核心的Frangi滤波环节
void ApplyFrgiFilter(const Mat& grSrcImg,
                     int scaleRatio,
                     Mat& frangiRespRz);

//-----------------------------------------------------------------------------
// 从frangi滤波的结果（经过了二值化、细化、反模糊化等处理）中，提取深皱纹、长皱纹
// DL: deep and long
void PickDLWrkInFrgiMapV2(int minWrkTh, int longWrkTh,
                      Mat& frgiResp8U,
                      CONTOURS& deepWrkConts,
                      CONTOURS& longWrkConts);
#endif /* WRINKLE_FRANGI_H */
