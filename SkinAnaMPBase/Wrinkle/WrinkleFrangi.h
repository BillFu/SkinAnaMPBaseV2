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

// 计算Frangi滤波响应，并提取深皱纹和长皱纹
void CalcFrgiRespAndPickWrk(const Mat& imgInFR,
                         const Mat& wrkMaskInFR,
                         int scaleRatio,
                         int minWrkSize,
                         int longWrkThresh,
                         Mat& frangiRespRz,  // Rz: resized, i.e., scale down
                         CONTOURS& deepWrkConts,
                         CONTOURS& longWrkConts,
                         float& avgFrgiRespValue);

// 从frangi滤波的结果（经过了二值化、细化、反模糊化等处理）中，提取深皱纹、长皱纹
// DL: deep and long
void PickDLWrkFromFrgiResp(const Mat& frgiRespOS, //Original Scale
                                  const Mat& wrkMaskInFR, // 原始尺度，经过了Face_Rect裁切
                                  int longWrkThresh,
                                  unsigned int minsWrkSize,
                                  CONTOURS& longWrkConts,
                                  CONTOURS& DeepWrkConts);

void CalcFrgiRespInFhReg(const Mat& grSrcImg,
                         const Rect& fhRect,
                         int scaleRatio,
                         Mat& frangiRespRz);


void CalcSobelRespInFhReg(const Mat& grSrcImg,
                         const Rect& fhRect,
                         int scaleRatio,
                          Mat& frgiRespRz);
#endif /* WRINKLE_FRANGI_H */
