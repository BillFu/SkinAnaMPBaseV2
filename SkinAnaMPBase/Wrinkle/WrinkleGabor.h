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

/*
void InitLCheekGaborBank(CvGabor lcGabor[5]);

// 返回一个面颊区域的Gabor滤波响应值
Mat CalcGaborRespInOneCheek(const vector<CvGabor*>& lGaborBank,
                          const Mat& grSrcImg,
                          const Rect& cheekRect);

// 眼袋
Mat CalcGaborRespInOneEyeBag(const vector<CvGabor*>& lGaborBank,
                          const Mat& grSrcImg,
                          const Rect& eyeBagRect);

// glabella，眉间，印堂
Mat CalcGaborRespOnGlab(const Mat& grSrcImg,
                        const Rect& glabRect);

// forehead，前额
Mat CalcGaborRespOnFh(const Mat& grSrcImg,
                      const Rect& fhRect);

// 鼻梁上半部，rhinion(keystone)
Mat CalcUpperNoseGaborResp(const vector<Point2i>& wrinkle_spline_curve,
                  const Rect& FaceContour_Rect,
                  const Mat& allWrinkle, Rect& nrect);
*/
//WrinkRespMap的大小和在原始影像坐标系中的位置由Face_Rect限定
void CalcGaborResp(const Mat& grSrcImg,
                   WrkRegGroup& wrkRegGroup,
                   Mat& WrinkRespMap);


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
