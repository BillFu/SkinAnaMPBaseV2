//
//  DetectRegion.hpp
//
//
/*
本模块构建SkinMaskßß。
 
Author: Fu Xiaoqiang
Date:   2022/10/8
*/

#ifndef SKIN_MAS_HPP
#define SKIN_MAS_HPP

#include "opencv2/opencv.hpp"
#include "Common.hpp"

#include "ForeheadMask.hpp"

using namespace std;
using namespace cv;


//-------------------------------------------------------------------------------------------

/**********************************************************************************************
本函数构建皮肤区域的矢量版雏形。
***********************************************************************************************/
void ForgeSkinPgV3(const FaceInfo& faceInfo, POLYGON& skinPolygon,
                        Point2i raisedFhCurve[NUM_PT_TOP_FH], // for debugging, output
                        int raisedPtIndices[NUM_PT_TOP_FH]    //for debugging, output
);

/**********************************************************************************************
本函数构建skinMask，挖掉眉毛、嘴唇、眼睛等区域。
***********************************************************************************************/
void ForgeSkinMaskV3(const FaceInfo& faceInfo,
                     const Mat& mouthMask,
                     const Mat& eyebrowMask,
                     const Mat& eyeMask,
                     const Mat& lowFaceMask,
                     Mat& outMask,
                     //---the following items are used for debugging, output
                     Point2i raisedFhCurve[NUM_PT_TOP_FH],
                     int raisedPtIndices[NUM_PT_TOP_FH]);

//-------------------------------------------------------------------------------------------

void ForgeSkinPgV4(const FaceInfo& faceInfo,
                   const Mat& lowFaceMask,
                   POLYGON& skinPolygon,
                        Point2i raisedFhCurve[NUM_PT_TOP_FH], // for debugging, output
                        int raisedPtIndices[NUM_PT_TOP_FH]    //for debugging, output
);

void ForgeSkinMaskV4(const FaceInfo& faceInfo,
                     const Mat& mouthMask,
                     const Mat& eyebrowMask,
                     const Mat& eyeMask,
                     const Mat& lowFaceMask,
                     Mat& outMask,
                     //---the following items are used for debugging, output
                     Point2i raisedFhCurve[NUM_PT_TOP_FH],
                     int raisedPtIndices[NUM_PT_TOP_FH]);

#endif /* end of SKIN_MAS_HPP */
