//
//  DetectRegion.hpp
//
//
/*
本模块构建SkinMask。
 
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

//-------------------------------------------------------------------------------------------

// 新思路：先有一组关键点连接起来，构成脸部外轮廓的雏形，而后，若它们不在分割出的脸部Mask之内，则向内收缩，直到进入
// 分割出的脸部Mask之内。收缩的方向垂直于该点的法线，而法线方向是依据当前点的左右邻居连线而估计出来的。
// 目前我们的分割算法，在眼睛以下的脸部较为准确，在额头部位受到头发的干扰而降低了精准度。
// 因此，眼睛以下采用分割结果引导关键点收缩的思路，眼睛以上则要另谋出路。
void ForgeSkinPgV5(const FaceInfo& faceInfo,
                   const Mat& lowFaceMask,
                   POLYGON& skinPolygon,
                        Point2i raisedFhCurve[NUM_PT_TOP_FH], // for debugging, output
                        int raisedPtIndices[NUM_PT_TOP_FH]    //for debugging, output
);

void ForgeSkinMaskV5(const FaceInfo& faceInfo,
                     const Mat& mouthMask,
                     const Mat& eyebrowMask,
                     const Mat& eyeMask,
                     const Mat& lowFaceMask,
                     Mat& outMask,
                     //---the following items are used for debugging, output
                     Point2i raisedFhCurve[NUM_PT_TOP_FH],
                     int raisedPtIndices[NUM_PT_TOP_FH]);

void TestMaskV5(const FaceInfo& faceInfo,
            const Mat& lowFaceMask,
            const Mat& srcImage);

#endif /* end of SKIN_MAS_HPP */
