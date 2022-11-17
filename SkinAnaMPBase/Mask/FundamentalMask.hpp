//
//  DetectRegion.hpp
//
//
/*
本模块目前构建基础性的Polygon、Mask，也提供辅助型的一些函数。
本模块为SkinFeatureMask奠定一些基础。
 
Author: Fu Xiaoqiang
Date:   2022/9/15
*/

#ifndef DETECT_REGION_HPP
#define DETECT_REGION_HPP

#include "opencv2/opencv.hpp"
#include "Common.hpp"

using namespace std;
using namespace cv;


//-------------------------------------------------------------------------------------------
void expanMask(const Mat& inMask, int expandSize, Mat& outMask);


Mat ContourGroup2Mask(int img_width, int img_height, const POLYGON_GROUP& contoursGroup);
Mat ContourGroup2Mask(Size imgS, const POLYGON_GROUP& contoursGroup);

/**********************************************************************************************
本函数构建皮肤区域的矢量版雏形。
***********************************************************************************************/
//void ForgeSkinPolygon(const FaceInfo& faceInfo, POLYGON& skinPolygon);

//void ForgeSkinPolygonV2(const FaceInfo& faceInfo, POLYGON& skinPolygon);

//-------------------------------------------------------------------------------------------

void ForgeSkinMask(const FaceInfo& faceInfo, Mat& outMask);

void ForgeMouthPolygon(const FaceInfo& faceInfo,
                       int& mouthWidth, int& mouthHeigh,
                       POLYGON& mouthPolygon);

//expanRatio: expansion width toward outside / half of mouth height
void ForgeMouthMask(const FaceInfo& faceInfo, float expanRatio, Mat& outMask);


// EeyeFullMask包含眼睛、眉毛、眼袋的大范围区域
void ForgeOneEyeFullMask(const FaceInfo& faceInfo, EyeID eyeID, Mat& outMask);


// the returned mask covers the two eyes, eyebows, the surrounding area, and with somewhat expansion
void ForgeEyesFullMask(const FaceInfo& faceInfo, Mat& outEyesFullMask);

void ForgeNoseMask(const FaceInfo& faceInfo, Mat& outNoseMask);

// face mask below the eyes，在鼻子部位向上凸出，接近额头
//void ForgeLowFaceMask(const FaceInfo& faceInfo, Mat& outMask);

//-------------------------------------------------------------------------------------------
//forge the polygon that start from the middle nose bone and extend downside like a bell
// until reach outside the face area.
void ForgeNoseBellPg(const FaceInfo& faceInfo,
                     POLYGON& noseBellPg);

void ForgeNoseBellMask(const FaceInfo& faceInfo, Mat& outNoseBellMask);

//-------------------------------------------------------------------------------------------


#endif /* end of DETECT_REGION_HPP */
