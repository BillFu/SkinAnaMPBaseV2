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

// !!!调用这个函数前，outMask必须进行过初始化，或者已有内容在里面！！！
void DrawContOnMask(int img_width, int img_height, const POLYGON& contours, Mat& outMask);

Mat ContourGroup2Mask(int img_width, int img_height, const POLYGON_GROUP& contoursGroup);


/**********************************************************************************************
本函数构建皮肤区域的矢量版雏形。
***********************************************************************************************/
void ForgeSkinPolygon(const FaceInfo& faceInfo, POLYGON& skinPolygon);
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
void ForgeTwoEyesFullMask(const FaceInfo& faceInfo, Mat& outEyesFullMask);

void ForgeNoseMask(const FaceInfo& faceInfo, Mat& outNoseMask);

// face mask below the eyes，在鼻子部位向上凸出，接近额头
//void ForgeLowFaceMask(const FaceInfo& faceInfo, Mat& outMask);
//-------------------------------------------------------------------------------------------
void OverlayMaskOnImage(const Mat& srcImg, const Mat& mask,
                        const string& maskName,
                        const char* out_filename);

#endif /* end of DETECT_REGION_HPP */
