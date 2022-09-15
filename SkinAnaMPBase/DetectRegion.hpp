//
//  DetectRegion.hpp
//
//
/*
本模块。
 
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

/**********************************************************************************************

本函数构建皮肤区域的矢量版雏形。
***********************************************************************************************/

Mat contour2mask(int img_width, int img_height, const POLYGON& contours);


void ForgeSkinPolygon(const FaceInfo& faceInfo, POLYGON& skinPolygon);
//-------------------------------------------------------------------------------------------

void ForgeSkinMask(int img_width, int img_height,
                   const FaceInfo& faceInfo, Mat& outMask);


void ForgeMouthMask(int img_width, int img_height,
                    const FaceInfo& faceInfo, Mat& outMask);
//-------------------------------------------------------------------------------------------
void OverlayMaskOnImage(const Mat& srcImg, const Mat& mask,
                        const string& maskName,
                        const char* out_filename);

#endif /* end of DETECT_REGION_HPP */
