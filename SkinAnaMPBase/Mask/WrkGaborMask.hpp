//
//  WrkGaborMask.hpp
//
//
/*
本模块构建用于Gabor滤波来进行皱纹提取的各个小区域。
 
Author: Fu Xiaoqiang
Date:   2022/11/2
*/

#ifndef WRK_GABOR_MASK_HPP
#define WRK_GABOR_MASK_HPP

#include "opencv2/opencv.hpp"

using namespace std;
using namespace cv;

#include "Common.hpp"

Mat ForgeGlabellaMask(const FaceInfo& faceInfo);

void ForgeRCrowFeetPg(const FaceInfo& faceInfo, POLYGON& outPolygon);
Mat ForgeRCrowFeetMask(const FaceInfo& faceInfo);

void ForgeLCrowFeetPg(const FaceInfo& faceInfo, POLYGON& outPolygon);
Mat ForgeLCrowFeetMask(const FaceInfo& faceInfo);

//------------------------------------------------------------------
// 生成皱纹检测的各个检测区（只针对正脸）----新版本
void ForgeWrkTenRegs(const FaceInfo& faceInfo,
                     const Mat& fbBiLab, WrkRegGroup& wrkRegGroup);

void ForgeWrkTenRegs(const Mat& annoLmImage, const FaceInfo& faceInfo,
                     const Mat& fbBiLab, WrkRegGroup& wrkRegGroup);

#endif /* end of WRK_GABOR_MASK_HPP */
