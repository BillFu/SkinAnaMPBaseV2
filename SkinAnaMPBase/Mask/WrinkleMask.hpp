//
//  WrinkleMask.hpp
//
//
/*
本模块。
 
Author: Fu Xiaoqiang
Date:   2022/11/2
*/

#ifndef WRINKLE_MASK_HPP
#define WRINKLE_MASK_HPP

#include "opencv2/opencv.hpp"

using namespace std;
using namespace cv;

#include "Common.hpp"

Mat ForgeGlabellaMask(const FaceInfo& faceInfo);

//------------------------------------------------------------------
// 生成皱纹检测的各个检测区（只针对正脸）----新版本
void ForgeWrkTenRegs(const FaceInfo& faceInfo,
                     const Mat& fbBiLab, WrkRegGroup& wrkRegGroup);

void ForgeWrkTenRegsDebug(const Mat& annoLmImage, const FaceInfo& faceInfo,
                     const Mat& fbBiLab, WrkRegGroup& wrkRegGroup);

#endif /* end of WRINKLE_MASK_HPP */
