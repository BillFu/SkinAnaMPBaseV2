//
//  SkinFeatureMask.hpp
//
//
/*
本模块目前构建皮肤各类特征的检测区域的矢量版雏形。
 
Author: Fu Xiaoqiang
Date:   2022/9/23
*/

#ifndef SKIN_FEATURE_MASK_HPP
#define SKIN_FEATURE_MASK_HPP

#include "opencv2/opencv.hpp"
#include "../Common.hpp"

using namespace std;
using namespace cv;

#include <filesystem>
#include <algorithm>

namespace fs = std::filesystem;

//-------------------------------------------------------------------------------------------

/**********************************************************************************************
本函数构建Pore的矢量版Mask雏形。
***********************************************************************************************/
void ForgePoreMask(const FaceInfo& faceInfo,
                   const Mat& fullFaceMask,
                   const Mat& eyeFullMask,  // cover the eyes and eyebows and the surrounding nearby area
                   const Mat& mouthMask, // the enlarged mouth mask
                   Mat& outPoreMask);

void ForgePoreMaskV2(const FaceInfo& faceInfo,
                   const Mat& faceLowMask,  // lower than eyes
                   const Mat& foreheadMask,
                   const Mat& eyeFullMask,  // cover the eyes and eyebows and the surrounding nearby area
                   const Mat& mouthMask, // the enlarged mouth mask
                   const Mat& noseMask,
                   Mat& outPoreMask);

void ForgePoreMaskV3(const FaceInfo& faceInfo,
                     const Mat& faceLowMask,  // lower than eyes
                     const Mat& expFhMask,
                     const Mat& eyeFullMask,  // cover the eyes and eyebows and the surrounding nearby area
                     const Mat& mouthMask, // the enlarged mouth mask
                     Mat& poreMask);
//-------------------------------------------------------------------------------------------

/**********************************************************************************************
本函数构建Wrinkle的矢量版Mask雏形，用于Frangi滤波。
***********************************************************************************************/
void ForgeWrkFrgiMask(const FaceInfo& faceInfo,
                      const Mat& faceLowMask,  // lower than eyes
                      const Mat& expFhMask,
                      const Mat& eyeFullMask,  // cover the eyes and eyebows and the surrounding nearby area
                      //const Mat& noseMask,
                      const Mat& noseBellMask,
                      Mat& wrkFrgiMask);

//-------------------------------------------------------------------------------------------

// 一揽子函数，生成各类Mask和它们的Anno Image
// 用TEST_RUN打印出许多辅助性结果和信息
void ForgeDetRegPack(const Mat& srcImage, const Mat& annoLmImage,
                     const fs::path& outDir, const FaceInfo& faceInfo,
                     const FaceSegRst& segResult,
                     DetRegPack& detRegPack);


// 一揽子函数，生成各类Mask和它们的Anno Image
// 简洁版
void ForgeMaskAnnoPackV2(const Mat& srcImage,
                       const fs::path& outDir, const string& fileNameBone,
                       const FaceInfo& faceInfo,
                         const FaceSegRst& segResult);

//-------------------------------------------------------------------------------------------
/*
void ForgeSkinMaskV2(const FaceInfo& faceInfo,
                     const Mat& mouthMask,
                     const Mat& eyebrowMask,
                     const Mat& eyeMask,
                     Mat& outMask);
*/

#endif /* end of SKIN_FEATURE_MASK_HPP */
