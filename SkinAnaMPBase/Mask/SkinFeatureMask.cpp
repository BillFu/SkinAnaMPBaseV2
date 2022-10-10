//
//  SkinFeatureMask.cpp

/*******************************************************************************
Author: Fu Xiaoqiang
Date:   2022/9/23

********************************************************************************/

#include "SkinFeatureMask.hpp"
#include "FundamentalMask.hpp"

#include <iostream>
#include <string>
#include <fstream>
#include <sstream>
#include <vector>

#include <filesystem>
#include <algorithm>

#include "../Utils.hpp"
#include "EyebrowMaskV2.hpp"
#include "ForeheadMask.hpp"
#include "LowerFaceMask.hpp"
#include "../AnnoImage.hpp"
#include "../FaceBgSeg/FaceBgSegV2.hpp"
#include "../BSpline/ParametricBSpline.hpp"
#include "SkinMask.hpp"


namespace fs = std::filesystem;

/**********************************************************************************************
本函数构建Pore的矢量版Mask雏形。
***********************************************************************************************/
void ForgePoreMask(const FaceInfo& faceInfo,
                   const Mat& fullFaceMask,
                   const Mat& eyeFullMask,  // cover the eyes and eyebows and the surrounding nearby area
                   const Mat& mouthMask, // the enlarged mouth mask
                   Mat& outPoreMask)
{
    outPoreMask = fullFaceMask & (~eyeFullMask) & (~mouthMask);
}


void ForgePoreMaskV2(const FaceInfo& faceInfo,
                   const Mat& faceLowMask,  // lower than eyes
                   const Mat& foreheadMask,
                   const Mat& eyeFullMask,  // cover the eyes and eyebows and the surrounding nearby area
                   const Mat& mouthMask, // the enlarged mouth mask
                   const Mat& noseMask,
                   Mat& outPoreMask)
{
    Mat outMask = faceLowMask | foreheadMask | noseMask ;
    outPoreMask = outMask & (~eyeFullMask) & (~mouthMask);
    
    //int expandSize = 20;
    //expanMask(outMask, expandSize, outPoreMask);
}

void ForgePoreMaskV3(const FaceInfo& faceInfo,
                   const Mat& faceLowMask,  // lower than eyes
                   const Mat& expFhMask,
                   const Mat& eyeFullMask,  // cover the eyes and eyebows and the surrounding nearby area
                   const Mat& mouthMask, // the enlarged mouth mask
                   Mat& outPoreMask)
{
    Mat outMask = faceLowMask | expFhMask ;
    outPoreMask = outMask & (~eyeFullMask) & (~mouthMask);
    
    //int expandSize = 20;
    //expanMask(outMask, expandSize, outPoreMask);
}

//-------------------------------------------------------------------------------------------

/**********************************************************************************************
本函数构建Wrinkle的矢量版Mask雏形。
***********************************************************************************************/
void ForgeWrinkleMask(const FaceInfo& faceInfo,
                      const Mat& faceLowMask,  // lower than eyes
                      const Mat& expFhMask,
                      const Mat& eyeFullMask,  // cover the eyes and eyebows and the surrounding nearby area
                      const Mat& noseBellMask,
                      Mat& outWrkMask)
{
    outWrkMask = faceLowMask | expFhMask ;
    outWrkMask = outWrkMask & (~eyeFullMask) & (~noseBellMask);
}

//-------------------------------------------------------------------------------------------

// 一揽子函数，生成各类Mask和它们的Anno Image
void ForgeMaskAnnoPackDebug(const Mat& srcImage, const Mat& annoLmImage,
                       const fs::path& outDir, const string& fileNameBone,
                       const FaceInfo& faceInfo,
                       const FaceSegResult& segResult)
{
    cv::Size2i srcImgS = srcImage.size();
    
    // 0 for bg, 255 for face, in source space.
    //Mat fbBiLab = FaceBgSegmentor::CalcFaceBgBiLabel(segResult);
    Mat fbBiLab = FaceBgSegmentor::CalcFBBiLabExBeard(segResult);

    Mat mouthMask(srcImgS, CV_8UC1, cv::Scalar(0));
    string mouthMaskImgFile = BuildOutImgFileName(outDir,
                             fileNameBone, "mc_");
    float expanRatio = 0.3;
    ForgeMouthMask(faceInfo, expanRatio, mouthMask);
    
    Mat eyebrowsMask(srcImgS, CV_8UC1, cv::Scalar(0));
    string ebsMaskImgFile = BuildOutImgFileName(outDir,
                             fileNameBone, "ebc_");
    ForgeBrowsMask(faceInfo, eyebrowsMask);
    
    Mat eyesMask(srcImgS, CV_8UC1, cv::Scalar(0));
    string eyeMaskImgFile = BuildOutImgFileName(outDir,
                             fileNameBone, "eye_");
    ForgeEyesMask(faceInfo, segResult, eyesMask);
    OverlayMaskOnImage(annoLmImage, eyesMask,
                        "eyes_mask", eyeMaskImgFile.c_str());

    Mat eyesFullMask(srcImgS, CV_8UC1, cv::Scalar(0));
    string eyeFullMaskImgFile = BuildOutImgFileName(outDir,
                             fileNameBone, "efc_");
    ForgeEyesFullMask(faceInfo, eyesFullMask);
    
    string noseBellMaskFile = BuildOutImgFileName(outDir,
                             fileNameBone, "nb_");
    Mat noseBellMask(srcImgS, CV_8UC1, cv::Scalar(0));
    ForgeNoseBellMask(faceInfo, noseBellMask);
    
    Mat lowFaceMask(srcImgS, CV_8UC1, cv::Scalar(0));
    ForgeLowerFaceMask(segResult, fbBiLab, lowFaceMask);
    
    string expFhMaskFile = BuildOutImgFileName(outDir,
                             fileNameBone, "efh_");
    Mat expFhMask(srcImgS, CV_8UC1, cv::Scalar(0));
    ForgeExpFhMask(faceInfo, fbBiLab, expFhMask);
    
    string poreMaskAnnoFile = BuildOutImgFileName(outDir,
                             fileNameBone, "pore_");
    Mat poreMask(srcImgS, CV_8UC1, cv::Scalar(0));
    ForgePoreMaskV3(faceInfo, lowFaceMask, expFhMask, eyesFullMask,
                    mouthMask, poreMask);
    OverlayMaskOnImage(annoLmImage, poreMask,
                        "pore mask", poreMaskAnnoFile.c_str());
    
    string wrkMaskAnnoFile = BuildOutImgFileName(outDir,
                             fileNameBone, "wrk_");
    Mat wrkMask(srcImgS, CV_8UC1, cv::Scalar(0));
    ForgeWrinkleMask(faceInfo, lowFaceMask, expFhMask, eyesFullMask,
                    noseBellMask, wrkMask);
    OverlayMaskOnImage(annoLmImage, wrkMask,
                        "wrinkle mask", wrkMaskAnnoFile.c_str());
    
    Mat skinMask(srcImgS, CV_8UC1, cv::Scalar(0));
    string skinMaskImgFile = BuildOutImgFileName(outDir,
                             fileNameBone, "skinMask_");
    
    Point2i raisedFhCurve[NUM_PT_TOP_FH];
    int raisedPtIndices[NUM_PT_TOP_FH];
    
    //TestMaskV5(faceInfo, fbBiLab, srcImage);
    
    ForgeSkinMaskV5(faceInfo, mouthMask,
                    eyebrowsMask, eyesMask,
                    lowFaceMask,
                    skinMask, raisedFhCurve, raisedPtIndices);
    
    cv::Scalar yellow(0, 255, 255); // (B, G, R)
    cv::Scalar blue(255, 0, 0);
    for(int i = 0; i < NUM_PT_TOP_FH; i++)
    {
        // cv::Point center(faceInfo.lm_2d[i].x, faceInfo.lm_2d[i].y);
        cv::circle(annoLmImage, raisedFhCurve[i], 5, blue, cv::FILLED);
        
        string caption = to_string(raisedPtIndices[i]) + "r";
        cv::putText(annoLmImage, caption, raisedFhCurve[i],
                        FONT_HERSHEY_SIMPLEX, 0.5, blue, 1);
    }
    
    OverlayMaskOnImage(annoLmImage, skinMask,
                       "Skin Mask", skinMaskImgFile.c_str());
    
}

//-------------------------------------------------------------------------------------------

/**********************************************************************************************
本函数构建skinMask，挖掉眉毛、嘴唇、眼睛等区域。
***********************************************************************************************/
/*
void ForgeSkinMaskV2(const FaceInfo& faceInfo,
                     const Mat& mouthMask,
                     const Mat& eyebrowMask,
                     const Mat& eyeMask,
                     Mat& outMask)
{
    POLYGON coarsePolygon, refinedPolygon;

    ForgeSkinPolygonV2(faceInfo, coarsePolygon);
    
    int csNumPoint = 200;
    CloseSmoothPolygon(coarsePolygon, csNumPoint, refinedPolygon);

    DrawContOnMask(faceInfo.imgWidth, faceInfo.imgHeight, refinedPolygon, outMask);
    
    outMask = outMask & (~mouthMask) & (~eyebrowMask) & (~eyeMask);
}
*/

//-------------------------------------------------------------------------------------------
/**********************************************************************************************
本函数构建各类综合性的Mask。
***********************************************************************************************/
// 一揽子函数，生成各类Mask和它们的Anno Image
// 简洁版
void ForgeMaskAnnoPackV2(const Mat& srcImage,
                       const fs::path& outDir, const string& fileNameBone,
                       const FaceInfo& faceInfo,
                       const FaceSegResult& segResult)
{
    cv::Size2i srcImgS = srcImage.size();
    
    // 0 for bg, 255 for face, in source space.
    //Mat fbBiLab = FaceBgSegmentor::CalcFaceBgBiLabel(segResult);
    Mat fbBiLab = FaceBgSegmentor::CalcFBBiLabExBeard(segResult);
    
    Mat mouthMask(srcImgS, CV_8UC1, cv::Scalar(0));
    string mouthMaskImgFile = BuildOutImgFileName(outDir,
                             fileNameBone, "mc_");
    float expanRatio = 0.3;
    ForgeMouthMask(faceInfo, expanRatio, mouthMask);
    
    Mat eyebrowsMask(srcImgS, CV_8UC1, cv::Scalar(0));
    ForgeBrowsMask(faceInfo, eyebrowsMask);
    
    Mat eyesMask(srcImgS, CV_8UC1, cv::Scalar(0));
    ForgeEyesMask(faceInfo, segResult, eyesMask);
    
    Mat eyesFullMask(srcImgS, CV_8UC1, cv::Scalar(0));
    string eyeFullMaskImgFile = BuildOutImgFileName(outDir,
                             fileNameBone, "efc_");
    ForgeEyesFullMask(faceInfo, eyesFullMask);
    
    string noseBellMaskFile = BuildOutImgFileName(outDir,
                             fileNameBone, "nb_");
    Mat noseBellMask(srcImgS, CV_8UC1, cv::Scalar(0));
    ForgeNoseBellMask(faceInfo, noseBellMask);

    Mat lowFaceMask(srcImgS, CV_8UC1, cv::Scalar(0));
    ForgeLowerFaceMask(segResult, fbBiLab, lowFaceMask);
    
    string expFhMaskFile = BuildOutImgFileName(outDir,
                             fileNameBone, "efh_");
    Mat expFhMask(srcImgS, CV_8UC1, cv::Scalar(0));
    ForgeExpFhMask(faceInfo, fbBiLab, expFhMask);
    //AnnoGenKeyPoints(annoImage2, faceInfo, true);
    
    string poreMaskAnnoFile = BuildOutImgFileName(outDir,
                             fileNameBone, "pore_");
    Mat poreMask(srcImgS, CV_8UC1, cv::Scalar(0));
    ForgePoreMaskV3(faceInfo, lowFaceMask, expFhMask, eyesFullMask,
                    mouthMask, poreMask);
    OverlayMaskOnImage(srcImage, poreMask,
                        "pore mask", poreMaskAnnoFile.c_str());
    
    string wrkMaskAnnoFile = BuildOutImgFileName(outDir,
                             fileNameBone, "wrk_");
    Mat wrkMask(srcImgS, CV_8UC1, cv::Scalar(0));
    ForgeWrinkleMask(faceInfo, lowFaceMask, expFhMask, eyesFullMask,
                    noseBellMask, wrkMask);
    OverlayMaskOnImage(srcImage, wrkMask,
                        "wrinkle mask", wrkMaskAnnoFile.c_str());
    
    Mat skinMask(srcImgS, CV_8UC1, cv::Scalar(0));
    string skinMaskImgFile = BuildOutImgFileName(outDir,
                             fileNameBone, "skinMask_");
    
    Point2i raisedFhCurve[NUM_PT_TOP_FH];
    int raisedPtIndices[NUM_PT_TOP_FH];
    ForgeSkinMaskV3(faceInfo, mouthMask,
                    eyebrowsMask, eyesMask,
                    lowFaceMask, skinMask,
                    raisedFhCurve, raisedPtIndices);
    OverlayMaskOnImage(srcImage, skinMask,
                       "Skin Mask", skinMaskImgFile.c_str());
}

