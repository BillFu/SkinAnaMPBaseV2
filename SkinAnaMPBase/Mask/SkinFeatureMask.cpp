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
#include "../AnnoImage.hpp"
#include "../FaceBgSeg/FaceBgSegV2.hpp"
#include "../BSpline/ParametricBSpline.hpp"

#include "EyebrowMaskV8.hpp"
#include "ForeheadMask.hpp"
#include "LowerFaceMask.hpp"
#include "SkinMask.hpp"
#include "WrkGaborMask.hpp"


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
                     Mat& poreMask)
{
    poreMask = faceLowMask | expFhMask ;
    poreMask = poreMask & (~eyeFullMask) & (~mouthMask);
}

//-------------------------------------------------------------------------------------------

/**********************************************************************************************
本函数构建Wrinkle的矢量版Mask雏形，用于Frangi滤波。
***********************************************************************************************/
void ForgeWrkFrgiMask(const FaceInfo& faceInfo,
                      const Mat& faceLowMask,  // lower than eyes
                      const Mat& expFhMask,
                      const Mat& eyeFullMask,  // cover the eyes and eyebows and the surrounding nearby area
                      const Mat& noseBellMask,
                      Mat& wrkFrgiMask)
{
    wrkFrgiMask = faceLowMask | expFhMask ;
    wrkFrgiMask = wrkFrgiMask & (~eyeFullMask) & (~noseBellMask);
}



//-------------------------------------------------------------------------------------------

// 一揽子函数，生成各类Mask和它们的Anno Image
void ForgeDetRegPack(const Mat& srcImage, const Mat& annoLmImage,
                       const fs::path& outDir, 
                       const FaceInfo& faceInfo,
                       const FaceSegRst& segResult,
                       DetRegPack& detRegPack)
{
    cv::Size2i srcImgS = srcImage.size();
    
    // 0 for bg, 255 for face, in source space.
    //Mat fbBiLab = FaceBgSegmentor::CalcFaceBgBiLabel(segResult);
    Mat fbBiLab = FaceBgSegmentor::CalcFBBiLabExBeard(segResult);

    Mat mouthMask(srcImgS, CV_8UC1, cv::Scalar(0));
    float expanRatio = 0.3;
    ForgeMouthMask(faceInfo, expanRatio, mouthMask);
    
#ifdef TEST_RUN
    string mouthMaskImgFile = BuildOutImgFNV2(outDir, "mouth_ct.png");
    OverlayMaskOnImage(annoLmImage, mouthMask,
                        "mouth_mask", mouthMaskImgFile.c_str());
#endif
    
    Mat browsMask(srcImgS, CV_8UC1, cv::Scalar(0));
    ForgeBrowsMask(faceInfo, segResult, browsMask);
    
#ifdef TEST_RUN
    string ebsMaskImgFile = BuildOutImgFNV2(outDir, "brow_ct.png");
    OverlayMaskOnImage(annoLmImage, browsMask,
                        "mouth_mask", ebsMaskImgFile.c_str());
#endif
    
    Mat eyesMask(srcImgS, CV_8UC1, cv::Scalar(0));
    ForgeEyesMask(srcImage, faceInfo, segResult, eyesMask,
                  detRegPack.lEyeReg, detRegPack.rEyeReg);
    detRegPack.eyesMask = eyesMask;

#ifdef TEST_RUN2
    string eyeMaskImgFile = BuildOutImgFNV2(outDir, "eye_mask.png");
    AnnoMaskOnImage(annoLmImage, eyesMask,
                        "eyes_mask", eyeMaskImgFile.c_str());
#endif

    Mat eyesFullMask(srcImgS, CV_8UC1, cv::Scalar(0));
    ForgeEyesFullMask(faceInfo, eyesFullMask);
    detRegPack.eyesFullMask = eyesFullMask;
    
#ifdef TEST_RUN2
    string eyeFullMaskImgFile = BuildOutImgFNV2(outDir, "eye_full_mask.png");
    AnnoMaskOnImage(annoLmImage, eyesFullMask,
                        "eyes_full_mask", eyeFullMaskImgFile.c_str());
#endif

    Mat noseBellMask(srcImgS, CV_8UC1, cv::Scalar(0));
    ForgeNoseBellMask(faceInfo, noseBellMask);
    
#ifdef TEST_RUN2
    string noseBellMaskFile = BuildOutImgFNV2(outDir, "nose_bell.png");
    AnnoMaskOnImage(annoLmImage, noseBellMask,
                        "nose_bell_mask", noseBellMaskFile.c_str());
#endif
    
    Mat lowFaceMask(srcImgS, CV_8UC1, cv::Scalar(0));
    ForgeLowerFaceMask(segResult, fbBiLab, lowFaceMask);
    
    Mat expFhMask(srcImgS, CV_8UC1, cv::Scalar(0));
    ForgeExpFhMask(faceInfo, fbBiLab, expFhMask);
    
#ifdef TEST_RUN
    string expFhMaskFile = BuildOutImgFNV2(outDir, "ext_forehead.png");
    OverlayMaskOnImage(annoLmImage, expFhMask,
                        "ext_forehead", expFhMaskFile.c_str());
#endif
    
    ForgeWrkTenRegs(annoLmImage, faceInfo, fbBiLab, detRegPack);

    ForgePoreMaskV3(faceInfo, lowFaceMask, expFhMask, eyesFullMask,
                    mouthMask, detRegPack.poreMask);
    
#ifdef TEST_RUN2
    string poreMaskAnnoFile = BuildOutImgFNV2(outDir, "PoreMask.png");
    AnnoMaskOnImage(annoLmImage, detRegPack.poreMask,
                        "pore mask", poreMaskAnnoFile.c_str());
#endif
    
    //Mat wrkMask(srcImgS, CV_8UC1, cv::Scalar(0));
    ForgeWrkFrgiMask(faceInfo, lowFaceMask, expFhMask, eyesFullMask,
                    noseBellMask, detRegPack.wrkFrgiMask);
    
#ifdef TEST_RUN2
    string wrkMaskAnnoFile = BuildOutImgFNV2(outDir, "WrkFrgiMask.png");
    AnnoMaskOnImage(annoLmImage, detRegPack.wrkFrgiMask,
                        "wrinkle Frgi mask", wrkMaskAnnoFile.c_str());
#endif
    
    Mat skinMask(srcImgS, CV_8UC1, cv::Scalar(0));
    
    Point2i raisedFhCurve[NUM_PT_TOP_FH];
    int raisedPtIndices[NUM_PT_TOP_FH];
    
    ForgeSkinMaskV5(faceInfo, mouthMask,
                    browsMask, eyesMask,
                    lowFaceMask,
                    skinMask, raisedFhCurve, raisedPtIndices);
    
    /*
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
    */
    
#ifdef TEST_RUN2
    string skinMaskImgFile = BuildOutImgFNV2(outDir, "SkinMask.png");
    AnnoMaskOnImage(annoLmImage, skinMask,
                       "Skin Mask", skinMaskImgFile.c_str());
#endif
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
