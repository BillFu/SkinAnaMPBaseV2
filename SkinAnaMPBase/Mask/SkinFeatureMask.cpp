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
#include "EyebrowMask.hpp"
#include "ForeheadMask.hpp"
#include "LowerFaceMask.hpp"
#include "../AnnoImage.hpp"
#include "../FaceBgSeg/FaceBgSeg.hpp"
#include "../BSpline/ParametricBSpline.hpp"


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
    
    /*
    Mat skinMask(srcImgS, CV_8UC1, cv::Scalar(0));
    string faceMaskImgFile = BuildOutImgFileName(outDir,
                             fileNameBone, "fc_");
    ForgeSkinMask(faceInfo, skinMask);
    //OverlayMaskOnImage(annoLmImage, skinMask,
     //                   "face_contour", faceMaskImgFile.c_str());
    */

    Mat mouthMask(srcImgS, CV_8UC1, cv::Scalar(0));
    string mouthMaskImgFile = BuildOutImgFileName(outDir,
                             fileNameBone, "mc_");
    float expanRatio = 0.3;
    ForgeMouthMask(faceInfo, expanRatio, mouthMask);
    //OverlayMaskOnImage(srcImage, mouthMask,
     //               "mouth_contour", mouthMaskImgFile.c_str());

    Mat eyebrowsMask(srcImgS, CV_8UC1, cv::Scalar(0));
    string ebsMaskImgFile = BuildOutImgFileName(outDir,
                             fileNameBone, "ebc_");
    ForgeEyebrowsMask(faceInfo, eyebrowsMask);
    //OverlayMaskOnImage(annoLmImage, eyebowsMask,
    //                    "eyebows_contour", ebsMaskImgFile.c_str());

    Mat eyesMask(srcImgS, CV_8UC1, cv::Scalar(0));
    string eyeMaskImgFile = BuildOutImgFileName(outDir,
                             fileNameBone, "eye_");
    ForgeEyesMask(faceInfo, eyesMask);
    OverlayMaskOnImage(annoLmImage, eyesMask,
                        "eyes_mask", eyeMaskImgFile.c_str());

    Mat eyesFullMask(srcImgS, CV_8UC1, cv::Scalar(0));
    string eyeFullMaskImgFile = BuildOutImgFileName(outDir,
                             fileNameBone, "efc_");
    ForgeEyesFullMask(faceInfo, eyesFullMask);
    //OverlayMaskOnImage(annoLmImage, eyesFullMask,
    //                    "eye_full_contour", eyeFullMaskImgFile.c_str());

    Mat annoImage2 = srcImage.clone();
    AnnoGenKeyPoints(annoImage2, faceInfo, true);
    
    string noseBellMaskFile = BuildOutImgFileName(outDir,
                             fileNameBone, "nb_");
    Mat noseBellMask(srcImgS, CV_8UC1, cv::Scalar(0));
    ForgeNoseBellMask(faceInfo, noseBellMask);
    //OverlayMaskOnImage(annoImage2, noseBellMask,
    //                    "nose bell mask", noseBellMaskFile.c_str());
    

    //string fleMaskAnnoFile = config_json.at("FaceLowThEyeImage");
    Mat lowFaceMask(srcImgS, CV_8UC1, cv::Scalar(0));
    ForgeLowerFaceMask(segResult, fbBiLab, lowFaceMask);
    //OverlayMaskOnImage(annoImage2, fleMask,
    //                    "face low th eye", fleMaskAnnoFile.c_str());

    string expFhMaskFile = BuildOutImgFileName(outDir,
                             fileNameBone, "efh_");
    Mat expFhMask(srcImgS, CV_8UC1, cv::Scalar(0));
    ForgeExpFhMask(faceInfo, fbBiLab, expFhMask);
    AnnoGenKeyPoints(annoImage2, faceInfo, true);
    //OverlayMaskOnImage(annoImage2, expFhMask,
    //                    "expand Fh mask", expFhMaskFile.c_str());

    string poreMaskAnnoFile = BuildOutImgFileName(outDir,
                             fileNameBone, "pore_");
    Mat poreMask(srcImgS, CV_8UC1, cv::Scalar(0));
    ForgePoreMaskV3(faceInfo, lowFaceMask, expFhMask, eyesFullMask,
                    mouthMask, poreMask);
    OverlayMaskOnImage(annoImage2, poreMask,
                        "pore mask", poreMaskAnnoFile.c_str());
    
    string wrkMaskAnnoFile = BuildOutImgFileName(outDir,
                             fileNameBone, "wrk_");
    Mat wrkMask(srcImgS, CV_8UC1, cv::Scalar(0));
    ForgeWrinkleMask(faceInfo, lowFaceMask, expFhMask, eyesFullMask,
                    noseBellMask, wrkMask);
    OverlayMaskOnImage(annoImage2, wrkMask,
                        "wrinkle mask", wrkMaskAnnoFile.c_str());
    
    Mat skinMask(srcImgS, CV_8UC1, cv::Scalar(0));
    string skinMaskImgFile = BuildOutImgFileName(outDir,
                             fileNameBone, "skinMask_");
    
    ForgeSkinMaskV2(faceInfo, mouthMask,
                    eyebrowsMask, eyesMask,
                    skinMask);
    OverlayMaskOnImage(annoLmImage, skinMask,
                       "Skin Mask", skinMaskImgFile.c_str());
}

//-------------------------------------------------------------------------------------------

/**********************************************************************************************
本函数构建skinMask，挖掉眉毛、嘴唇、眼睛等区域。
***********************************************************************************************/
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
    ForgeEyebrowsMask(faceInfo, eyebrowsMask);
    
    Mat eyesMask(srcImgS, CV_8UC1, cv::Scalar(0));
    ForgeEyesMask(faceInfo, eyesMask);
    
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
    ForgeSkinMaskV2(faceInfo, mouthMask,
                    eyebrowsMask, eyesMask,
                    skinMask);
    OverlayMaskOnImage(srcImage, skinMask,
                       "Skin Mask", skinMaskImgFile.c_str());
}

