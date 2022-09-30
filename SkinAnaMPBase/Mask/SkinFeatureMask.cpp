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
#include "EyebowMask.hpp"
#include "ForeheadMask.hpp"
#include "../AnnoImage.hpp"

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

//-------------------------------------------------------------------------------------------

// 一揽子函数，生成各类Mask和它们的Anno Image
void ForgeMaskAnnoPack(const Mat& srcImage, const Mat& annoLmImage,
                       const fs::path& outDir, const string& fileNameBone,
                       const FaceInfo& faceInfo)
{
    cv::Size2i srcImgS = srcImage.size();
    
    Mat skinMask(srcImgS, CV_8UC1, cv::Scalar(0));
    string faceMaskImgFile = BuildOutImgFileName(outDir,
                             fileNameBone, "fc_");
    ForgeSkinMask(faceInfo, skinMask);
    OverlayMaskOnImage(annoLmImage, skinMask,
                        "face_contour", faceMaskImgFile.c_str());

    Mat mouthMask(srcImgS, CV_8UC1, cv::Scalar(0));
    string mouthMaskImgFile = BuildOutImgFileName(outDir,
                             fileNameBone, "mc_");
    float expanRatio = 0.3;
    ForgeMouthMask(faceInfo, expanRatio, mouthMask);
    OverlayMaskOnImage(srcImage, mouthMask,
                        "mouth_contour", mouthMaskImgFile.c_str());

    Mat eyebowsMask(srcImgS, CV_8UC1, cv::Scalar(0));
    string ebsMaskImgFile = BuildOutImgFileName(outDir,
                             fileNameBone, "ebc_");
    ForgeTwoEyebowsMask(faceInfo, eyebowsMask);
    OverlayMaskOnImage(annoLmImage, eyebowsMask,
                        "eyebows_contour", ebsMaskImgFile.c_str());

    Mat eyesFullMask(srcImgS, CV_8UC1, cv::Scalar(0));
    string eyeFullMaskImgFile = BuildOutImgFileName(outDir,
                             fileNameBone, "efc_");
    ForgeTwoEyesFullMask(faceInfo, eyesFullMask);
    OverlayMaskOnImage(annoLmImage, eyesFullMask,
                        "eye_full_contour", eyeFullMaskImgFile.c_str());

    //string fhMaskAnnoFile = config_json.at("ForeheadMaskImage");
    Mat fhMask(srcImgS, CV_8UC1, cv::Scalar(0));
    ForgeForeheadMask(faceInfo, fhMask);

    Mat annoImage2 = srcImage.clone();
    AnnoGeneralKeyPoints(annoImage2, faceInfo, true);
    //OverlayMaskOnImage(annoImage2, fhMask,
    //                    "forehead mask", fhMaskAnnoFile.c_str());

    //string noseMaskAnnoFile = config_json.at("NoseMaskImage");
    Mat noseMask(srcImgS, CV_8UC1, cv::Scalar(0));
    ForgeNoseMask(faceInfo, noseMask);

    //Mat combinedMask = noseMask | fhMask;
    //OverlayMaskOnImage(annoImage2, combinedMask,
    //                    "combined mask", noseMaskAnnoFile.c_str());

    //string fleMaskAnnoFile = config_json.at("FaceLowThEyeImage");
    Mat fleMask(srcImgS, CV_8UC1, cv::Scalar(0));
    ForgeFaceLowThEyeMask(faceInfo, fleMask);
    //OverlayMaskOnImage(annoImage2, fleMask,
    //                    "face low th eye", fleMaskAnnoFile.c_str());

    string poreMaskAnnoFile = BuildOutImgFileName(outDir,
                             fileNameBone, "pore_");
    Mat poreMask(srcImgS, CV_8UC1, cv::Scalar(0));
    ForgePoreMaskV2(faceInfo, fleMask, fhMask, eyesFullMask,
                    mouthMask, noseMask,
                    poreMask);
    OverlayMaskOnImage(annoImage2, poreMask,
                        "pore mask", poreMaskAnnoFile.c_str());
}
