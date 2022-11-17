//
//  WrkGaborMask.cpp

/*******************************************************************************
本模块构建通过Gabor滤波来进行皱纹提取的各个小区域。
  
Author: Fu Xiaoqiang
Date:   2022/11/2
********************************************************************************/
#include <algorithm>
#include "../Geometry.hpp"
#include "../Utils.hpp"
#include "../AnnoImage.hpp"
#include "../polyitems_fit.hpp"
#include "../BSpline/ParametricBSpline.hpp"

#include "WrkGaborMask.hpp"
// #include "FundamentalMask.hpp"
#include "ForeheadMask.hpp"

//-------------------------------------------------------------------------------
void ForgeFhPgForWrk(const FaceInfo& faceInfo, POLYGON& outPolygon)
{
    // 点的索引针对468个general landmark而言
    /*
    int fhPtIndices[] = { // 顺时针计数
        67*, 67r, 109r, 10r, 338r, 297r, 297*,  // up line
        333, 334, 296, 336, 9, 107, 66, 105, 104  // bottom line
            ---------- 285, 8, 55 --------- expanded alternative sub-path
    };
    67*: 在67和103之间插值出来的。
    297*: 在297和332之间插值出来的。
    67r, 109r, 10r, 338r, 297r: 这5个点是原来点抬高后的版本。
    */
    
    // asterisk, 星号
    Point2i pt67a = IpGLmPtWithPair(faceInfo, 67, 103, 0.60);
    Point2i pt297a = IpGLmPtWithPair(faceInfo, 297, 332, 0.60);
    
    Point2i raisedFhPts[NUM_PT_TOP_FH];
    int raisedPtIndices[NUM_PT_TOP_FH];
    RaiseupFhCurve(faceInfo.lm_2d, raisedFhPts, raisedPtIndices, 0.7);

    Point2i pt67r = raisedFhPts[2];
    Point2i pt109r = raisedFhPts[3];
    Point2i pt10r = raisedFhPts[4];
    Point2i pt338r = raisedFhPts[5];
    Point2i pt297r = raisedFhPts[6];
    
    outPolygon.push_back(pt67a);
    outPolygon.push_back(pt67r);
    outPolygon.push_back(pt109r);
    outPolygon.push_back(pt10r);
    outPolygon.push_back(pt338r);
    outPolygon.push_back(pt297r);
    outPolygon.push_back(pt297a);

    //int botLinePts[] = {333, 334*, 296*, 336, 285, 8, 55, 107, 66*, 105*, 104};
    outPolygon.push_back(getPtOnGLm(faceInfo, 333));

    Point2i pt334a = IpGLmPtWithPair(faceInfo, 334, 333, 0.4);
    outPolygon.push_back(pt334a);
        
    Point2i pt296a = IpGLmPtWithPair(faceInfo, 296, 299, 0.25);
    outPolygon.push_back(pt296a);
    
    Point2i pt336a = IpGLmPtWithPair(faceInfo, 336, 337, 0.25);
    outPolygon.push_back(pt336a);
    
    Point2i pt9a = IpGLmPtWithPair(faceInfo, 9, 151, 0.25);
    outPolygon.push_back(pt9a);

    Point2i pt107a = IpGLmPtWithPair(faceInfo, 107, 108, 0.25);
    outPolygon.push_back(pt107a);
    
    Point2i pt66a = IpGLmPtWithPair(faceInfo, 66, 69, 0.25);
    outPolygon.push_back(pt66a);
    
    outPolygon.push_back(getPtOnGLm(faceInfo, 104));
}

void ForgeFhMaskForWrk(const FaceInfo& faceInfo, const Mat& fbBiLab, Mat& outMask)
{
    POLYGON coarsePolygon, refinedPolygon;

    ForgeFhPgForWrk(faceInfo, coarsePolygon);
    
    int csNumPoint = 80;
    DenseSmoothPolygon(coarsePolygon, csNumPoint, refinedPolygon);

    outMask = Mat(faceInfo.srcImgS, CV_8UC1, Scalar(0));
    DrawContOnMask(refinedPolygon, outMask);
    
    outMask = outMask & fbBiLab;
}
//-------------------------------------------------------------------------------

void ForgeGlabellPg(const FaceInfo& faceInfo, POLYGON& outPolygon)
{
    // 点的索引针对468个general landmark而言
    
    // middle and top point, 逆时针计数
    Point2i pt1 = IpGLmPtWithPair(faceInfo, 151, 9, 0.25);
    outPolygon.push_back(pt1);
    
    Point2i pt108a = IpGLmPtWithPair(faceInfo, 108, 9, 0.45);
    outPolygon.push_back(pt108a);
    
    outPolygon.push_back(getPtOnGLm(faceInfo, 107));
    outPolygon.push_back(getPtOnGLm(faceInfo, 55));
    outPolygon.push_back(getPtOnGLm(faceInfo, 193));
    outPolygon.push_back(getPtOnGLm(faceInfo, 122));
    outPolygon.push_back(getPtOnGLm(faceInfo, 351));
    outPolygon.push_back(getPtOnGLm(faceInfo, 417));
    outPolygon.push_back(getPtOnGLm(faceInfo, 285));
    outPolygon.push_back(getPtOnGLm(faceInfo, 336));
    
    Point2i pt337a = IpGLmPtWithPair(faceInfo, 337, 9, 0.45);
    outPolygon.push_back(pt337a);
}

Mat ForgeGlabellaMask(const FaceInfo& faceInfo)
{
    POLYGON coarsePolygon, refinedPolygon;
    ForgeGlabellPg(faceInfo, coarsePolygon);
    
    int csNumPoint = 80;
    DenseSmoothPolygon(coarsePolygon, csNumPoint, refinedPolygon);

    Mat outMask(faceInfo.srcImgS, CV_8UC1, cv::Scalar(0));
    // !!!调用这个函数前，outMask必须进行过初始化，或者已有内容在里面！！！
    DrawContOnMask(refinedPolygon, outMask);
    
    return outMask;
}

//-------------------------------------------------------------------------------
// nasolabial grooves 鼻唇沟
void ForgeRightNagvPg( const FaceInfo& faceInfo, POLYGON& outPolygon)
{
    outPolygon.push_back(getPtOnGLm(faceInfo, 355));
    outPolygon.push_back(getPtOnGLm(faceInfo, 429));
    outPolygon.push_back(getPtOnGLm(faceInfo, 279));
    outPolygon.push_back(getPtOnGLm(faceInfo, 358));
    outPolygon.push_back(getPtOnGLm(faceInfo, 410));
    outPolygon.push_back(getPtOnGLm(faceInfo, 273));
    outPolygon.push_back(getPtOnGLm(faceInfo, 431));

    outPolygon.push_back(getPtOnGLm(faceInfo, 430));
    outPolygon.push_back(getPtOnGLm(faceInfo, 432));
    outPolygon.push_back(getPtOnGLm(faceInfo, 436));
    Point2i pt426a = IpGLmPtWithPair(faceInfo, 426, 425, 0.2);
    outPolygon.push_back(pt426a);
    Point2i pt423a = IpGLmPtWithPair(faceInfo, 423, 266, 0.65);
    outPolygon.push_back(pt423a);
    outPolygon.push_back(getPtOnGLm(faceInfo, 371));
}

void ForgeLeftNagvPg( const FaceInfo& faceInfo, POLYGON& outPolygon)
{
    outPolygon.push_back(getPtOnGLm(faceInfo, 126));
    outPolygon.push_back(getPtOnGLm(faceInfo, 209));
    outPolygon.push_back(getPtOnGLm(faceInfo, 49));
    outPolygon.push_back(getPtOnGLm(faceInfo, 129));
    outPolygon.push_back(getPtOnGLm(faceInfo, 186));
    outPolygon.push_back(getPtOnGLm(faceInfo, 43));
    outPolygon.push_back(getPtOnGLm(faceInfo, 211));

    outPolygon.push_back(getPtOnGLm(faceInfo, 210));
    outPolygon.push_back(getPtOnGLm(faceInfo, 212));
    outPolygon.push_back(getPtOnGLm(faceInfo, 216));
    Point2i pt426a = IpGLmPtWithPair(faceInfo, 206, 205, 0.2);
    outPolygon.push_back(pt426a);
    Point2i pt423a = IpGLmPtWithPair(faceInfo, 203, 36, 0.65);
    outPolygon.push_back(pt423a);
    outPolygon.push_back(getPtOnGLm(faceInfo, 142));
}

Mat ForgeRNagvMask(const FaceInfo& faceInfo)
{
    POLYGON coarsePolygon, refinedPolygon;
    //ForgeGlabellPg(faceInfo, coarsePolygon);
    ForgeRightNagvPg(faceInfo, coarsePolygon);

    int csNumPoint = 80;
    DenseSmoothPolygon(coarsePolygon, csNumPoint, refinedPolygon);

    Mat outMask(faceInfo.srcImgS, CV_8UC1, cv::Scalar(0));
    // !!!调用这个函数前，outMask必须进行过初始化，或者已有内容在里面！！！
    DrawContOnMask(refinedPolygon, outMask);
    
    return outMask;
}

Mat ForgeLNagvMask(const FaceInfo& faceInfo)
{
    POLYGON coarsePolygon, refinedPolygon;
    //ForgeGlabellPg(faceInfo, coarsePolygon);
    ForgeLeftNagvPg(faceInfo, coarsePolygon);

    int csNumPoint = 80;
    DenseSmoothPolygon(coarsePolygon, csNumPoint, refinedPolygon);

    Mat outMask(faceInfo.srcImgS, CV_8UC1, cv::Scalar(0));
    // !!!调用这个函数前，outMask必须进行过初始化，或者已有内容在里面！！！
    DrawContOnMask(refinedPolygon, outMask);
    
    return outMask;
}

//-------------------------------------------------------------------------------

void ForgeLCheekPg(const FaceInfo& faceInfo, POLYGON& outPolygon)
{
    Point2i pt350a = IpGLmPtWithPair(faceInfo, 121, 47, 0.35);
    outPolygon.push_back(pt350a);

    outPolygon.push_back(getPtOnGLm(faceInfo, 47));
    
    Point2i pt355a = IpGLmPtWithPair(faceInfo, 126, 100, 0.2);
    outPolygon.push_back(pt355a);
    
    outPolygon.push_back(getPtOnGLm(faceInfo, 142));
    
    Point2i pt266a = IpGLmPtWithPair(faceInfo, 36, 203, 0.5);
    outPolygon.push_back(pt266a);
    
    outPolygon.push_back(getPtOnGLm(faceInfo, 206));
    
    Point2i pt322a = IpGLmPtWithPair(faceInfo, 92, 216, 0.75);
    outPolygon.push_back(pt322a);
    
    Point2i pt212a = IpGLmPtWithPair(faceInfo, 212, 57, 0.5);
    outPolygon.push_back(pt212a);
    
    Point2i pt214a = IpGLmPtWithPair(faceInfo, 214, 210, 0.6);
    outPolygon.push_back(pt214a);
    
    outPolygon.push_back(getPtOnGLm(faceInfo, 136));
    outPolygon.push_back(getPtOnGLm(faceInfo, 172));
    outPolygon.push_back(getPtOnGLm(faceInfo, 215));
    outPolygon.push_back(getPtOnGLm(faceInfo, 177));
    outPolygon.push_back(getPtOnGLm(faceInfo, 137));
    
    Point2i pt323a = IpGLmPtWithPair(faceInfo, 93, 227, 0.3);
    outPolygon.push_back(pt323a);
    
    outPolygon.push_back(getPtOnGLm(faceInfo, 116));
    outPolygon.push_back(getPtOnGLm(faceInfo, 111));
    outPolygon.push_back(getPtOnGLm(faceInfo, 117));
    outPolygon.push_back(getPtOnGLm(faceInfo, 118));
    outPolygon.push_back(getPtOnGLm(faceInfo, 119));
    outPolygon.push_back(getPtOnGLm(faceInfo, 120));
}

void ForgeRCheekPg(const FaceInfo& faceInfo, POLYGON& outPolygon)
{
    Point2i pt350a = IpGLmPtWithPair(faceInfo, 350, 277, 0.35);
    outPolygon.push_back(pt350a);

    outPolygon.push_back(getPtOnGLm(faceInfo, 277));
    
    Point2i pt355a = IpGLmPtWithPair(faceInfo, 355, 329, 0.2);
    outPolygon.push_back(pt355a);
    
    outPolygon.push_back(getPtOnGLm(faceInfo, 371));
    
    Point2i pt266a = IpGLmPtWithPair(faceInfo, 266, 423, 0.5);
    outPolygon.push_back(pt266a);
    
    outPolygon.push_back(getPtOnGLm(faceInfo, 426));
    
    Point2i pt322a = IpGLmPtWithPair(faceInfo, 322, 436, 0.75);
    outPolygon.push_back(pt322a);
    
    //outPolygon.push_back(getPtOnGLm(faceInfo, 287));
    Point2i pt287a = IpGLmPtWithPair(faceInfo, 287, 432, 0.65);
    outPolygon.push_back(pt287a);
    
    Point2i pt434a = IpGLmPtWithPair(faceInfo, 434, 430, 0.6);
    outPolygon.push_back(pt434a);
    
    outPolygon.push_back(getPtOnGLm(faceInfo, 365));
    outPolygon.push_back(getPtOnGLm(faceInfo, 397));
    outPolygon.push_back(getPtOnGLm(faceInfo, 288));
    outPolygon.push_back(getPtOnGLm(faceInfo, 401));
    outPolygon.push_back(getPtOnGLm(faceInfo, 366));
    
    Point2i pt323a = IpGLmPtWithPair(faceInfo, 323, 447, 0.3);
    outPolygon.push_back(pt323a);
    
    outPolygon.push_back(getPtOnGLm(faceInfo, 345));
    outPolygon.push_back(getPtOnGLm(faceInfo, 340));
    outPolygon.push_back(getPtOnGLm(faceInfo, 346));
    outPolygon.push_back(getPtOnGLm(faceInfo, 347));
    outPolygon.push_back(getPtOnGLm(faceInfo, 348));
    outPolygon.push_back(getPtOnGLm(faceInfo, 349));
}

Mat ForgeRCheekMask(const FaceInfo& faceInfo)
{
    POLYGON coarsePolygon, refinedPolygon;
    ForgeRCheekPg(faceInfo, coarsePolygon);

    int csNumPoint = 120;
    DenseSmoothPolygon(coarsePolygon, csNumPoint, refinedPolygon);

    Mat outMask(faceInfo.srcImgS, CV_8UC1, cv::Scalar(0));
    // !!!调用这个函数前，outMask必须进行过初始化，或者已有内容在里面！！！
    DrawContOnMask(refinedPolygon, outMask);
    
    return outMask;
}

Mat ForgeLCheekMask(const FaceInfo& faceInfo)
{
    POLYGON coarsePolygon, refinedPolygon;
    ForgeLCheekPg(faceInfo, coarsePolygon);

    int csNumPoint = 120;
    DenseSmoothPolygon(coarsePolygon, csNumPoint, refinedPolygon);

    Mat outMask(faceInfo.srcImgS, CV_8UC1, cv::Scalar(0));
    // !!!调用这个函数前，outMask必须进行过初始化，或者已有内容在里面！！！
    DrawContOnMask(refinedPolygon, outMask);
    
    return outMask;
}

//-------------------------------------------------------------------------------------------
//  环眼睛区域，不包括眉毛
void ForgeOneCirEyePg(const Point2i eyeRefinePts[71], POLYGON& outPolygon)
{
    // 采用Lip Refine Region的点！
    int fullEyeOuterPtIndices[] = { // 顺时针计数
        23, 22, 21, 20, 19, 18, 17, 32,
        49, 63, 54, 56, 57, 58, 59, 60, 39   // 上外轮廓线，从右到左
    };
    
    int num_pts = sizeof(fullEyeOuterPtIndices) / sizeof(int);
    
    for(int i = 0; i<num_pts; i++)
    {
        int index = fullEyeOuterPtIndices[i];
        outPolygon.push_back(eyeRefinePts[index]);
    }
}

// 环眼睛周边区域，眼睛被抠除
void ForgeOneCirEyeMask(const FaceInfo& faceInfo, EyeID eyeID,
                        const DetectRegion& eyeReg,
                        DetectRegion& lssReg)
{
    POLYGON coarsePolygon, refinedPolygon;
    
    if(eyeID == LEFT_EYE)
        ForgeOneCirEyePg(faceInfo.lEyeRefinePts, coarsePolygon);
    else
        ForgeOneCirEyePg(faceInfo.rEyeRefinePts, coarsePolygon);
    
    int csNumPoint = 50; //200;
    DenseSmoothPolygon(coarsePolygon, csNumPoint, refinedPolygon);
    
    DetectRegion cirFullReg;
    TransPgGS2LSMask(refinedPolygon, cirFullReg);
    
    SubstractDetReg(faceInfo.srcImgS,
                         cirFullReg, eyeReg, lssReg);

}

void ForgeCirEyesMask(const FaceInfo& faceInfo, Mat& outCirEyesMask,
                      const DetectRegion& lEyeReg,
                      const DetectRegion& rEyeReg,
                      DetectRegion& lCirEyeReg,
                      DetectRegion& rCirEyeReg)
{
    cv::Mat outMask(faceInfo.srcImgS, CV_8UC1, cv::Scalar(0));

    ForgeOneCirEyeMask(faceInfo, LEFT_EYE,
                       lEyeReg, lCirEyeReg);
    ForgeOneCirEyeMask(faceInfo, RIGHT_EYE,
                       rEyeReg, rCirEyeReg);
        
    SumDetReg2GSMask(faceInfo.srcImgS, lCirEyeReg,
                     rCirEyeReg, outCirEyesMask);
}
//-------------------------------------------------------------------------------

// 生成皱纹检测的各个检测区（只针对正脸）----新版本
void ForgeWrkTenRegs(const FaceInfo& faceInfo, const Mat& fbBiLab,
                     DetRegPack& detRegPack)
{
    Mat fhMaskGS;
    ForgeFhMaskForWrk(faceInfo, fbBiLab, fhMaskGS);
    TransMaskGS2LS(fhMaskGS, detRegPack.wrkRegGroup.fhReg);
    
    Mat glabMaskGS = ForgeGlabellaMask(faceInfo);
    TransMaskGS2LS(glabMaskGS, detRegPack.wrkRegGroup.glabReg);
    
    glabMaskGS.release();
    fhMaskGS.release();
        
    Mat rNagvMask = ForgeRNagvMask(faceInfo);
    TransMaskGS2LS(rNagvMask, detRegPack.wrkRegGroup.rNagvReg);
    rNagvMask.release();
    
    Mat lNagvMask = ForgeLNagvMask(faceInfo);
    TransMaskGS2LS(lNagvMask, detRegPack.wrkRegGroup.lNagvReg);
    rNagvMask.release();
    
    Mat cirEyesMask(faceInfo.srcImgS, CV_8UC1, cv::Scalar(0));
    ForgeCirEyesMask(faceInfo, cirEyesMask,
                     detRegPack.lEyeReg, detRegPack.rEyeReg,
                     detRegPack.wrkRegGroup.lCirEyeReg,
                     detRegPack.wrkRegGroup.rCirEyeReg);
    detRegPack.cirEyesMask = cirEyesMask;
}

void ForgeWrkTenRegs(const Mat& annoLmImage,
                     const FaceInfo& faceInfo,
                     const Mat& fbBiLab,
                     DetRegPack& detRegPack)
{
    ForgeWrkTenRegs(faceInfo, fbBiLab, detRegPack);
    
#ifdef TEST_RUN2
    Mat fhMaskGS = TransMaskLS2GS(faceInfo.srcImgS, detRegPack.wrkRegGroup.fhReg);
    Mat glabMaskGS = TransMaskLS2GS(faceInfo.srcImgS, detRegPack.wrkRegGroup.glabReg);
    
    string fhMaskImgFile = BuildOutImgFNV2(wrkMaskOutDir, "FhMask.png");
    AnnoMaskOnImage(annoLmImage, fhMaskGS,
                        "Forehead Mask", fhMaskImgFile.c_str());
    
    string glabMaskImgFile = BuildOutImgFNV2(wrkMaskOutDir, "glabMask.png");
    AnnoMaskOnImage(annoLmImage, glabMaskGS,
                        "Glabella Mask", glabMaskImgFile.c_str());
        
    fhMaskGS.release();
    glabMaskGS.release();

    Mat rNagvMaskGS = TransMaskLS2GS(faceInfo.srcImgS, detRegPack.wrkRegGroup.rNagvReg);
    string rNagvMaskImgFile = BuildOutImgFNV2(wrkMaskOutDir, "rNagvMaskGS.png");
    AnnoMaskOnImage(annoLmImage, rNagvMaskGS,
                        "rNagvMaskGS", rNagvMaskImgFile.c_str());
    rNagvMaskGS.release();
    
    Mat lNagvMaskGS = TransMaskLS2GS(faceInfo.srcImgS, detRegPack.wrkRegGroup.lNagvReg);
    string lNagvMaskImgFile = BuildOutImgFNV2(wrkMaskOutDir, "lNagvMaskGS.png");
    AnnoMaskOnImage(annoLmImage, lNagvMaskGS,
                        "lNagvMaskGS", lNagvMaskImgFile.c_str());
    lNagvMaskGS.release();
    
    string eyeCirMaskImgFile = BuildOutImgFNV2(wrkMaskOutDir, "eye_cir_mask.png");
    AnnoMaskOnImage(annoLmImage, detRegPack.cirEyesMask,
                        "eyes_cir_mask", eyeCirMaskImgFile.c_str());

#endif

}
