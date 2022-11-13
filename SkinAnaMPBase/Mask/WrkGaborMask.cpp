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
#include "../polyitems_fit.hpp"
#include "../BSpline/ParametricBSpline.hpp"

#include "WrkGaborMask.hpp"
#include "FundamentalMask.hpp"
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
    
    outPolygon.push_back(getPtOnGLm(faceInfo, 296));
    outPolygon.push_back(getPtOnGLm(faceInfo, 336));
    outPolygon.push_back(getPtOnGLm(faceInfo, 9));
    outPolygon.push_back(getPtOnGLm(faceInfo, 107));
    outPolygon.push_back(getPtOnGLm(faceInfo, 66));
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

void ForgeEyebagPg(const Point2i eyeRefPts[71],
                   POLYGON& outPolygon)
{
    /*
    int ptIndices[] = { 34, 35, 36/20, 37/21, 22, 8, 40/47, 61/39, 60, 59, 58, 68/57 // 逆时针计数
    };
    */
    
    outPolygon.push_back(eyeRefPts[34]);
    outPolygon.push_back(eyeRefPts[35]);
    
    Point2i pt36a = Interpolate(eyeRefPts[36], eyeRefPts[20], 0.25);
    outPolygon.push_back(pt36a);
    
    Point2i pt37a = Interpolate(eyeRefPts[37], eyeRefPts[21], 0.6);
    outPolygon.push_back(pt37a);
        
    outPolygon.push_back(eyeRefPts[22]);
    outPolygon.push_back(eyeRefPts[8]);
    
    Point2i pt40a = Interpolate(eyeRefPts[40], eyeRefPts[47], 0.6);
    outPolygon.push_back(pt40a);
    
    //outPolygon.push_back(eyeRefPts[61]);
    Point2i pt61a = Interpolate(eyeRefPts[61], eyeRefPts[39], 0.35);
    outPolygon.push_back(pt61a);
    
    outPolygon.push_back(eyeRefPts[60]);

    Point2i pt59a = Interpolate(eyeRefPts[59], eyeRefPts[37], -0.2);
    outPolygon.push_back(pt59a);
    
    //outPolygon.push_back(eyeRefPts[58]);
    Point2i pt58a = Interpolate(eyeRefPts[58], eyeRefPts[36], -0.2);
    outPolygon.push_back(pt58a);
    
    Point2i pt58b = Interpolate(eyeRefPts[58], eyeRefPts[57], 0.75);
    outPolygon.push_back(pt58b);
}

Mat ForgeEyebagMask(Size srcImgS, const Point2i eyeRefPts[71])
{
    POLYGON coarsePolygon, refinedPolygon;
    ForgeEyebagPg(eyeRefPts, coarsePolygon);
    
    int csNumPoint = 60;
    DenseSmoothPolygon(coarsePolygon, csNumPoint, refinedPolygon);

    Mat outMask(srcImgS, CV_8UC1, cv::Scalar(0));
    // !!!调用这个函数前，outMask必须进行过初始化，或者已有内容在里面！！！
    DrawContOnMask(refinedPolygon, outMask);
    
    return outMask;
}
//-------------------------------------------------------------------------------
// nasolabial grooves 鼻唇沟
void ForgeRightNagvPg( const FaceInfo& faceInfo, POLYGON& outPolygon)
{
    outPolygon.push_back(getPtOnGLm(faceInfo, 355));
    
    Point2i pt429a = IpGLmPtWithPair(faceInfo, 429, 420, 0.65);
    outPolygon.push_back(pt429a);
    Point2i pt360a = IpGLmPtWithPair(faceInfo, 360, 363, 0.5);
    outPolygon.push_back(pt360a);
    Point2i pt344a = IpGLmPtWithPair(faceInfo, 344, 440, 0.7);
    outPolygon.push_back(pt344a);
    outPolygon.push_back(getPtOnGLm(faceInfo, 309));
    Point2i pt391a = IpGLmPtWithPair(faceInfo, 391, 393, 0.6);
    outPolygon.push_back(pt391a);
    Point2i pt269a = IpGLmPtWithPair(faceInfo, 269, 270, 0.5);
    outPolygon.push_back(pt269a);
    outPolygon.push_back(getPtOnGLm(faceInfo, 409));
    Point2i pt287a = IpGLmPtWithPair(faceInfo, 287, 273, 0.6);
    outPolygon.push_back(pt287a);
    outPolygon.push_back(getPtOnGLm(faceInfo, 430));
    outPolygon.push_back(getPtOnGLm(faceInfo, 432));
    outPolygon.push_back(getPtOnGLm(faceInfo, 436));
    Point2i pt426a = IpGLmPtWithPair(faceInfo, 426, 425, 0.2);
    outPolygon.push_back(pt426a);
    Point2i pt423a = IpGLmPtWithPair(faceInfo, 423, 266, 0.65);
    outPolygon.push_back(pt423a);
    outPolygon.push_back(getPtOnGLm(faceInfo, 371));
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

//-------------------------------------------------------------------------------

void ForgeRCrowFeetPg(const FaceInfo& faceInfo, POLYGON& outPolygon)
{
    //从右上角开始，逆时针次序。
    Point2i pt368a = IpGLmPtWithPair(faceInfo, 368, 356, 0.15);
    outPolygon.push_back(pt368a);
    
    outPolygon.push_back(getPtOnGLm(faceInfo, 383));
    outPolygon.push_back(getPtOnGLm(faceInfo, 353));
    outPolygon.push_back(getPtOnGLm(faceInfo, 446));
    outPolygon.push_back(getPtOnGLm(faceInfo, 261));
    outPolygon.push_back(getPtOnGLm(faceInfo, 448));
    
    Point2i pt340a = IpGLmPtWithPair(faceInfo, 340, 346, 0.25);
    outPolygon.push_back(pt340a);
    
    outPolygon.push_back(getPtOnGLm(faceInfo, 345));
    
    Point2i pt345a = IpGLmPtWithPair(faceInfo, 345, 454, 0.5);
    Point2i pt447 = getPtOnGLm(faceInfo, 447);
    Point2i pt447a = Interpolate(pt447, pt345a, 0.6);
    outPolygon.push_back(pt447a);
    
    outPolygon.push_back(getPtOnGLm(faceInfo, 454));
    outPolygon.push_back(getPtOnGLm(faceInfo, 356));
}

Mat ForgeRCrowFeetMask(const FaceInfo& faceInfo)
{
    POLYGON coarsePolygon, refinedPolygon;
    ForgeRCrowFeetPg(faceInfo, coarsePolygon);

    int csNumPoint = 60;
    DenseSmoothPolygon(coarsePolygon, csNumPoint, refinedPolygon);

    Mat outMask(faceInfo.srcImgS, CV_8UC1, cv::Scalar(0));
    // !!!调用这个函数前，outMask必须进行过初始化，或者已有内容在里面！！！
    DrawContOnMask(refinedPolygon, outMask);
    
    return outMask;
}

void ForgeLCrowFeetPg(const FaceInfo& faceInfo, POLYGON& outPolygon)
{
    //从右上角开始，逆时针次序。
    Point2i pt368a = IpGLmPtWithPair(faceInfo, 139, 127, 0.15);
    outPolygon.push_back(pt368a);
    
    outPolygon.push_back(getPtOnGLm(faceInfo, 156));
    outPolygon.push_back(getPtOnGLm(faceInfo, 124));
    outPolygon.push_back(getPtOnGLm(faceInfo, 226));
    outPolygon.push_back(getPtOnGLm(faceInfo, 31));
    outPolygon.push_back(getPtOnGLm(faceInfo, 228));
    
    Point2i pt340a = IpGLmPtWithPair(faceInfo, 111, 117, 0.25);
    outPolygon.push_back(pt340a);
    
    outPolygon.push_back(getPtOnGLm(faceInfo, 116));
    
    Point2i pt345a = IpGLmPtWithPair(faceInfo, 116, 234, 0.5);
    Point2i pt447 = getPtOnGLm(faceInfo, 227);
    Point2i pt447a = Interpolate(pt447, pt345a, 0.6);
    outPolygon.push_back(pt447a);
    
    outPolygon.push_back(getPtOnGLm(faceInfo, 234));
    outPolygon.push_back(getPtOnGLm(faceInfo, 127));
}

Mat ForgeLCrowFeetMask(const FaceInfo& faceInfo)
{
    POLYGON coarsePolygon, refinedPolygon;
    ForgeLCrowFeetPg(faceInfo, coarsePolygon);

    int csNumPoint = 60;
    DenseSmoothPolygon(coarsePolygon, csNumPoint, refinedPolygon);

    Mat outMask(faceInfo.srcImgS, CV_8UC1, cv::Scalar(0));
    // !!!调用这个函数前，outMask必须进行过初始化，或者已有内容在里面！！！
    DrawContOnMask(refinedPolygon, outMask);
    
    return outMask;
}

//-------------------------------------------------------------------------------

// 生成皱纹检测的各个检测区（只针对正脸）----新版本
void ForgeWrkTenRegs(const FaceInfo& faceInfo, const Mat& fbBiLab,
                     WrkRegGroup& wrkRegGroup)
{
    Mat fhMaskSS;
    ForgeFhMaskForWrk(faceInfo, fbBiLab, fhMaskSS);
    TransMaskFromGS2LS(fhMaskSS, wrkRegGroup.fhReg);
    fhMaskSS.release();
    
    Mat glabelMaskSS = ForgeGlabellaMask(faceInfo);
    TransMaskFromGS2LS(glabelMaskSS, wrkRegGroup.glabReg);
    glabelMaskSS.release();
    
    Mat lEyeBagMask = ForgeEyebagMask(faceInfo.srcImgS, faceInfo.lEyeRefinePts);
    TransMaskFromGS2LS(lEyeBagMask, wrkRegGroup.lEyeBagReg);
    lEyeBagMask.release();
    
    Mat rEyeBagMask = ForgeEyebagMask(faceInfo.srcImgS, faceInfo.rEyeRefinePts);
    TransMaskFromGS2LS(rEyeBagMask, wrkRegGroup.rEyeBagReg);
    rEyeBagMask.release();
    
    Mat rNagvMask = ForgeRNagvMask(faceInfo);
    TransMaskFromGS2LS(rNagvMask, wrkRegGroup.rNagvReg);
    rNagvMask.release();

    Mat rCheekMask = ForgeRCheekMask(faceInfo);
    TransMaskFromGS2LS(rCheekMask, wrkRegGroup.rCheekReg);
    rCheekMask.release();
    
    Mat lCheekMask = ForgeLCheekMask(faceInfo);
    TransMaskFromGS2LS(lCheekMask, wrkRegGroup.lCheekReg);
    lCheekMask.release();
    
    Mat rCFMask = ForgeRCrowFeetMask(faceInfo);
    TransMaskFromGS2LS(rCFMask, wrkRegGroup.rCrowFeetReg);
    rCFMask.release();
    
    Mat lCFMask = ForgeLCrowFeetMask(faceInfo);
    TransMaskFromGS2LS(lCFMask, wrkRegGroup.lCrowFeetReg);
    lCFMask.release();
}

void ForgeWrkTenRegs(const Mat& annoLmImage,
                     const FaceInfo& faceInfo,
                     const Mat& fbBiLab, WrkRegGroup& wrkRegGroup)
{
    ForgeWrkTenRegs(faceInfo, fbBiLab, wrkRegGroup);
    
    Mat fhMaskGS = TransMaskFromLS2GS(faceInfo.srcImgS, wrkRegGroup.fhReg);
    Mat glabMaskGS = TransMaskFromLS2GS(faceInfo.srcImgS, wrkRegGroup.glabReg);
    Mat lEyeBagMaskGS = TransMaskFromLS2GS(faceInfo.srcImgS, wrkRegGroup.lEyeBagReg);
    Mat rEyeBagMaskGS = TransMaskFromLS2GS(faceInfo.srcImgS, wrkRegGroup.rEyeBagReg);
            
#ifdef TEST_RUN2
    string fhMaskImgFile = BuildOutImgFNV2(wrkMaskOutDir, "FhMask.png");
    OverlayMaskOnImage(annoLmImage, fhMaskGS,
                        "Forehead Mask", fhMaskImgFile.c_str());
    
    string glabMaskImgFile = BuildOutImgFNV2(wrkMaskOutDir, "glabMask.png");
    OverlayMaskOnImage(annoLmImage, glabMaskGS,
                        "Glabella Mask", glabMaskImgFile.c_str());
#endif
    fhMaskGS.release();
    glabMaskGS.release();
    
#ifdef TEST_RUN2
    Mat eyeBagMaskGS = lEyeBagMaskGS | rEyeBagMaskGS;
    string eyeBagMaskImgFile = BuildOutImgFNV2(wrkMaskOutDir, "EyeBagMask.png");
    OverlayMaskOnImage(annoLmImage, eyeBagMaskGS,
                        "EyeBagMask", eyeBagMaskImgFile.c_str());
#endif
    
    Mat rNagvMaskGS = TransMaskFromLS2GS(faceInfo.srcImgS, wrkRegGroup.rNagvReg);
#ifdef TEST_RUN2
    string rNagvMaskImgFile = BuildOutImgFNV2(wrkMaskOutDir, "rNagvMaskGS.png");
    OverlayMaskOnImage(annoLmImage, rNagvMaskGS,
                        "rNagvMaskGS", rNagvMaskImgFile.c_str());
#endif

    Mat rCheekMaskGS = TransMaskFromLS2GS(faceInfo.srcImgS, wrkRegGroup.rCheekReg);
    Mat lCheekMaskGS = TransMaskFromLS2GS(faceInfo.srcImgS, wrkRegGroup.lCheekReg);
    Mat cheekMaskGS = rCheekMaskGS | lCheekMaskGS;
    
#ifdef TEST_RUN2
    string cheekMaskImgFile = BuildOutImgFNV2(wrkMaskOutDir, "CheekRegGS.png");
    OverlayMaskOnImage(annoLmImage, cheekMaskGS,
                        "CheekMaskGS", cheekMaskImgFile.c_str());
#endif
    
    Mat lCFMaskGS = TransMaskFromLS2GS(faceInfo.srcImgS, wrkRegGroup.lCrowFeetReg);
    Mat rCFMaskGS = TransMaskFromLS2GS(faceInfo.srcImgS, wrkRegGroup.rCrowFeetReg);
#ifdef TEST_RUN2
    string lCFMaskFile = BuildOutImgFNV2(wrkMaskOutDir, "lCFRegGS.png");
    OverlayMaskOnImage(annoLmImage, lCFMaskGS,
                        "lCFMaskGS", lCFMaskFile.c_str());
    string rCFMaskFile = BuildOutImgFNV2(wrkMaskOutDir, "rCFRegGS.png");
    OverlayMaskOnImage(annoLmImage, rCFMaskGS,
                        "rCFMaskGS", rCFMaskFile.c_str());
#endif

}
