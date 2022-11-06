//
//  WrinkleMask.cpp

/*******************************************************************************
本模块。
  
Author: Fu Xiaoqiang
Date:   2022/11/2
********************************************************************************/
#include <algorithm>
#include "../Geometry.hpp"
#include "../Utils.hpp"
#include "../polyitems_fit.hpp"
#include "../BSpline/ParametricBSpline.hpp"

#include "WrinkleMask.hpp"
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
    /*
    int ptIndices[] = { // 逆时针计数
        107, 55, 193, 122, 351, 417, 285, 336, // 107是最左、最上的点
        pt1, pt2
    };
    pt1: 在9和337之间插值出来的。
    pt2: 在9和108之间插值出来的。
    */
    
    //outPolygon.push_back(getPtOnGLm(faceInfo, 107));
    outPolygon.push_back(getPtOnGLm(faceInfo, 55));
    outPolygon.push_back(getPtOnGLm(faceInfo, 193));
    outPolygon.push_back(getPtOnGLm(faceInfo, 122));
    outPolygon.push_back(getPtOnGLm(faceInfo, 351));
    outPolygon.push_back(getPtOnGLm(faceInfo, 417));
    outPolygon.push_back(getPtOnGLm(faceInfo, 285));
    //outPolygon.push_back(getPtOnGLm(faceInfo, 336));
    
    //Point2i pt1 = IpGLmPtWithPair(faceInfo, 8, 9, 337, 0.6);
    //outPolygon.push_back(pt1);
    outPolygon.push_back(getPtOnGLm(faceInfo, 9));
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

    Point2i pt391a = IpGLmPtWithPair(faceInfo, 391, 393, 0.5);
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
    TransMaskFromGS2LS(glabelMaskSS, wrkRegGroup.glaReg);
    glabelMaskSS.release();
    
    Mat lEyeBagMask = ForgeEyebagMask(faceInfo.srcImgS, faceInfo.lEyeRefinePts);
    TransMaskFromGS2LS(lEyeBagMask, wrkRegGroup.lEyeBagReg);
    lEyeBagMask.release();
    
    Mat rEyeBagMask = ForgeEyebagMask(faceInfo.srcImgS, faceInfo.rEyeRefinePts);
    TransMaskFromGS2LS(rEyeBagMask, wrkRegGroup.rEyeBagReg);
    rEyeBagMask.release();
}


void ForgeWrkTenRegsDebug(const Mat& annoLmImage, const FaceInfo& faceInfo,
                     const Mat& fbBiLab, WrkRegGroup& wrkRegGroup)
{
    ForgeWrkTenRegs(faceInfo, fbBiLab, wrkRegGroup);
    
    Mat fhMaskGS = TransMaskFromLS2GS(faceInfo.srcImgS, wrkRegGroup.fhReg);
    Mat glaMaskGS = TransMaskFromLS2GS(faceInfo.srcImgS, wrkRegGroup.glaReg);
    Mat lEyeBagMaskGS = TransMaskFromLS2GS(faceInfo.srcImgS, wrkRegGroup.lEyeBagReg);
    Mat rEyeBagMaskGS = TransMaskFromLS2GS(faceInfo.srcImgS, wrkRegGroup.rEyeBagReg);
        
    Mat showImg = annoLmImage.clone();
    OverMaskOnCanvas(showImg, fhMaskGS, Scalar(0, 0, 255));
    fhMaskGS.release();
    
    OverMaskOnCanvas(showImg, glaMaskGS, Scalar(255, 0, 0));
    glaMaskGS.release();

    
    // string showImgFile = BuildOutImgFNV2(outDir, "WrkTenRegs.png");
    // imwrite(showImgFile.c_str(), showImg);
    
#ifdef TEST_RUN
    string lEyeBagMaskImgFile = BuildOutImgFNV2(outDir, "lEyeBagMask.png");
    OverlayMaskOnImage(annoLmImage, lEyeBagMaskGS,
                        "lEyeBagMask", lEyeBagMaskImgFile.c_str());
#endif

#ifdef TEST_RUN
    string rEyeBagMaskImgFile = BuildOutImgFNV2(outDir, "rEyeBagMask.png");
    OverlayMaskOnImage(annoLmImage, rEyeBagMaskGS,
                        "rEyeBagMask", rEyeBagMaskImgFile.c_str());
#endif
}
