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

// 生成皱纹检测的各个检测区（只针对正脸）----新版本
void ForgeWrkTenRegs(const FaceInfo& faceInfo, const Mat& fbBiLab,
                     WrkRegGroup& wrkRegGroup)
{
    Mat fhMaskSS;
    ForgeFhMaskForWrk(faceInfo, fbBiLab, fhMaskSS);
    Rect fhRect = boundingRect(fhMaskSS);
    Mat fhCropMask;
    fhMaskSS(fhRect).copyTo(fhCropMask);
    wrkRegGroup.fhReg = DetectRegion(fhRect, fhCropMask);
    fhMaskSS.release();
    
    Mat glabelMaskSS = ForgeGlabellaMask(faceInfo);
    Rect glabelRect = boundingRect(glabelMaskSS);
    Mat glabelCropMask;
    glabelMaskSS(glabelRect).copyTo(glabelCropMask);
    wrkRegGroup.glaReg = DetectRegion(glabelRect, glabelCropMask);
    glabelMaskSS.release();

}


void ForgeWrkTenRegsDebug(const Mat& annoLmImage, const FaceInfo& faceInfo,
                     const Mat& fbBiLab, WrkRegGroup& wrkRegGroup)
{
    ForgeWrkTenRegs(faceInfo, fbBiLab, wrkRegGroup);
    
    Mat fhMaskGS = TransMaskFromLS2GS(faceInfo.srcImgS, wrkRegGroup.fhReg);
    Mat glaMaskGS = TransMaskFromLS2GS(faceInfo.srcImgS, wrkRegGroup.glaReg);
    
    Mat showImg = annoLmImage.clone();
    OverMaskOnCanvas(showImg, fhMaskGS, Scalar(0, 0, 255));
    OverMaskOnCanvas(showImg, glaMaskGS, Scalar(255, 0, 0));

    string showImgFile = BuildOutImgFNV2(outDir, "WrkTenRegs.png");
    imwrite(showImgFile.c_str(), showImg);
}
