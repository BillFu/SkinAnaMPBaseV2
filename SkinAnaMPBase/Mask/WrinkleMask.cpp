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
    
    outPolygon.push_back(getPtOnGLm(faceInfo, 107));
    outPolygon.push_back(getPtOnGLm(faceInfo, 55));
    outPolygon.push_back(getPtOnGLm(faceInfo, 193));
    outPolygon.push_back(getPtOnGLm(faceInfo, 122));
    outPolygon.push_back(getPtOnGLm(faceInfo, 351));
    outPolygon.push_back(getPtOnGLm(faceInfo, 417));
    outPolygon.push_back(getPtOnGLm(faceInfo, 285));
    outPolygon.push_back(getPtOnGLm(faceInfo, 336));
    
    Point2i pt1 = IpGLmPtWithPair(faceInfo, 9, 337, 0.6);
    outPolygon.push_back(pt1);
    
    Point2i pt2 = IpGLmPtWithPair(faceInfo, 9, 108, 0.6);
    outPolygon.push_back(pt2);
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
    ForgeForeheadMask(faceInfo, fbBiLab, fhMaskSS);
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
