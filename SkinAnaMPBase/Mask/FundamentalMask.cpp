//
//  DetectRegion.cpp

/*******************************************************************************
Author: Fu Xiaoqiang
Date:   2022/9/15

********************************************************************************/

#include "FundamentalMask.hpp"
#include "../BSpline/ParametricBSpline.hpp"
#include "ForeheadMask.hpp"
#include "../Geometry.hpp"
#include "../Utils.hpp"

//-------------------------------------------------------------------------------------------
void expanMask(const Mat& inMask, int expandSize, Mat& outMask)
{
    Mat element = getStructuringElement(MORPH_ELLIPSE,
                           Size(2*expandSize + 1, 2*expandSize+1),
                           Point(expandSize, expandSize));
    
    dilate(inMask, outMask, element);
}

Mat ContourGroup2Mask(int img_width, int img_height, const POLYGON_GROUP& contoursGroup)
{
    cv::Mat mask(img_height, img_width, CV_8UC1, cv::Scalar(0));
    
    for(auto contours: contoursGroup)
    {
        cv::fillPoly(mask, contours, cv::Scalar(255));
    }
    
    return mask;
}

Mat ContourGroup2Mask(Size imgS, const POLYGON_GROUP& contoursGroup)
{
    Mat rst = ContourGroup2Mask(imgS.width, imgS.height, contoursGroup);
    return rst;
}

//-------------------------------------------------------------------------------------------

void ForgeMouthPolygon(const FaceInfo& faceInfo,
                       int& mouthWidth, int& mouthHeight,
                       POLYGON& mouthPolygon)
{
    // the indices for lm in meadiapipe mesh from
    //https://github.com/tensorflow/tfjs-models/blob/838611c02f51159afdd77469ce67f0e26b7bbb23/face-landmarks-detection/src/mediapipe-facemesh/keypoints.ts
    
    // 采用Lip Refine Region的点！
    int lipsOuterPtIndices[] = {
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,  // 下外轮廓线，从左到右
        19, 18, 17, 16, 15, 14, 13, 12, 11 // 上外轮廓线，从右到左
    };
    
    int num_pts = sizeof(lipsOuterPtIndices) / sizeof(int);
    
    for(int i = 0; i<num_pts; i++)
    {
        int index = lipsOuterPtIndices[i];
        mouthPolygon.push_back(faceInfo.lipRefinePts[index]);
    }
    
    // the following five points come from the general lms
    Point2i pt37 = getPtOnGLm(faceInfo, 37);
    Point2i pt267 = getPtOnGLm(faceInfo, 267) ;
    Point2i pt57 = getPtOnGLm(faceInfo, 57);
    Point2i pt287 = getPtOnGLm(faceInfo, 287);
    Point2i pt17 = getPtOnGLm(faceInfo, 17);

    mouthHeight = pt17.y - (pt37.y + pt267.y) / 2;
    mouthWidth  = pt287.x - pt57.x;
}

//-------------------------------------------------------------------------------------------

//expanRatio: expansion width toward outside / half of mouth height
void ForgeMouthMask(const FaceInfo& faceInfo, float expanRatio, Mat& outFinalMask)
{
    POLYGON coarsePolygon, refinedPolygon;

    int mouthW, mouthH;
    ForgeMouthPolygon(faceInfo, mouthW, mouthH, coarsePolygon);
    
    int csNumPoint = 60;
    DenseSmoothPolygon(coarsePolygon, csNumPoint, refinedPolygon);

    // when to construct a Mat, Height first, and then Width!
    Mat basicMask(faceInfo.srcImgS, CV_8UC1, cv::Scalar(0));
    DrawContOnMask(refinedPolygon, basicMask);
    
    int expandSize = expanRatio * mouthH / 2;
    expanMask(basicMask, expandSize, outFinalMask);
}

//-------------------------------------------------------------------------------------------
// Pg: Polygon
void ForgeOneEyeFullPg(const Point2i eyeRefinePts[71], POLYGON& outPolygon)
{
    // 采用Lip Refine Region的点！
    int fullEyeOuterPtIndices[] = { // 顺时针计数
        70, 68, 66, 65, 64, 54,  // 下外轮廓线，从左到右
        56, 57, 58, 59, 60, 61, 62   // 上外轮廓线，从右到左
    };
    
    int num_pts = sizeof(fullEyeOuterPtIndices) / sizeof(int);
    
    for(int i = 0; i<num_pts; i++)
    {
        int index = fullEyeOuterPtIndices[i];
        outPolygon.push_back(eyeRefinePts[index]);
    }
}

// EeyeFullMask包含眼睛、眉毛、眼袋的大范围区域
void ForgeOneEyeFullMask(const FaceInfo& faceInfo, EyeID eyeID, Mat& outMask)
{
    POLYGON coarsePolygon, refinedPolygon;
    
    if(eyeID == LEFT_EYE)
        ForgeOneEyeFullPg(faceInfo.lEyeRefinePts, coarsePolygon);
    else
        ForgeOneEyeFullPg(faceInfo.rEyeRefinePts, coarsePolygon);
    
    int csNumPoint = 50; //200;
    DenseSmoothPolygon(coarsePolygon, csNumPoint, refinedPolygon);

    DrawContOnMask(refinedPolygon, outMask);
}


void ForgeEyesFullMask(const FaceInfo& faceInfo, Mat& outEyesFullMask)
{
    cv::Mat outMask(faceInfo.srcImgS, CV_8UC1, cv::Scalar(0));

    ForgeOneEyeFullMask(faceInfo, LEFT_EYE, outMask);
    ForgeOneEyeFullMask(faceInfo, RIGHT_EYE, outMask);
    
    outEyesFullMask = outMask;
    //expanMask(outMask, 20, outEyesFullMask);
}

//-------------------------------------------------------------------------------------------
//  环眼睛区域，不包括眉毛
void ForgeOneCirEyePg(const Point2i eyeRefinePts[71], POLYGON& outPolygon)
{
    // 采用Lip Refine Region的点！
    int fullEyeOuterPtIndices[] = { // 顺时针计数
        47, 46, 45, 44, 50, 63, 54,  // 下外轮廓线，从左到右
        56, 57, 58, 59, 60, 39   // 上外轮廓线，从右到左
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

//-------------------------------------------------------------------------------------------
void ForgeNosePolygon(const FaceInfo& faceInfo, POLYGON& outPolygon)
{
    // 采用Lip Refine Region的点！
    int nosePtIndices[] = { // 逆时针
    55, 193, 245, 114, 142, 203, 98, 97, // No.55 point near left eye
        2,
    326, 327, 423, 371, 343, 465, 417, 285 };
    
    int num_pts = sizeof(nosePtIndices) / sizeof(int);
    
    for(int i = 0; i<num_pts; i++)
    {
        int index = nosePtIndices[i];
        outPolygon.push_back(faceInfo.lm_2d[index]);
    }
}

void ForgeNoseMask(const FaceInfo& faceInfo, Mat& outMask)
{
    POLYGON coarsePolygon, refinedPolygon;

    ForgeNosePolygon(faceInfo, coarsePolygon);
    
    int csNumPoint = 100;
    DenseSmoothPolygon(coarsePolygon, csNumPoint, refinedPolygon);

    DrawContOnMask(refinedPolygon, outMask);
}
//-------------------------------------------------------------------------------------------

void ForgeNoseBellPg(const FaceInfo& faceInfo, POLYGON& outPg)
{
    outPg.push_back(getPtOnGLm(faceInfo, 197)); // center, top most point.
    
    // then in CCW order.
    outPg.push_back(getPtOnGLm(faceInfo, 196));
    outPg.push_back(getPtOnGLm(faceInfo, 174));
    outPg.push_back(getPtOnGLm(faceInfo, 126));
    Point2i pt142a = IpGLmPtWithPair(faceInfo, 142, 209, 0.25);
    outPg.push_back(pt142a);
    
    Point2i pt203a = IpGLmPtWithPair(faceInfo, 203, 64, 0.25);
    outPg.push_back(pt203a);
    outPg.push_back(getPtOnGLm(faceInfo, 92));

    Point2i pt202a = IpGLmPtWithPair(faceInfo, 202, 43, 0.7);
    outPg.push_back(pt202a);
    outPg.push_back(getPtOnGLm(faceInfo, 211));
    outPg.push_back(getPtOnGLm(faceInfo, 170));

    // add two special points and 152, the lowest point
    Point2i pt170 = getPtOnGLm(faceInfo, 170);
    Point2i pt152 = getPtOnGLm(faceInfo, 152);
    Point2i pt395 = getPtOnGLm(faceInfo, 395);
    
    Point2i sp1 =  getRectCornerPt(pt170, pt152);
    outPg.push_back(sp1);
    outPg.push_back(pt152);
    Point2i sp2 =  getRectCornerPt(pt395, pt152);
    outPg.push_back(sp2);
    
    outPg.push_back(getPtOnGLm(faceInfo, 395));
    outPg.push_back(getPtOnGLm(faceInfo, 431));
    Point2i pt422a = IpGLmPtWithPair(faceInfo, 422, 273, 0.7);
    outPg.push_back(pt422a);
    
    outPg.push_back(getPtOnGLm(faceInfo, 322));
    
    Point2i pt423a = IpGLmPtWithPair(faceInfo, 423, 294, 0.25);
    outPg.push_back(pt423a);
    
    Point2i pt371a = IpGLmPtWithPair(faceInfo, 371, 429, 0.25);
    outPg.push_back(pt371a);
    
    outPg.push_back(getPtOnGLm(faceInfo, 355));
    outPg.push_back(getPtOnGLm(faceInfo, 399));
    outPg.push_back(getPtOnGLm(faceInfo, 419));
}

void ForgeNoseBellMask(const FaceInfo& faceInfo, Mat& outMask)
{
    POLYGON coarsePg, refinedPg;

    ForgeNoseBellPg(faceInfo, coarsePg);
    
    int csNumPoint = 80;
    DenseSmoothPolygon(coarsePg, csNumPoint, refinedPg);

    DrawContOnMask(refinedPg, outMask);
}

//-------------------------------------------------------------------------------------------
