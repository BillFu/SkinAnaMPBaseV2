//
//  EyebrowMaskV2.cpp

/*******************************************************************************
Author: Fu Xiaoqiang
Date:   2022/10/10

********************************************************************************/

#include "EyebrowMaskV2.hpp"
#include "../BSpline/ParametricBSpline.hpp"
#include "FundamentalMask.hpp"
#include "Geometry.hpp"


//-------------------------------------------------------------------------------------------
// Eb is abbreviation for Eyebrow
// Pg is abbreviation for Polygon
Point2i getPtOnEb(const FaceInfo& faceInfo, EyeID eyeID, int ptIndex)
{
    if(eyeID == LEFT_EYE)
    {
        return faceInfo.lEyeRefinePts[ptIndex];
    }
    else  // RIGHT_EYE
    {
        return faceInfo.rEyeRefinePts[ptIndex];
    }
}

Point2i IpPtInERG(const Point2i eyeRefPts[NUM_PT_EYE_REFINE_GROUP],
                  int pIndex1, int pIndex2, float t)
{
    Point2i P1 = eyeRefPts[pIndex1];
    Point2i P2 = eyeRefPts[pIndex2];
    
    // be careful with the Point order and the value of t when to invoke InnerInterpolate()
    Point2i P3 = Interpolate(P1, P2, t);
    return P3;
}

/*
// Note: Only forge One!!!
void ForgeBrowPg(const Point2i eyeRefPts[NUM_PT_EYE_REFINE_GROUP],
                 const Point2i& segBrowCP,
                 POLYGON& browPg)
{
    // 采用Eye Refine Region的点，左眉毛、右眉毛的坐标索引是相同的！
    //int eyeBowPtIndices[] = {i69, i68, i67, 66, 65, 64, 50, 43, 45};
    Point2i fixedERPts[NUM_PT_EYE_REFINE_GROUP]; // ER: eye refine
    FixERPsBySegBrowCP(eyeRefPts, segBrowCP, fixedERPts);

    POLYGON coarseRbPg;
    
    Point2i iRP69 = IpPtInERG(eyeRefPts, 69, 45, -0.30);
    coarseRbPg.push_back(iRP69);
    
    Point2i iRP68 = IpPtInERG(eyeRefPts, 68, 67, -0.25);
    coarseRbPg.push_back(iRP68);
    
    Point2i eP67 = IpPtInERG(eyeRefPts, 67, 53, -0.55);
    coarseRbPg.push_back(eP67);

    Point2i eP66 = IpPtInERG(eyeRefPts, 66, 52, -0.35);
    coarseRbPg.push_back(eP66);
    Point2i eP65 = IpPtInERG(eyeRefPts, 65, 51, -0.25);
    coarseRbPg.push_back(eP65);
    
    Point2i iP64 = IpPtInERG(eyeRefPts, 64, 51, -0.30);
    coarseRbPg.push_back(iP64);
    
    Point2i iP50 = IpPtInERG(eyeRefPts, 50, 65, 0.10);
    coarseRbPg.push_back(iP50);
    
    Point2i iP43 = IpPtInERG(eyeRefPts, 43, 52, 0.40);
    // Note: The Order for push_back() is Very Important!!!
    coarseRbPg.push_back(iP43);
    
    Point2i iP45 = IpPtInERG(eyeRefPts, 45, 53, 0.25);
    coarseRbPg.push_back(iP45);

    // then convert the corse to the refined
    int csNumPoint = 100;
    CloseSmoothPolygon(coarseRbPg, csNumPoint, browPg);
}
*/

void ForgeBrowPg(const Point2i eyeRefPts[NUM_PT_EYE_REFINE_GROUP],
                 const Point2i& segBrowCP,
                 POLYGON& browPg)
{
    
    // 采用Eye Refine Region的点，左眉毛、右眉毛的坐标索引是相同的！
    // 对右侧眉毛而言，69点是内侧左下角点。点序列按顺时针方向排列。
    //int browERPIDs[] = {69, 68, 67, 66, 65, 64, 50, 43, 44, 45};
    Point2i fixedERPts[NUM_PT_EYE_REFINE_GROUP]; // ER: eye refine
    FixERPsBySegBrowCP(eyeRefPts, segBrowCP, fixedERPts);

    POLYGON coarsePg;
    
    Point2i pt69a = InterpolateX(fixedERPts[69], fixedERPts[70], 0.5);
    coarsePg.push_back(pt69a);
    
    Point2i pt68a = InterpolateY(fixedERPts[68], fixedERPts[69], 0.10);
    coarsePg.push_back(pt68a);
    
    Point2i pt67a = Interpolate(fixedERPts[67], fixedERPts[53], -0.25);
    coarsePg.push_back(pt67a);

    coarsePg.push_back(fixedERPts[66]);
    coarsePg.push_back(fixedERPts[65]);
    
    Point2i iPt64 = IpPtInERG(fixedERPts, 64, 63, 0.5);
    coarsePg.push_back(iPt64);

    coarsePg.push_back(fixedERPts[50]);

    Point2i iPt43 = IpPtInERG(fixedERPts, 50, 44, 0.5);
    coarsePg.push_back(iPt43);
    
    coarsePg.push_back(fixedERPts[44]);
    coarsePg.push_back(fixedERPts[45]);

    // then convert the corse to the refined
    int csNumPoint = 50;
    CloseSmoothPolygon(coarsePg, csNumPoint, browPg);
}
/******************************************************************************************
*******************************************************************************************/
void ForgeBrowsMask(const FaceInfo& faceInfo,
                    const FaceSegResult& segRst,  // Rst: result
                    Mat& outMask)
{
    POLYGON leftBrowPg, rightBrowPg;
    ForgeBrowPg(faceInfo.lEyeRefinePts, segRst.leftBrowCP, leftBrowPg);
    ForgeBrowPg(faceInfo.rEyeRefinePts, segRst.rightBrowCP, rightBrowPg);

    POLYGON_GROUP polygonGroup;
    polygonGroup.push_back(leftBrowPg);
    polygonGroup.push_back(rightBrowPg);
    
    //Mat outOrigMask = ContourGroup2Mask(faceInfo.imgWidth, faceInfo.imgHeight, polygonGroup);
    
    /*
    int dila_size = 10;
    Mat element = getStructuringElement(MORPH_ELLIPSE,
                           Size(2*dila_size + 1, 2*dila_size+1),
                           Point(dila_size, dila_size));
    
    cv::Mat outExpandedMask(faceInfo.imgHeight, faceInfo.imgWidth, CV_8UC1, cv::Scalar(0));
    dilate(outOrigMask, outMask, element);
    */
    
    outMask = ContourGroup2Mask(faceInfo.imgWidth, faceInfo.imgHeight, polygonGroup);
}

//-------------------------------------------------------------------------------------------
// (dx, dy) = kpCP - segCP
Point2i TransEyeRefPt2SegSpace(const Point2i& eyeRefPt, int dx, int dy)
{
    return Point2i(eyeRefPt.x - dx, eyeRefPt.y - dy);
}

//RefPts: refined points
// transform the eye refine points in face mesh space into the segment space
void FixERPsBySegEyeCP(const Point2i eyeRefPts[NUM_PT_EYE_REFINE_GROUP], // input
                       const Point2i& segCP, // eye center point in segment space
                       Point2i fixedEyeRefPts[NUM_PT_EYE_REFINE_GROUP] /* utput */)
{
    Point2i kpCP = (eyeRefPts[19] + eyeRefPts[20]) / 2; // eye center point in key points space
    int dx = kpCP.x - segCP.x;  //
    int dy = kpCP.y - segCP.y;  //
    
    for(int i=0; i<NUM_PT_EYE_REFINE_GROUP; i++)
    {
        fixedEyeRefPts[i] = TransEyeRefPt2SegSpace(eyeRefPts[i], dx, dy);
    }
}

//ERPs: eye refined points
// 用分割出的眉毛中心点来修正ERPs的坐标
// transform the eye refine points in face mesh space into the segment space
void FixERPsBySegBrowCP(const Point2i eyeRefPts[NUM_PT_EYE_REFINE_GROUP], // input
                       const Point2i& segCP, // eye center point in segment space
                       Point2i fixedEyeRefPts[NUM_PT_EYE_REFINE_GROUP]   /* output */)
{
    Point2i kpCP = (eyeRefPts[53] + eyeRefPts[52]) / 2; // brow center point in key points space
    int dx = kpCP.x - segCP.x;  //
    int dy = kpCP.y - segCP.y;  //
    
    for(int i=0; i<NUM_PT_EYE_REFINE_GROUP; i++)
    {
        fixedEyeRefPts[i] = TransEyeRefPt2SegSpace(eyeRefPts[i], dx, dy);
    }
}
/**********************************************************************************************
only cover the two eyes
***********************************************************************************************/

void ForgeOneEyePg(const Point2i eyeRefPts[NUM_PT_EYE_REFINE_GROUP],
                   const Point2i& segEyeCP,
                   POLYGON& outPg)
{
    Point2i fixedERPts[NUM_PT_EYE_REFINE_GROUP]; // ER: eye refine
    FixERPsBySegEyeCP(eyeRefPts, segEyeCP, fixedERPts);

    POLYGON coarsePg;
    
    // 采用Eye Refine Region的点！
    // 以右眼为基准，从内侧上角点开始，顺时针绕一周
    int eyePtIndices[] = { // 顺时针计数
        24, 15, 14, 13, 12, 11, 10, 9, 16,
        //32, // 上轮廓线，从左到右
        //33 //, 34 //, 58, 59, 60, 39    // 下轮廓线，从右到左
    };
    
    int num_pts = sizeof(eyePtIndices) / sizeof(int);
    
    for(int i = 0; i<num_pts; i++)
    {
        int index = eyePtIndices[i];
        coarsePg.push_back(fixedERPts[index]);
    }
    
    Point2i pt25p = Interpolate(fixedERPts[25], fixedERPts[48], 0.52);
    coarsePg.push_back(pt25p);
    
    coarsePg.push_back(fixedERPts[32]);
    coarsePg.push_back(fixedERPts[33]);
    
    Point2i pt57p = Interpolate(fixedERPts[57], fixedERPts[35], 0.60);
    coarsePg.push_back(pt57p);
    
    // 58, 36 插一个
    Point2i pt58p = Interpolate(fixedERPts[58], fixedERPts[36], 0.25);
    coarsePg.push_back(pt58p);

    Point2i pt59p = Interpolate(fixedERPts[59], fixedERPts[37], 0.2);
    coarsePg.push_back(pt59p);
    
    Point2i pt60p = Interpolate(fixedERPts[60], fixedERPts[38], 0.42);
    coarsePg.push_back(pt60p);
    
    Point2i pt39p = Interpolate(fixedERPts[39], fixedERPts[23], 0.15);
    coarsePg.push_back(pt39p);
    
    int csNumPoint = 50; //200;
    CloseSmoothPolygon(coarsePg, csNumPoint, outPg);
}


void ForgeEyesMask(const FaceInfo& faceInfo, const FaceSegResult& segResult, Mat& outMask)
{
    POLYGON leftEyePg, rightEyePg;
    
    ForgeOneEyePg(faceInfo.lEyeRefinePts, segResult.leftEyeCP, leftEyePg);
    ForgeOneEyePg(faceInfo.rEyeRefinePts, segResult.rightEyeCP, rightEyePg);

    POLYGON_GROUP polygonGroup;
    polygonGroup.push_back(leftEyePg);
    polygonGroup.push_back(rightEyePg);
    
    //Mat outOrigMask = ContourGroup2Mask(faceInfo.imgWidth, faceInfo.imgHeight, polygonGroup);
    
    /*
    int dila_size = 10;
    Mat element = getStructuringElement(MORPH_ELLIPSE,
                           Size(2*dila_size + 1, 2*dila_size+1),
                           Point(dila_size, dila_size));
    
    cv::Mat outExpandedMask(faceInfo.imgHeight, faceInfo.imgWidth, CV_8UC1, cv::Scalar(0));
    dilate(outOrigMask, outMask, element);
    */
    
    outMask = ContourGroup2Mask(faceInfo.imgWidth, faceInfo.imgHeight, polygonGroup);
}
