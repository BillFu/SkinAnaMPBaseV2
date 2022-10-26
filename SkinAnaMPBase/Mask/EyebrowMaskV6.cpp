//
//  EyebrowMaskV6.cpp

/*******************************************************************************
Author: Fu Xiaoqiang
Date:   2022/10/26

********************************************************************************/

#include "EyebrowMaskV6.hpp"
#include "../BSpline/ParametricBSpline.hpp"
#include "FundamentalMask.hpp"
#include "Geometry.hpp"
#include "Common.hpp"

#include "../Snake/activecontours.h"

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

// (dx, dy) = kpCP - segCP
Point2i TransEyeRefPt2SegSpace(const Point2i& eyeRefPt, int dx, int dy)
{
    return Point2i(eyeRefPt.x - dx, eyeRefPt.y - dy);
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

/******************************************************************************************
*******************************************************************************************/

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
                    const FaceSegRst& segRst,  // Rst: result
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
    
    outMask = ContourGroup2Mask(faceInfo.srcImgS, polygonGroup);
}

//-------------------------------------------------------------------------------------------


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


/**********************************************************************************************

***********************************************************************************************/

// 用分割的结果构造出一只眼睛的轮廓多边形
void ForgeEyePgBySnakeAlg(Size srcImgS,
                          const SegMask& eyeSegMask,
                          const EyeFPs& eyeFPs,
                          POLYGON& eyePg)
{
    cvalg::ActiveContours acAlg;
    AlgoParams ap;
    acAlg.setParams(&ap);

    int i_width = 0; // ???
    int i_height = 0; // ???
    acAlg.init(i_width, i_height);

    //activeContourAlg.insertPoint(poi.front());
    
    //Mat newFrame = customContourAlg.iterate(frame);
    //Mat nf = customContourAlg.drawSnake(frame);
    //nf.copyTo(frame);

    //eyePg = smEyeCt;
}

void ForgeEyesMask(const Mat& srcImage, // add this variable just for debugging
                   const FaceInfo& faceInfo,
                   const FaceSegRst& segRst, //Rst: result,
                   Mat& outMask)
{
    POLYGON leftEyePg, rightEyePg;

    Size srcImgS = faceInfo.srcImgS;
    
    ForgeEyePgBySnakeAlg(faceInfo.srcImgS, segRst.lEyeMaskNOS, segRst.lEyeFPs,
                         leftEyePg);
    
    ForgeEyePgBySnakeAlg(faceInfo.srcImgS, segRst.rEyeMaskNOS, segRst.rEyeFPs,
                         rightEyePg);
    
    POLYGON_GROUP polygonGroup;
    polygonGroup.push_back(leftEyePg);
    polygonGroup.push_back(rightEyePg);
    
    Mat outOrigMask = ContourGroup2Mask(srcImgS.width, srcImgS.height, polygonGroup);
    
    int dila_size = segRst.faceBBox.width * 0.005;
    Mat element = getStructuringElement(MORPH_ELLIPSE,
                           Size(2*dila_size + 1, 2*dila_size+1),
                           Point(dila_size, dila_size));
    
    cv::Mat outExpandedMask(srcImgS.height, srcImgS.width, CV_8UC1, cv::Scalar(0));
    dilate(outOrigMask, outMask, element);
    
    //outMask = ContourGroup2Mask(faceInfo.srcImgS, polygonGroup);
}
