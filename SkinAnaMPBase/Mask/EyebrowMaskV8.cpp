//
//  EyebrowMaskV8.cpp

/*******************************************************************************
Author: Fu Xiaoqiang
Date:   2022/11/3

********************************************************************************/

#include "../BSpline/ParametricBSpline.hpp"
#include "FundamentalMask.hpp"
#include "../Geometry.hpp"
#include "../Chaikin.hpp"
#include "../Common.hpp"

#include "../FaceBgSeg/FaceBgSegV2.hpp"
#include "EyebrowMaskV8.hpp"
#include "SmEyePg.hpp"
#include "polyitems_fit.hpp"

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
    DenseSmoothPolygon(coarsePg, csNumPoint, browPg);
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


/**********************************************************************************************/

void ForgeInitEyePg(const Point2i eyeRefinePts[NUM_PT_EYE_REFINE_GROUP],
                    const Point2i& eyeSegCP, POLYGON& initEyePg)
{
    POLYGON refinedPolygon;
        
    // 采用Eye Refine Region的点！
    // 以右眼为基准，从内侧上角点开始，顺时针绕一周
    int ptIDsOnCurve[] = {40, 23, 22, 21, 20, 19, 18, 17, 32,
        55, 56, 57, 58, 59, 60, 61}; // 顺时针计数, total 9 points
    
    POLYGON initEyePg0;
    int numCoarsePts = sizeof(ptIDsOnCurve) / sizeof(int);
    for(int i = 0; i<numCoarsePts; i++)
    {
        int index = ptIDsOnCurve[i];
        initEyePg0.push_back(eyeRefinePts[index]);
    }
    
    // 根据两个区域（分割出的栅格版，和关键点构造出的矢量版）的质心位置差异，
    // 对矢量版的点位置进行移动，使得二者的重叠面积尽可能地大。
    Moments m = moments(initEyePg0, false);
    Point2f mc_f(m.m10/m.m00, m.m01/m.m00);
    
    Point2i eyePgCP((int)mc_f.x, (int)mc_f.y);
    
    Point2i vecDiff = eyePgCP - eyeSegCP;
    for(Point2i pt: initEyePg0)
    {
        initEyePg.push_back(pt - vecDiff);
    }
}

// forge the polygong of one eye, only use the result of face/bg segment
void ForgeEyePg(Size srcImgS, const SegMask& eyeSegMask,
                const EyeSegFPs& eyeFPs,
                const Mat& srcImage,
                const string& outFileName)
{
    CONTOURS contours;
    findContours(eyeSegMask.mask, contours,
                     cv::noArray(), RETR_EXTERNAL, CHAIN_APPROX_NONE);
    
    //findContours(eyeSegMask.mask, contours,
    //                 cv::noArray(), RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
  
    //cout << "The Number of Points on the contour of eye: "
    //    << contours[0].size() << endl;

    CONTOUR ssEyeCt; // in Source Space
    transCt_LocalSegNOS2SS(contours[0], eyeSegMask.bbox.tl(),
                    srcImgS, ssEyeCt);
    
    Mat canvas = srcImage.clone();
    //CONTOURS curCts;
    //curCts.push_back(_points);
    //drawContours(canvas, curCts, 0, Scalar(150), 2);
    
    CONTOUR upEyeCurve, lowEyeCurve;
    SplitEyeCt(ssEyeCt,eyeFPs.lCorPt, eyeFPs.rCorPt,
                upEyeCurve, lowEyeCurve);
    
    CONTOUR smUpEyeCurve;
    SmCurveByFit(upEyeCurve, smUpEyeCurve);
    //SmoothCtByPIFit(upEyeCurve, smUpEyeCurve);

    CONTOUR smLowEyeCurve;
    SmCurveByFit(lowEyeCurve, smLowEyeCurve);
    //SmoothCtByPIFit(lowEyeCurve, smLowEyeCurve);
    for(int i = 0; i < lowEyeCurve.size(); i++)
    {
        cv::circle(canvas, lowEyeCurve[i], 1, Scalar(150, 200, 50), cv::FILLED);
    }
    /*
    for(int i = 0; i < lowEyeCurve.size(); i++)
    {
        cv::circle(canvas, lowEyeCurve[i], 1, Scalar(150, 200, 50), cv::FILLED);
    }
    */
    
    //Mat canvas = srcImage.clone();
    polylines(canvas, smUpEyeCurve, false,
                  Scalar(0, 0, 255), 2, 8);
    
    polylines(canvas, smLowEyeCurve, false,
                  Scalar(0, 0, 255), 2, 8);
    
    imwrite(outFileName, canvas);
}

/*
// forge the polygong of one eye, only use the result of face/bg segment
void ForgeEyePgV2(Size srcImgS, const SegMask& eyeSegMask,
                  const SegEyeFPsNOS& eyeFPsNOS,
                const Mat& srcImage,
                const string& outFileName)
{
    CONTOURS contours;
    findContours(eyeSegMask.mask, contours,
                     cv::noArray(), RETR_EXTERNAL, CHAIN_APPROX_NONE);  //CHAIN_APPROX_SIMPLE);
    
    Point2i lLocCorPtNOS = eyeFPsNOS.lCorPtNOS - eyeSegMask.bbox.tl();
    Point2i rLocCorPtNOS = eyeFPsNOS.rCorPtNOS - eyeSegMask.bbox.tl();
    
    CONTOUR upEyeCurLocNOS, lowEyeCurLocNOS;
    SplitEyeCt(contours[0],
               lLocCorPtNOS, rLocCorPtNOS,
               upEyeCurLocNOS, lowEyeCurLocNOS);
        
    CONTOUR smUpEyeCurve;
    SmoothCtByPIFitV2(upEyeCurLocNOS, smUpEyeCurve);

    CONTOUR smLowEyeCurve;
    SmoothCtByPIFitV2(lowEyeCurLocNOS, smLowEyeCurve);

}
*/

void ForgeEyesMask(const Mat& srcImage, // add this variable just for debugging
                   const FaceInfo& faceInfo,
                   const FaceSegRst& segRst, //Rst: result,
                   Mat& outMask)
{
    POLYGON leftEyePg, rightEyePg;

    Size srcImgS = faceInfo.srcImgS;
    
    ForgeEyePg(srcImgS, segRst.lEyeMaskNOS, segRst.lEyeFPs,
               srcImage, "leftSmCurve.png");
    ForgeEyePg(srcImgS, segRst.rEyeMaskNOS, segRst.rEyeFPs,
               srcImage, "rightSmCurve.png");

    
    /*
    POLYGON smLEyePg, smREyePg;
    
    POLYGON_GROUP polygonGroup;
    polygonGroup.push_back(smLEyePg);
    polygonGroup.push_back(smREyePg);
    */
    
    POLYGON_GROUP polygonGroup;
    polygonGroup.push_back(rightEyePg);
    polygonGroup.push_back(leftEyePg);
    
    //Mat outOrigMask = ContourGroup2Mask(srcImgS.width, srcImgS.height, polygonGroup);
    outMask = ContourGroup2Mask(srcImgS.width, srcImgS.height, polygonGroup);
    /*
    int dila_size = segRst.faceBBox.width * 0.005;
    Mat element = getStructuringElement(MORPH_ELLIPSE,
                           Size(2*dila_size + 1, 2*dila_size+1),
                           Point(dila_size, dila_size));
    
    //cv::Mat outExpandedMask(srcImgS.height, srcImgS.width, CV_8UC1, cv::Scalar(0));
    //dilate(outOrigMask, outMask, element);
    
    //outMask = ContourGroup2Mask(faceInfo.srcImgS, polygonGroup);
     */
}

/*
void ForgeEyesMaskV2(const Mat& srcImage, // add this variable just for debugging
                   const FaceInfo& faceInfo,
                   const FaceSegRst& segRst, //Rst: result,
                   Mat& outMask)
{
    POLYGON leftEyePg, rightEyePg;

    Size srcImgS = faceInfo.srcImgS;
    
    ForgeEyePgV2(srcImgS, segRst.lEyeMaskNOS, segRst.lEyeFPsNOS,
               srcImage, "leftSmCurve.png");
    ForgeEyePgV2(srcImgS, segRst.rEyeMaskNOS, segRst.rEyeFPsNOS,
               srcImage, "rightSmCurve.png");

    
    POLYGON smLEyePg, smREyePg;
    
    POLYGON_GROUP polygonGroup;
    polygonGroup.push_back(smLEyePg);
    polygonGroup.push_back(smREyePg);

    
    POLYGON_GROUP polygonGroup;
    polygonGroup.push_back(rightEyePg);
    polygonGroup.push_back(leftEyePg);
    
    //Mat outOrigMask = ContourGroup2Mask(srcImgS.width, srcImgS.height, polygonGroup);
    outMask = ContourGroup2Mask(srcImgS.width, srcImgS.height, polygonGroup);
    
}
*/
