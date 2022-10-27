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

#include "../FaceBgSeg/FaceBgSegV2.hpp"
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
                    float expandScale, int& numPtsUC, POLYGON& initEyePg)
{
    POLYGON refinedPolygon;
        
    // 采用Eye Refine Region的点！
    // 以右眼为基准，从内侧上角点开始，顺时针绕一周
    // 分为上弧线和下弧线
    int ptIDsOnUpCurve[] = {40, 23, 22, 21, 20, 19, 18, 17, 32}; // 顺时针计数, total 9 points
    int ptIDsOnLowCurve[] = {55, 56, 57, 58, 59, 60, 61};  // total 7 points
    
    CONTOUR coarseUpCurve, coarseLowCurve;
    int numCoarsePtsUC = sizeof(ptIDsOnUpCurve) / sizeof(int);
    for(int i = 0; i<numCoarsePtsUC; i++)
    {
        int index = ptIDsOnUpCurve[i];
        coarseUpCurve.push_back(eyeRefinePts[index]);
    }
    
    int numCoarsePtsLC = sizeof(ptIDsOnLowCurve) / sizeof(int);
    for(int i = 0; i<numCoarsePtsLC; i++)
    {
        int index = ptIDsOnLowCurve[i];
        coarseLowCurve.push_back(eyeRefinePts[index]);
    }
    
    CONTOUR refineUpCurve, refineLowCurve;
    DenseSmoothPolygon(coarseUpCurve, 120, refineUpCurve, false);
    DenseSmoothPolygon(coarseLowCurve, 100, refineLowCurve, false);
    
    numPtsUC = (int)(refineUpCurve.size());

    // combine the upper and lower curve into a closed contour
    POLYGON combinedCurve(refineUpCurve);
    for(Point2i pt: refineLowCurve)
    {
        combinedCurve.push_back(pt);
    }
    
    Rect coaPgBBox = boundingRect(combinedCurve); // maybe need padding and enlarge
    Point2i centerPt = (coaPgBBox.tl() + coaPgBBox.br())/ 2;
    
    for(Point2i pt: combinedCurve)
    {
        Point2i expPt = (pt - centerPt) * expandScale + centerPt;
        initEyePg.push_back(expPt);
    }
}

// 移动初始眼睛轮廓线上的点，让它们跳出Mask的包围圈
// 逃出包围圈的方向：从中心点出发，到当前轮廓点连一条线，沿这条线远离中心点，最终可逃离包围圈，或到达工作区边界。
// 采用新思路，就可以不用区分上弧线、下弧线。
// eyeCPWC: center point of eye in working coordinate system
void MoveInitEyePtsOutMask(const POLYGON& eyePgWC,
                           int workRegW, int workRegH,
                           const Mat& workMask,
                           const Point2i& eyeCPWC,  
                           POLYGON& adjustEyePgWC)
{
    Rect workRect(0, 0, workRegW, workRegH);
    
    for(int i = 0; i<eyePgWC.size(); i++)
    {
        Point pt = eyePgWC[i];
        
        if(workMask.at<uchar>(pt) == 0) // already be out of mask
            adjustEyePgWC.push_back(pt);
        else
        {
            while((workRect.contains(pt)) && (workMask.at<uchar>(pt) == 255))
            {
                Point dvec = pt - eyeCPWC;
                float len = sqrt(dvec.x * dvec.x + dvec.y * dvec.y);
                float unit_dx = dvec.x / len;
                float unit_dy = dvec.y / len;
                Point mv_offset(unit_dx * 5, unit_dy * 5);
                pt += mv_offset;
            }
            
            // check the moved point and make sure that it still lies in working region.
            if(workRect.contains(pt) == false)
            {
                pt = MakePtInRect(workRect, pt);
            }
        
            adjustEyePgWC.push_back(pt);
        }
    }
}

// 用分割的结果构造出一只眼睛的轮廓多边形
// eyeCP: given by face/bg segment and in source space
void ForgeEyePgBySnakeAlg(Size srcImgS,
                          const Point2i eyeRefinePts[NUM_PT_EYE_REFINE_GROUP],
                          const SegMask& eyeSegMask, // in NOS
                          const EyeFPs& eyeFPs,
                          const Point2i& eyeCP,
                          POLYGON& eyePg)
{
    POLYGON initEyePg;
    int numPtsUC;
    ForgeInitEyePg(eyeRefinePts, 1.6, numPtsUC, initEyePg);
    
    Rect initEyePgBBox = boundingRect(initEyePg); // maybe need padding and enlarge

    Rect eyeMaskBBox = RectNOS2RectSS(srcImgS, eyeSegMask.bbox);
    
    Mat segEyeMaskSS;
    resize(eyeSegMask.mask, segEyeMaskSS, eyeMaskBBox.size(), INTER_NEAREST);
    
    if(RectContainsRect(initEyePgBBox, eyeMaskBBox) == false)
    {
        // enlarge the initEyePgBBox by use of union operation
        initEyePgBBox = initEyePgBBox | eyeMaskBBox;
    }
    
    Mat workImg(initEyePgBBox.size(), CV_8UC1, Scalar(0));
    Rect relativeRect = CalcRelativeRect(initEyePgBBox, eyeMaskBBox);
    segEyeMaskSS.copyTo(workImg(relativeRect));
    
    //--------------------------------------------------------------
    // output the current data state to check whether they are appropriate or not
    
    POLYGON eyePgWC; // working coordinate system
    //ForgeInitEyePg(eyeRefinePts, initEyePg);
    for(Point2i pt: initEyePg)
    {
        Point2i ptWC = pt - initEyePgBBox.tl();
        eyePgWC.push_back(ptWC);
    }
    
    Point2i eyeCPWC = eyeCP - initEyePgBBox.tl();

    // 通过移动向上、向上的移动，使eyePgWC上的点跳出Mask的包围圈。
    POLYGON adjustEyePgWC;
    MoveInitEyePtsOutMask(eyePgWC, workImg.cols, workImg.rows,
                        workImg, eyeCPWC, adjustEyePgWC);
    
    polylines(workImg, adjustEyePgWC, true, Scalar(150), 2, 8);
    circle(workImg, eyeCPWC, 5, Scalar(150), -1);  // 画半径为5的圆(画点）

    imwrite("initDataAC.png", workImg);

    //--------------------------------------------------------------
    cvalg::ActiveContours acAlg;
    AlgoParams ap;
    acAlg.setParams(&ap);

    Size workImgS = workImg.size();
    acAlg.init(workImgS.width, workImgS.height);

    for(Point2i pt: adjustEyePgWC)
    {
        acAlg.insertPoint(pt);
    }
    
    Mat newFrame = acAlg.iterate(workImg);
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
    
    ForgeEyePgBySnakeAlg(faceInfo.srcImgS, faceInfo.lEyeRefinePts,
                         segRst.lEyeMaskNOS, segRst.lEyeFPs,
                         segRst.leftEyeCP, leftEyePg);
    
    /*
    ForgeEyePgBySnakeAlg(faceInfo.srcImgS, faceInfo.rEyeRefinePts,
                         segRst.rEyeMaskNOS, segRst.rEyeFPs,
                         segRst.rightEyeCP, rightEyePg);
    */
    
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
