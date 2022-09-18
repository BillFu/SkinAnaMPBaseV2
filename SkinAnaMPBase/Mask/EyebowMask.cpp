//
//  EyebowMask.cpp

/*******************************************************************************
Author: Fu Xiaoqiang
Date:   2022/9/11

********************************************************************************/

#include "EyebowMask.hpp"
#include "../BSpline/ParametricBSpline.hpp"
#include "DetectRegion.hpp"
#include "Geometry.hpp"

//-------------------------------------------------------------------------------------------

void ForgeOneEyebowPolygon(const FaceInfo& faceInfo, EyeID eyeID, POLYGON& eyebowPolygon)
{
    // 采用Eye Refine Region的点，左眉毛、右眉毛的坐标索引是相同的！
    int eyeBowPtIndices[] = {69, 68, 66, 65, 64, 50, 43, 45};
    //int eyeBowPtIndices[] = {69, 68, 67, 66, 65, 64, 50, 43, 44, 45};

    POLYGON coarseEyebowPolygon;
    int num_pts = sizeof(eyeBowPtIndices) / sizeof(int);
    for(int i = 0; i<num_pts; i++)
    {
        int index = eyeBowPtIndices[i];
        
        int x, y;
        
        if(eyeID == LEFT_EYE)
        {
            x = faceInfo.leftEyeRefinePts[index][0];
            y = faceInfo.leftEyeRefinePts[index][1];
        }
        else  //RightEyeID
        {
            x = faceInfo.rightEyeRefinePts[index][0];
            y = faceInfo.rightEyeRefinePts[index][1];
        }
        
        coarseEyebowPolygon.push_back(Point2i(x, y));
    }
    
    // then convert the corse to the refined
    int csNumPoint = 100;
    ForgeClosedSmoothPolygon(coarseEyebowPolygon, csNumPoint, eyebowPolygon);
}

//-------------------------------------------------------------------------------------------
// Eb is abbreviation for Eyebow
// Pg is abbreviation for Polygon
Point2i getPtOnEb(const FaceInfo& faceInfo, EyeID eyeID, int ptIndex)
{
    int x,y;
     
    if(eyeID == LEFT_EYE)
    {
        x = faceInfo.leftEyeRefinePts[ptIndex][0];
        y = faceInfo.leftEyeRefinePts[ptIndex][1];
    }
    else  // RIGHT_EYE
    {
        x = faceInfo.rightEyeRefinePts[ptIndex][0];
        y = faceInfo.rightEyeRefinePts[ptIndex][1];
    }
    
    return Point2i(x, y);
}

// return new interpolated No.69 and No.68 points, near the right eye!
void CalcRightInter69_and68Pt(const FaceInfo& faceInfo,
                              Point2i& iRP69, Point2i& iRP68)
{
    // iRP69 means the Interpolated right point 69 by interpolating
    // between original lefe 69 and right 69 piont.
    Point2i lP69 = getPtOnEb(faceInfo, LEFT_EYE, 69);
    Point2i rP69 = getPtOnEb(faceInfo, RIGHT_EYE, 69);
    // be careful with the Point order and the value of t when to invoke InnerInterpolate()
    iRP69 = InnerInterpolate(rP69, lP69, 0.20);
    
    Point2i lP68 = getPtOnEb(faceInfo, LEFT_EYE, 68);
    Point2i rP68 = getPtOnEb(faceInfo, RIGHT_EYE, 68);
    // be careful with the Point order and the value of t when to invoke InnerInterpolate()
    iRP68 = InnerInterpolate(rP68, lP68, 0.15);
}

// return new interpolated No.45 and No.43 points, can be applied to two eyes!
void CalcInter45_and43Pt(const FaceInfo& faceInfo, EyeID eyeID,
                              Point2i& iP45, Point2i& iP43)
{
    // iP45 means the Interpolated No.45 point by interpolating
    // between original No.45 and No.53 piont.
    Point2i P45 = getPtOnEb(faceInfo, eyeID, 45);
    Point2i P53 = getPtOnEb(faceInfo, eyeID, 53);
    // be careful with the Point order and the value of t when to invoke InnerInterpolate()
    iP45 = InnerInterpolate(P45, P53, 0.35);
    
    Point2i P43 = getPtOnEb(faceInfo, eyeID, 43);
    Point2i P52 = getPtOnEb(faceInfo, eyeID, 52);
    // be careful with the Point order and the value of t when to invoke InnerInterpolate()
    iP43 = InnerInterpolate(P43, P52, 0.40);
}

// return new interpolated No.50 point, can be applied to two eyes!
void CalcInter50Pt(const FaceInfo& faceInfo, EyeID eyeID,
                              Point2i& iP50)
{
    // iP50 means the Interpolated No.50 point by interpolating
    // between original No.50 and No.65 piont.
    Point2i P50 = getPtOnEb(faceInfo, eyeID, 50);
    Point2i P65 = getPtOnEb(faceInfo, eyeID, 65);
    // be careful with the Point order and the value of t when to invoke InnerInterpolate()
    iP50 = InnerInterpolate(P50, P65, 0.20);
}

// return new interpolated No.64 point, can be applied to two eyes!
// make No.64 moved toward the outside of face a bit.
void CalcInter64Pt(const FaceInfo& faceInfo, EyeID eyeID,
                              Point2i& iP64)
{
    // iP64 means the Interpolated No.64 point by interpolating
    // between original No.64 and No.51 piont.
    Point2i P64 = getPtOnEb(faceInfo, eyeID, 64);
    Point2i P51 = getPtOnEb(faceInfo, eyeID, 51);
    // be careful with the Point order and the value of t when to invoke InnerInterpolate()
    iP64 = InnerInterpolate(P64, P51, -0.30);
}

// can be applied to two eyes!
void RaiseUpPt66_65(const FaceInfo& faceInfo, EyeID eyeID,
                    Point2i& eP66, Point2i& eP65)
{
    // eP66 means the raised up No.66 point by extrapolating
    // between original No.66 and No.52 piont.
    Point2i P66 = getPtOnEb(faceInfo, eyeID, 66);
    Point2i P52 = getPtOnEb(faceInfo, eyeID, 52);
    // be careful with the Point order and the value of t when to invoke InnerInterpolate()
    eP66 = InnerInterpolate(P66, P52, -0.35);
    
    Point2i P65 = getPtOnEb(faceInfo, eyeID, 65);
    Point2i P51 = getPtOnEb(faceInfo, eyeID, 51);
    // be careful with the Point order and the value of t when to invoke InnerInterpolate()
    eP65 = InnerInterpolate(P65, P51, -0.25);
}

// Note: Only forge One!!!
void ForgeRightEbPg(const FaceInfo& faceInfo, POLYGON& rightEbPg)
{
    // 采用Eye Refine Region的点，左眉毛、右眉毛的坐标索引是相同的！
    //int eyeBowPtIndices[] = {i69, i68, 66, 65, 64, 50, 43, 45};

    POLYGON coarseRbPg;
    
    Point2i iRP69, iRP68;
    CalcRightInter69_and68Pt(faceInfo, iRP69, iRP68);
    coarseRbPg.push_back(iRP69);
    coarseRbPg.push_back(iRP68);

    Point2i eP66, eP65;
    RaiseUpPt66_65(faceInfo, RIGHT_EYE,
                    eP66, eP65);
    coarseRbPg.push_back(eP66);
    coarseRbPg.push_back(eP65);
    Point2i RP64 = getPtOnEb(faceInfo, RIGHT_EYE, 64);
    
    Point2i iP64;
    CalcInter64Pt(faceInfo, RIGHT_EYE, iP64);
    coarseRbPg.push_back(iP64);
    
    Point2i iRP50;
    CalcInter50Pt(faceInfo, RIGHT_EYE, iRP50);
    coarseRbPg.push_back(iRP50);
    
    Point2i iP45, iP43;
    CalcInter45_and43Pt(faceInfo, RIGHT_EYE, iP45, iP43);
    // Note: The Order for push_back() is Very Important!!!
    coarseRbPg.push_back(iP43);
    coarseRbPg.push_back(iP45);

    
    // then convert the corse to the refined
    int csNumPoint = 100;
    ForgeClosedSmoothPolygon(coarseRbPg, csNumPoint, rightEbPg);
}

/******************************************************************************************
*******************************************************************************************/
void ForgeTwoEyebowsMask(const FaceInfo& faceInfo, Mat& outMask)
{
    POLYGON leftEyeBowPolygon, rightEbPg;
    ForgeOneEyebowPolygon(faceInfo, LEFT_EYE, leftEyeBowPolygon);
    //ForgeOneEyebowPolygon(faceInfo, RIGHT_EYE, rightEyeBowPolygon);
    ForgeRightEbPg(faceInfo, rightEbPg);

    POLYGON_GROUP polygonGroup;
    polygonGroup.push_back(leftEyeBowPolygon);
    polygonGroup.push_back(rightEbPg);
    
    outMask = ContourGroup2Mask(faceInfo.imgWidth, faceInfo.imgHeight, polygonGroup);
}

//-------------------------------------------------------------------------------------------
