//
//  EyebowMask.cpp

/*******************************************************************************
Author: Fu Xiaoqiang
Date:   2022/9/11

********************************************************************************/

#include "EyebowMask.hpp"
#include "../BSpline/ParametricBSpline.hpp"
#include "FundamentalMask.hpp"
#include "Geometry.hpp"


//-------------------------------------------------------------------------------------------
// Eb is abbreviation for Eyebow
// Pg is abbreviation for Polygon
Point2i getPtOnEb(const FaceInfo& faceInfo, EyeID eyeID, int ptIndex)
{
    int x,y;
     
    if(eyeID == LEFT_EYE)
    {
        x = faceInfo.lEyeRefinePts[ptIndex].x;
        y = faceInfo.lEyeRefinePts[ptIndex].y;
    }
    else  // RIGHT_EYE
    {
        x = faceInfo.rEyeRefinePts[ptIndex].x;
        y = faceInfo.rEyeRefinePts[ptIndex].y;
    }
    
    return Point2i(x, y);
}

// return new interpolated No.64 point, can be applied to two eyes!
// make No.64 moved toward the outside of face a bit.
// Ip is the abbrevation for Interpolate
Point2i IpPtwithPair(const FaceInfo& faceInfo, EyeID eyeID, int pIndex1, int pIndex2, float t)
{
    Point2i P1 = getPtOnEb(faceInfo, eyeID, pIndex1);
    Point2i P2 = getPtOnEb(faceInfo, eyeID, pIndex2);
    
    // be careful with the Point order and the value of t when to invoke InnerInterpolate()
    Point2i P3 = Interpolate(P1, P2, t);
    return P3;
}

// Note: Only forge One!!!
void ForgeOneEbPg(const FaceInfo& faceInfo, EyeID eyeID, POLYGON& rightEbPg)
{
    // 采用Eye Refine Region的点，左眉毛、右眉毛的坐标索引是相同的！
    //int eyeBowPtIndices[] = {i69, i68, i67, 66, 65, 64, 50, 43, 45};

    POLYGON coarseRbPg;
    
    Point2i iRP69 = IpPtwithPair(faceInfo, eyeID, 69, 45, -0.30);
    coarseRbPg.push_back(iRP69);
    
    Point2i iRP68 = IpPtwithPair(faceInfo, eyeID, 68, 67, -0.25);
    coarseRbPg.push_back(iRP68);
    
    Point2i eP67 = IpPtwithPair(faceInfo, eyeID, 67, 53, -0.55);
    coarseRbPg.push_back(eP67);

    Point2i eP66 = IpPtwithPair(faceInfo, eyeID, 66, 52, -0.35);
    coarseRbPg.push_back(eP66);
    Point2i eP65 = IpPtwithPair(faceInfo, eyeID, 65, 51, -0.25);
    coarseRbPg.push_back(eP65);
    
    Point2i iP64 = IpPtwithPair(faceInfo, eyeID, 64, 51, -0.30);
    coarseRbPg.push_back(iP64);
    
    Point2i iP50 = IpPtwithPair(faceInfo, eyeID, 50, 65, 0.10);
    coarseRbPg.push_back(iP50);
    
    Point2i iP43 = IpPtwithPair(faceInfo, eyeID, 43, 52, 0.40);
    // Note: The Order for push_back() is Very Important!!!
    coarseRbPg.push_back(iP43);
    
    Point2i iP45 = IpPtwithPair(faceInfo, eyeID, 45, 53, 0.25);
    coarseRbPg.push_back(iP45);

    // then convert the corse to the refined
    int csNumPoint = 100;
    CloseSmoothPolygon(coarseRbPg, csNumPoint, rightEbPg);
}

/******************************************************************************************
*******************************************************************************************/
void ForgeTwoEyebowsMask(const FaceInfo& faceInfo, Mat& outMask)
{
    POLYGON leftEbPg, rightEbPg;
    ForgeOneEbPg(faceInfo, LEFT_EYE, leftEbPg);
    ForgeOneEbPg(faceInfo, RIGHT_EYE, rightEbPg);

    POLYGON_GROUP polygonGroup;
    polygonGroup.push_back(leftEbPg);
    polygonGroup.push_back(rightEbPg);
    
    Mat outOrigMask = ContourGroup2Mask(faceInfo.imgWidth, faceInfo.imgHeight, polygonGroup);
    
    int dila_size = 10;
    Mat element = getStructuringElement(MORPH_ELLIPSE,
                           Size(2*dila_size + 1, 2*dila_size+1),
                           Point(dila_size, dila_size));
    
    cv::Mat outExpandedMask(faceInfo.imgHeight, faceInfo.imgWidth, CV_8UC1, cv::Scalar(0));
    dilate(outOrigMask, outMask, element);
}

//-------------------------------------------------------------------------------------------
