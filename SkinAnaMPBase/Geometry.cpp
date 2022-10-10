//
//  Geometry.cpp

/*******************************************************************************
本模块提供一些用于2D矢量几何计算的函数。

Author: Fu Xiaoqiang
Date:   2022/9/18

********************************************************************************/

#include "Geometry.hpp"
//#include "FaceBgSeg/FaceBgSeg.hpp"

Point2i getPtOnGLm(const FaceInfo& faceInfo, int pIndex)
{
    int x = faceInfo.lm_2d[pIndex].x;
    int y = faceInfo.lm_2d[pIndex].y;
    
    return Point2i(x, y);
}


/**********************************************************************************************
(x3, y3) is the inner interpolated result.
t: in range[0.0 1.0].
when t --> 0.0, then (x3, y3) --> (x1, y1);
when t --> 1.0, then (x3, y3) --> (x2, y2);

***********************************************************************************************/
void Interpolate(int x1, int y1, int x2, int y2, float t, int& x3, int& y3)
{
    x3 = (int)(x1 + (x2 - x1) * t);
    y3 = (int)(y1 + (y2 - y1) * t);
}

Point2i Interpolate(int x1, int y1, int x2, int y2, float t)
{
    int x3, y3;
    Interpolate(x1, y1, x2, y2, t, x3, y3);

    return Point2i(x3, y3);
}

Point2i Interpolate(const Point2i& p1, const Point2i& p2, float t)
{
    Point2i p3 = Interpolate(p1.x, p1.y, p2.x, p2.y, t);
    return p3;
}
//-------------------------------------------------------------------------------------------

/**********************************************************************************************
Ip: interpolate
GLm: general landmark
Pt: point
t: lies in [0.0 1.0], t-->0, out point-->Pt1;
***********************************************************************************************/
// Ip is the abbrevation for Interpolate
Point2i IpGLmPtWithPair(const FaceInfo& faceInfo, int pIndex1, int pIndex2, float t)
{
    Point2i P1 = getPtOnGLm(faceInfo, pIndex1);
    Point2i P2 = getPtOnGLm(faceInfo, pIndex2);
    
    // be careful with the Point order and the value of t when to invoke InnerInterpolate()
    Point2i P3 = Interpolate(P1, P2, t);
    return P3;
}

//-------------------------------------------------------------------------------------------

// return a new point, which with x from p1, y from p2
Point2i getRectCornerPt(const Point2i& p1, const Point2i& p2)
{
    return Point2i(p1.x, p2.y);
}

//-------------------------------------------------------------------------------------------
/**********************************************************************************************
 segLabels: 512*512
***********************************************************************************************/
/*
int CalcLowerJawWidth(const FaceInfo& faceInfo, const Mat& segLabels)
{
    Point2i pt200 = getPtOnGLm(faceInfo, 200);
    int pt200y = pt200.y;  // in source image space
    
    // convert pt200y from the source space into seg net space (512*512)
    int pt200yp = pt200y * SEG_NET_OUTPUT_SIZE / faceInfo.imgHeight;
    
    Mat jawRow = segLabels.row(pt200yp);
    
    int jawWidth = 0;
    for(int i=0; i<SEG_NET_OUTPUT_SIZE; i++)
    {
        if(jawRow.at<uchar>(i) == SEG_FACE_LABEL)
            jawWidth += 1;
    }
    
    // convert to the source image
    jawWidth = jawWidth * faceInfo.imgWidth / SEG_NET_OUTPUT_SIZE;
    
    return jawWidth;
}
*/
