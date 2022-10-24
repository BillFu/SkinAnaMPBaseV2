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
记返回的点为p3. p3.y与p1.y值保持一致，p3.x由p1.x和p2.x内插而来
t: in range[0.0 1.0].
when t --> 0.0, then p3.x --> p1.x;
when t --> 1.0, then p3.x --> p2.x;
***********************************************************************************************/
Point2i InterpolateX(const Point2i& p1, const Point2i& p2, float t)
{
    Point2i p3;
    
    p3.y = p1.y;
    p3.x = (int)(p1.x + (p2.x - p1.x) * t);
    
    return p3;
}

// as similar with InterpolateX()
Point2i InterpolateY(const Point2i& p1, const Point2i& p2, float t)
{
    Point2i p3;
    
    p3.x = p1.x;
    p3.y = (int)(p1.y + (p2.y - p1.y) * t);
    
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

// oriPt + (dx, dy) ---> newPt
void MovePolygon(const POLYGON& oriPg, int dx, int dy, POLYGON& newPg)
{
    for(Point2i oriPt: oriPg)
    {
        Point2i newPt(oriPt.x + dx, oriPt.y + dy);
        newPg.push_back(newPt);
    }
}

// transform the contour(abbr. Ct) from Net Output Space to Source Space
// nosTlPt: the Top Left corner of BBox of contour in the NOS.
// scaleUpX, scaleUpY: the scale up ratio in X-axis and Y-axis.
// nosLocCt: contour represented in local space cropped by bbox from the global NOS.
void transCt_NOS2SS(const CONTOUR& nosLocCt, const Point& nosBBoxTlPt,
                    float scaleUpX, float scaleUpY,
                    CONTOUR& spCt)
{
    for(Point nosLocPt: nosLocCt)
    {
        Point nosPt = nosLocPt + nosBBoxTlPt;
        
        int spX = nosPt.x * scaleUpX;
        int spY = nosPt.y * scaleUpY;
        
        spCt.push_back(Point(spX, spY));
    }
}


void transCt_SMS2NOS(const CONTOUR& smsCt, const Point& nosBBoxTLPt,
                    CONTOUR& nosCt)
{
    for(Point pt: smsCt)
    {
        Point nosPt = pt + nosBBoxTLPt;
        nosCt.push_back(nosPt);
    }
}
