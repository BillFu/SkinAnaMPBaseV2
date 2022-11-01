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
void transCt_LocalSegNOS2SS(const CONTOUR& nosLocCt, const Point& nosBBoxTlPt,
                            Size srcImageS, CONTOUR& spCt)
{
    for(Point nosLocPt: nosLocCt)
    {
        Point nosPt = nosLocPt + nosBBoxTlPt;
        
        int spX = nosPt.x * srcImageS.width / SEG_NET_OUTPUT_SIZE;
        int spY = nosPt.y * srcImageS.height / SEG_NET_OUTPUT_SIZE;
        
        spCt.push_back(Point(spX, spY));
    }
}

// transform the contour(abbr. Ct) from Seg Net Output Space to Source Space
// nosGlobalCt: contour presented in global SegNOS.
// scaleUpX, scaleUpY: the scale up ratio in X-axis and Y-axis.
void transCt_GlobalSegNOS2SS(const CONTOUR& nosGlobalCt,
                             double scaleUpX, double scaleUpY,
                             CONTOUR& spCt)
{
    for(Point nosPt: nosGlobalCt)
    {
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

//-------------------------------------------------------------------------------------------
// 如果smallRect完全能被bigRect容纳，返回true
bool RectContainsRect(const Rect& bigRect, const Rect& smallRect)
{
    if((bigRect & smallRect) == smallRect)
        return true;
    else
        return false;
}


// 在refRect构成的局部坐标系（以左上角为原点，y轴朝下）中，计算transRect的新版本
Rect CalcRelativeRect(const Rect& refRect, const Rect& transRect)
{
    Point tlRelCd = transRect.tl() - refRect.tl();
    Rect relRect(tlRelCd, transRect.size());
    
    return relRect;
}


// return the corrected point which lies in the rectangle.
Point2i MakePtInRect(const Rect& rect, Point2i& pt)
{
    if(rect.contains(pt))
        return pt;
    
    int correctX = pt.x;
    int correctY = pt.y;
    
    if(correctX < 0)
        correctX = 0;
    if(correctX >= rect.width)
        correctX = rect.width - 1;
    
    if(correctY < 0)
        correctY = 0;
    if(correctY >= rect.height)
        correctY = rect.height - 1;
    
    return Point2i(correctX, correctY);
}


// inflate the original rect from the center and toward all sides.
void InflateRect(int inflateSize, Rect& rect)
{
    rect += cv::Point(-inflateSize, -inflateSize);
    rect += cv::Size(inflateSize*2, inflateSize*2);
    
    // cv::Rect has the + (and other) operators overloaded such that
    // if it's a cv::Point it will adjust the origin,
    // and if it's a cv::Size it will adjust width and height.
}

//-------------------------------------------------------------------------------------------

float DisBetw2Pts(const Point& pt1, const Point& pt2)
{
    Point dPt = pt1 - pt2;
    float dis = sqrt(dPt.x * dPt.x + dPt.y * dPt.y);
    return dis;
}

float LenOfVector(const Point& vect)
{
    float length = sqrt(vect.x * vect.x + vect.y * vect.y);
    return length;
}

float AvgPointDist(const CONTOUR& cont)
{
    int numPt = static_cast<int>(cont.size());
    if(numPt <= 1)
        return 0.0;
    
    float sum = 0.0;
    for(int i = 0; i <=numPt-1; i++)
    {
        int nextID = (i + 1) % numPt;
        float dis = DisBetw2Pts(cont[i], cont[nextID]);
        sum += dis;
    }
    
    float avgDist = sum / numPt;
    return avgDist;
}

// Ip: abbr. of Interpolate
void IpPtViaS(const Point2i& uPt, const Point2i& lPt,
              float upS, float lowS, float interS, Point2i& outPt)
{
        
    double t = (interS - lowS) / (upS - lowS);
    outPt = Interpolate(lPt, uPt, t);
}

// S: accumulated arc length
void MakePtsEvenAlongWithS(const CONTOUR& oriCont, int newNumPt, CONTOUR& evenCont)
{
    // 第一步，统计累计弧长分布
    int numPt = static_cast<int>(oriCont.size());

    vector<float> STable; // 存贮绝对数值，而非相对百分比
    STable.reserve(numPt+1);
    STable.push_back(0.0);
    float S = 0.0;
    
    for(int i=1; i<=numPt; i++)
    {
        int curIdx;
        int prevIdx = i - 1;
        if(i < numPt)
            curIdx = i;
        else
            curIdx = 0;
        
        float dis = DisBetw2Pts(oriCont[prevIdx], oriCont[curIdx]);
        S += dis;
        STable.push_back(S);
    }
    
    double interS = 0.0;
    double SStep = S / newNumPt;
    
    evenCont.push_back(oriCont[0]);

    int lID = 0;
    int uID = 1;
    while(interS < STable[numPt])
    {
        float SUpLimit = STable[uID];
        float SLowLimt = STable[lID];
        while(interS < SUpLimit)
        {
            interS += SStep;
            Point2i evenPt;
            IpPtViaS(oriCont[uID], oriCont[lID],
                     SUpLimit, SLowLimt, interS, evenPt);

            evenCont.push_back(evenPt);
        }
        
        lID = uID;
        uID++;
    }
}
