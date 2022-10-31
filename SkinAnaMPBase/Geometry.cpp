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


// 将contour多余的点（指挨得太近的点）删除掉
// 思路：计算contour上平均的点距，将小于alpha倍平均点距的点给依次删掉。
void SparsePtsOnContour(const CONTOUR& oriCont, float alpha, CONTOUR& sparCont)
{
    float avgDist = AvgPointDist(oriCont);
    float shortThresh = avgDist * alpha;
    
    int numPt = static_cast<int>(oriCont.size());

    for(int i=0; i<numPt; i++)
    {
        int prevID = (i + numPt - 1) % numPt;
        Point2i prevPt = oriCont[prevID];
        int nextID = (i + 1) % numPt;
        Point2i nextPt = oriCont[nextID];
        
        float dist0 = DisBetw2Pts(prevPt, oriCont[i]);
        float dist1 = DisBetw2Pts(oriCont[i], nextPt);
        
        if(dist0 <= shortThresh) // && dist1 <= shortThresh)
        {
            cout << "a point deleted!" << endl;
            continue;
        }
        else
        {
            sparCont.push_back(oriCont[i]);
        }
    }
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

//-------------------------------------------------------------------------------------------
int crossProduct(const Point2i& vect1, const Point2i& vect2)
{
    return (vect1.x * vect2.y - vect1.y * vect2.x);
}

bool CheckCrossOfTwoLineSegs(const LineSegment& lineSeg1, const LineSegment& lineSeg2)
{
    // <a, b> is lineSeg1; <c, d> is lineSeg2
    const Point2i& a = lineSeg1.sPt;
    const Point2i& b = lineSeg1.ePt;
    const Point2i& c = lineSeg2.sPt;
    const Point2i& d = lineSeg2.ePt;
    
    //快速排斥实验
    if( max(c.x, d.x) < min(a.x, b.x)||
        max(a.x, b.x) < min(c.x, d.x)||
        max(c.y, d.y) < min(a.y, b.y)||
        max(a.y, b.y) < min(c.y, d.y))
    {
        return false;
    }
    //跨立实验
    if(   crossProduct(a-d,c-d)*crossProduct(b-d,c-d)>0
       || crossProduct(d-b,a-b)*crossProduct(c-b,a-b)>0)
    {
        return  false;
    }
    
    return true;
}


// 1st version: just check there is a cross existed between lineSeg and
// any line segment of lineSegBuf.
// includeFinalSeg indicates whether the last one of lineSegBuf will be considered or not.
bool CheckCrossOfLineSegs(const LineSegment& lineSeg,
                          const CircularBuf<LineSegment>& lineSegBuf,
                          bool includeLastSeg)
{
    if(!includeLastSeg)
    {
        // the processing needs that there are at least two line segs existed
        if(lineSegBuf.elementNum() <= 1)
            return false;
        
        int headRevIndex = lineSegBuf.headReverseIndex();
        for(int i = -2; i>= headRevIndex; i--)
        {
            LineSegment lineSeg2 = lineSegBuf.getElement(i);
            bool isIntersect = CheckCrossOfTwoLineSegs(lineSeg, lineSeg2);
            if(isIntersect)
                return true;
        }
    }
    else
    {
        if(lineSegBuf.elementNum() <= 0)
            return false;
        
        int headRevIndex = lineSegBuf.headReverseIndex();
        for(int i = -1; i >= headRevIndex; i--)
        {
            LineSegment lineSeg2 = lineSegBuf.getElement(-i);
            bool isIntersect = CheckCrossOfTwoLineSegs(lineSeg, lineSeg2);
            if(isIntersect)
                return true;
        }
    }
        
    return false;
}

// 1st version: just check there is a tie existed or not
bool CheckTieOnContour(const CONTOUR& oriCont, int lineSegBufSize)
{
    CircularBuf<LineSegment> lineSegBuf(lineSegBufSize);
    
    int numPt = static_cast<int>(oriCont.size());
    if(numPt <= 3)
        return false;

    for(int i=0; i <=numPt-1; i++)
    {
        int j = (i + 1) % numPt;
        LineSegment curLineSeg(i, j, oriCont[i], oriCont[j]);
        
        bool hasCross = CheckCrossOfLineSegs(curLineSeg, lineSegBuf, false);
        if(hasCross)
            return true;
        
        lineSegBuf.pushBack(curLineSeg);
    }
    
    return false;
}
