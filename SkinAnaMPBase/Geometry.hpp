//
//  Geometry.hpp
//
//
/*
本模块提供一些用于2D矢量几何计算的函数。
 
Author: Fu Xiaoqiang
Date:   2022/9/18
*/

#ifndef GEOMETRY_HPP
#define GEOMETRY_HPP

#include "opencv2/opencv.hpp"

using namespace std;
using namespace cv;

#include "Common.hpp"

// contour上的线段
struct LineSegment{
    int     sIndex;
    int     eIndex;
    Point2i sPt;
    Point2i ePt;
    
    LineSegment(int sIndex0, int eIndex0,
                const Point2i& sPt0, const Point2i& ePt0):
            sIndex(sIndex0), eIndex(eIndex0),
            sPt(sPt0), ePt(ePt0)
    {
        
    }
};

// 从头部取出，从尾部加入，环形装置；
// 取出时，将head flag往后拨一格；
// 加入新元素时，将tail flag往后拨一格。
template <class T>
class CircularBuf
{
  private:
    vector<T> buf;
    int headFlag;
    int tailFlag;
    int bufSize; // 满员时的容量
    
  public:
    CircularBuf(int bufSize0)
    {
        bufSize = bufSize0;
        buf.reserve(bufSize);
        headFlag = -1; // !!!
        tailFlag = 0;
    }
    
    // return how much elements stored in the buf now
    int elementNum() const
    {
        return (tailFlag + bufSize - headFlag) % bufSize;
    }
    
    bool isFull()
    {
        if(tailFlag == headFlag)
            return true;
        else
            return false;
    }
    
    // if index is a negative value, counting from tail
    // if index is zero or a positive value, coutning from head
    T getElement(int index) const
    {
        if(index >= 0) // starting from head and iterating forward
        {
            int actIndex = (headFlag + index) % bufSize;
            return buf[actIndex];
        }
        else  // starting from tail and iterating backward
        {
            int actIndex = (tailFlag + index + bufSize) % bufSize;
            return buf[actIndex];
        }
    }
    
    // 如果tail位置为0的话，head位置应该是什么
    int headReverseIndex() const
    {
        return headFlag - tailFlag;
    }
    
    T popFront()
    {
        T popElement = buf[headFlag];
        headFlag = (headFlag+1) % bufSize;
        
        if(headFlag == tailFlag) // become empty
        {
            headFlag = -1; // !!!
            tailFlag = 0;
        }
        return popElement;
    }
    void pushBack(T& t)
    {
        if(headFlag == -1)  // add the first element
            headFlag = 0;
        
        buf[tailFlag] = t;
        tailFlag = (tailFlag + 1) % bufSize;
    }
};

/**********************************************************************************************
(x3, y3) is the inner interpolated result.
t: in range[0.0 1.0].
when t --> 0.0, then (x3, y3) --> (x1, y1);
when t --> 1.0, then (x3, y3) --> (x2, y2);
***********************************************************************************************/
void Interpolate(int x1, int y1, int x2, int y2, float t, int& x3, int& y3);

Point2i Interpolate(int x1, int y1, int x2, int y2, float t);

Point2i Interpolate(const Point2i& p1, const Point2i& p2, float t);

/**********************************************************************************************
记返回的点为p3. p3.y与p1.y值保持一致，p3.x由p1.x和p2.x内插而来
t: in range[0.0 1.0].
when t --> 0.0, then p3.x --> p1.x;
when t --> 1.0, then p3.x --> p2.x;
***********************************************************************************************/
Point2i InterpolateX(const Point2i& p1, const Point2i& p2, float t);

// as similar with InterpolateX()
Point2i InterpolateY(const Point2i& p1, const Point2i& p2, float t);

//-------------------------------------------------------------------------------------------
// oriPt + (dx, dy) ---> newPt
void MovePolygon(const POLYGON& oriPg, int dx, int dy, POLYGON& newPg);

// transform the contour(abbr. Ct) from Net Output Space to Source Space
// nosLocCt: contour represented in local space cropped by bbox from the global NOS.
// nosTlPt: the Top Left corner of BBox of contour in the NOS.
// scaleUpX, scaleUpY: the scale up ratio in X-axis and Y-axis.
void transCt_LocalSegNOS2SS(const CONTOUR& nosLocCt, const Point& nosBBoxTlPt,
                            Size srcImageS, CONTOUR& spCt);

// transform the contour(abbr. Ct) from Seg Net Output Space to Source Space
// nosGlobalCt: contour presented in global SegNOS.
// scaleUpX, scaleUpY: the scale up ratio in X-axis and Y-axis.
void transCt_GlobalSegNOS2SS(const CONTOUR& nosGlobalCt,
                             double scaleUpX, double scaleUpY, CONTOUR& spCt);

//SMS: sub-mask space
// nosBBoxTLPt: the top left corner point of sub-mask in the NOS
void transCt_SMS2NOS(const CONTOUR& smsCt, const Point& nosBBoxTLPt,
                    CONTOUR& nosCt);

/**********************************************************************************************
Ip: interpolate
GLm: general landmark
Pt: point
t: lies in [0.0 1.0], t-->0, out point-->Pt1;
***********************************************************************************************/
Point2i IpGLmPtWithPair(const FaceInfo& faceInfo, int pIndex1, int pIndex2, float t);

Point2i getPtOnGLm(const FaceInfo& faceInfo, int pIndex);

//-------------------------------------------------------------------------------------------
// return a new point, which with x from p1, y from p2
Point2i getRectCornerPt(const Point2i& p1, const Point2i& p2);

//-------------------------------------------------------------------------------------------
// 如果smallRect完全能被bigRect容纳，返回true
bool RectContainsRect(const Rect& bigRect, const Rect& smallRect);

// 在refRect构成的局部坐标系（以左上角为原点，y轴朝下）中，计算transRect的新版本
Rect CalcRelativeRect(const Rect& refRect, const Rect& transRect);

// return the corrected point which lies in the rectangle.
Point2i MakePtInRect(const Rect& rect, Point2i& pt);

// inflate the original rect from the center and toward all sides.
void InflateRect(int inflateSize, Rect& rect);

float DisBetw2Pts(const Point& pt1, const Point& pt2);
float LenOfVector(const Point& vect);

//-------------------------------------------------------------------------------------------
float AvgPointDist(const CONTOUR& cont);

// 将contour多余的点（指挨得太近的点）删除掉
// 思路：计算contour上平均的点距，将小于alpha倍平均点距的点给依次删掉。
void SparsePtsOnContour(const CONTOUR& oriCont, float alpha, CONTOUR& sparCont);

// 1st version: just check there is a tie existed or not
bool CheckTieOnContour(const CONTOUR& oriCont, int lineSegBufSize);

// 1st version: just check there is a cross existed between lineSeg and
// any line segment of lineSegBuf.
// includeFinalSeg indicates whether the last one of lineSegBuf will be considered or not.
bool CheckCrossOfLineSegs(const LineSegment& lineSeg,
                       const list<LineSegment>& lineSegBuf,
                       bool includeLastSeg=false);

bool CheckCrossOfTwoLineSegs(const LineSegment& lineSeg1, const LineSegment& lineSeg2);

#endif /* end of GEOMETRY_HPP */
