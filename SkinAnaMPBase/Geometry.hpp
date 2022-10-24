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
void transCt_NOS2SS(const CONTOUR& nosLocCt, const Point& nosBBoxTlPt,
                    float scaleUpX, float scaleUpY,
                    CONTOUR& spCt);

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

/**********************************************************************************************
segLabels: 512*512
the returned Jaw Width measured in the source space
***********************************************************************************************/

//int CalcLowerJawWidth(const FaceInfo& faceInfo, const Mat& segLabels);

//-------------------------------------------------------------------------------------------

#endif /* end of GEOMETRY_HPP */
