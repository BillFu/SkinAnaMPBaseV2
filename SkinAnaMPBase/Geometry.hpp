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

//-------------------------------------------------------------------------------------------

/**********************************************************************************************
Ip: interpolate
GLm: general landmark
Pt: point
t: lies in [0.0 1.0], t-->0, out point-->Pt1;
***********************************************************************************************/
Point2i IpGLmPtWithPair(const FaceInfo& faceInfo, int pIndex1, int pIndex2, float t);


Point2i getPtOnGLm(const FaceInfo& faceInfo, int pIndex);

#endif /* end of GEOMETRY_HPP */
