//
//  Geometry.cpp

/*******************************************************************************
本模块提供一些用于2D矢量几何计算的函数。

Author: Fu Xiaoqiang
Date:   2022/9/18

********************************************************************************/

#include "Geometry.hpp"


/**********************************************************************************************
(x3, y3) is the inner interpolated result.
t: in range[0.0 1.0].
when t --> 0.0, then (x3, y3) --> (x1, y1);
when t --> 1.0, then (x3, y3) --> (x2, y2);

***********************************************************************************************/
void InnerInterpolate(int x1, int y1, int x2, int y2, float t, int& x3, int& y3)
{
    x3 = (int)(x1 + (x2 - x1) * t);
    y3 = (int)(y1 + (y2 - y1) * t);
}

Point2i InnerInterpolate(int x1, int y1, int x2, int y2, float t)
{
    int x3, y3;
    InnerInterpolate(x1, y1, x2, y2, t, x3, y3);

    return Point2i(x3, y3);
}

Point2i InnerInterpolate(const Point2i& p1, const Point2i& p2, float t)
{
    Point2i p3 = InnerInterpolate(p1.x, p1.y, p2.x, p2.y, t);
    return p3;
}
//-------------------------------------------------------------------------------------------
