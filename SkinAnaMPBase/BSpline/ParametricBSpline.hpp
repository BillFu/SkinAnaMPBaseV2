//
//  ParametricBSpline.hpp
//
//
/*
本模块的功能在于，在github.com/ttk592/spline的基础上，将parametric bspline进行封装改造，
使之很容易地生成闭合、光滑的BSpline曲线。
 
Author: Fu Xiaoqiang
Date:   2022/9/16
*/

#ifndef PARAMETRIC_BSPLINE_HPP
#define PARAMETRIC_BSPLINE_HPP

#include "opencv2/opencv.hpp"

using namespace std;
using namespace cv;

#include "../Common.hpp"


/**********************************************************************************************
本函数的作用是，传入粗糙的折线多边形（或不封闭的折线），构造出光滑的多边形、折线，且增加点数使之稠密。
csPolygon: closed, opened BSpline polygon
***********************************************************************************************/

void DenseSmoothPolygon(const POLYGON& contours, int csNumPoint,
                        POLYGON& csPolygon, bool is_closed_curve = true);


//-------------------------------------------------------------------------------------------


#endif /* end of PARAMETRIC_BSPLINE_HPP */
