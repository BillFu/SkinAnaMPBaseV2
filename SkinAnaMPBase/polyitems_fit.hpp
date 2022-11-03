
/*
polyitems_fit.hpp

本模块用于对点序列进行二次曲线拟合。
OpenCV（C++）的曲线拟合polyfit
https://blog.csdn.net/jpc20144055069/article/details/103232641

Author: Fu Xiaoqiang
Date:   2022/10/24
*/

#ifndef POLY_ITEMS_FIT_HPP
#define POLY_ITEMS_FIT_HPP

#include <filesystem>

#include "opencv2/opencv.hpp"

using namespace std;
using namespace cv;

#include "Common.hpp"

/**********************************************************************************************
通过曲线拟合而对轮廓线进行光滑
***********************************************************************************************/
void SmoothCtByPIFit(const CONTOUR inCt, CONTOUR& outCt);

#endif /* end of POLY_ITEMS_FIT_HPP */
