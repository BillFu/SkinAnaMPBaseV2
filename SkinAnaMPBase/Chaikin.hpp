//
//  Chaikin.hpp
//
//
/*
本模块使用chaikin算法对折线多边形进行光滑处理。
// https://zhuanlan.zhihu.com/p/380519655

Author: Fu Xiaoqiang
Date:   2022/11/2
*/

#ifndef CHAIKIN_HPP
#define CHAIKIN_HPP

#include "opencv2/opencv.hpp"

using namespace std;
using namespace cv;

#include "Common.hpp"

//------目前版本只针对封闭多边形进行光滑，断开的折线不在目前的考虑范围之内-----------------------
// CK: chaikin
void SmoothClosedContourCK(const CONTOUR& oriCt, int shortThresh,
                     int iterTimes, CONTOUR& smCt);

void SmoothClosedOnceCK(const CONTOUR& inCt, int shortThresh,
                  CONTOUR& outCt);

void SmUnclosedContCK(const CONTOUR& oriCt,
                     int iterTimes, CONTOUR& smCt);

void SmUnclosedOnceCK(const CONTOUR& inCt,
                  CONTOUR& outCt, int K);


#endif /* end of CHAIKIN_HPP */
