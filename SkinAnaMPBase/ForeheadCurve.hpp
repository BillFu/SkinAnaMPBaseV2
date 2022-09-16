//
//  ForeheadCurve.hpp
//
//
/*
本模块最初的功能是，将额头的轮廓线适当地抬高一些。
 
Author: Fu Xiaoqiang
Date:   2022/9/16
*/

#ifndef FOREHEAD_CURVE_HPP
#define FOREHEAD_CURVE_HPP

#include "opencv2/opencv.hpp"

using namespace std;
using namespace cv;

// determine whether one Point(represented by index) on forehead curve
bool isPtOnForeheadCurve(int ptIndex);

/**********************************************************************************************
前额顶部轮廓线由9个lm点组成。
Input: lm_2d
Output: raisedForeheadCurve
 alpha: [0.0 1.0]，the greater this value is, the more raised up
***********************************************************************************************/
void RaiseupForeheadCurve(const int lm_2d[468][2], int raisedFhCurve[9][2],
                          float alpha);

//-------------------------------------------------------------------------------------------



#endif /* end of FOREHEAD_CURVE_HPP */
