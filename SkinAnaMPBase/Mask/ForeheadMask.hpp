//
//  ForeheadMask.hpp
//
//
/*
本模块最初的功能是，将额头的轮廓线适当地抬高一些。
更名后，增加了ForeheadMask的构建。
ForeheadMask也属于Fundamental Mask的一部分，只是单独放置在这里。
 
Author: Fu Xiaoqiang
Date:   2022/9/23
*/

#ifndef FOREHEAD_MASK_HPP
#define FOREHEAD_MASK_HPP

#include "opencv2/opencv.hpp"
#include "../Common.hpp"

using namespace std;
using namespace cv;

#define NUM_PT_TOP_FH  9   // the number of points on the top line of forehead

// 如果点在前额顶部轮廓线上，返回它在轮廓线点集中的index；否则返回-1
int getPtIndexOfFHCurve(int ptIndex);

/**********************************************************************************************
前额顶部轮廓线由9个lm点组成。
Input: lm_2d
Output: raisedForeheadCurve
 alpha: [0.0 1.0]，the greater this value is, the more raised up
***********************************************************************************************/
void RaiseupForeheadCurve(const Point2i lm_2d[468], int raisedFhCurve[9][2],
                          float alpha);

//void RaiseupForeheadCurve(const Point2i lm_2d[468], Point2i raisedFhCurve[9], float alpha);
// Fh: forehead
void RaiseupFhCurve(const Point2i lm_2d[468],
                          Point2i raisedFhCurve[NUM_PT_TOP_FH],
                          int raisedPtIndices[NUM_PT_TOP_FH],
                    float alpha);
//-------------------------------------------------------------------------------------------
/**********************************************************************************************
Pg: Polygon
***********************************************************************************************/
void ForgeForeheadPg(const FaceInfo& faceInfo, POLYGON& outPolygon);

/**********************************************************************************************

***********************************************************************************************/
void ForgeForeheadMask(const FaceInfo& faceInfo, const Mat& fbBiLab, Mat& outMask);


/**********************************************************************************************
扩展版的前额区域，向下扩展到部分鼻梁区域。
***********************************************************************************************/
void ForgeExpFheadPg(const FaceInfo& faceInfo, POLYGON& outPolygon);

void ForgeExpFhMask(const FaceInfo& faceInfo, const Mat& fbBiLab, Mat& outMask);


#endif /* end of FOREHEAD_MASK_HPP */
