//
//  SmEyePg.hpp
//
//
/*
本模块采用二次多项式拟合的方式，来光滑眼睛的轮廓线。
代码在EyebrowMaskV5模块的部分内容上“翻新”而来。
 
Author: Fu Xiaoqiang
Date:   2022/11/2
*/

#ifndef SM_EYE_PG_HPP
#define SM_EYE_PG_HPP

#include "opencv2/opencv.hpp"

using namespace std;
using namespace cv;

#include "Common.hpp"

/*
将眼睛轮廓线分割为上、下弧线。
原始数据必须在同一坐标系下。输入的左右角点不一定就在轮廓线上，需要找出距离它们最近的两点作为分裂点。
 */
void SplitEyeCt(const CONTOUR& EyeCont,
                const Point2i& LCorPt,
                const Point2i& RCorPt,
                CONTOUR& upEyeCurve,
                CONTOUR& lowEyeCurve);

void FindNearestPtOnCt(const CONTOUR& EyeCont, const Point2i& refPt,
                       int& nearestIdx, Point2i& nearestPt);

void SmCurveByFit(const CONTOUR& EyeCont, CONTOUR& smCont);

#endif /* end of SM_EYE_PG_HPP */
