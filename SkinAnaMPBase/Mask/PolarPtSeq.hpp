//
//  PolarPtSeq.hpp
//
//
/*
本模块采用极坐标(r, theta)来表示contour，由此带来的一系列处理。
旋转扫描轮廓一周，获得轮廓点序列，再对r序列按照均匀theta间隔进行插值，
并光滑，最后加以伸长，再由极坐标返回到笛卡尔坐标，获得我们想要的轮廓。
 
Author: Fu Xiaoqiang
Date:   2022/10/12
*/

#ifndef POLAR_PT_SEQ_HPP
#define POLAR_PT_SEQ_HPP

#include "opencv2/opencv.hpp"

using namespace std;
using namespace cv;

#include "../Common.hpp"


/**********************************************************************************************
only cover the two eyes
***********************************************************************************************/
// extract the primary and coarse polar sequence on the contour
void CalcPolarSeqOnCt(const CONTOUR& spCt, const Point2i& eyeCP,
                      PolarContour& polarCont);

// build a new version of polar point sequence that with even intervals of theta.
void BuildEvenPolarSeq(const PolarContour& rawPolarSeq,
                       int num_interval, // how many intervals from 0 to 2*Pi
                       PolarContour& evenPolarSeq);

void smoothPolarPtSeq(const PolarContour& evenPolarSeq,
                      int mwLen, //length of moving window
                      PolarContour& smoothPolarSeq);

#endif /* end of POLAR_PT_SEQ_HPP */
