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

void SmoothPolarPtSeq(const PolarContour& evenPolarSeq,
                      int mwLen, //length of moving window
                      PolarContour& smoothPolarSeq);

//---------------------Polar coordinate system not used in the following functions method,
//---------------------Cartesian coordinate system used instead--------------------------
// P1: the top left corner on the eye contour,
// P2: the top right corner on the eye contour.
// Here we assumed that one closed eye has a shape like meniscus(弯月形)
// 有可能因为眼睛相对于相机的Pose差异，而出现上弯月（凹陷朝上）和下弯月（凹陷朝下）之分。
void CalcEyeCtP1P2(const CONTOUR& eyeCont, const Point& eyeCP, Point& P1, Point& P2);

// P4: the top middle point on the eye contour,
// P3: the bottom middle point on the eye contour.
void CalcEyeCtP3P4(const CONTOUR& eyeCont, const Point& eyeCP, Point& P3, Point& P4);

#endif /* end of POLAR_PT_SEQ_HPP */
