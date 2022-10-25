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
// polePt: the origin of Pole Coordinate System, also called Pole Point.
// scanDir: CW or CCW when to scan in the natural order of curve
void CalcPolarSeqOnCurve(const CONTOUR& curve, const Point2i& polePt,
                         CLOCK_DIR scanDir,
                      PolarContour& polarSeq);

void CalcPolarSeqOnCurveCCW(const CONTOUR& curve, const Point2i& polePt,
                      PolarContour& polarSeq);

void CalcPolarSeqOnCurveCW(const CONTOUR& curve, const Point2i& polePt,
                      PolarContour& polarSeq);

void IpPolarSeqEvenlyCCW(const PtInPolarSeq& ascThetaSeq, int numInterval,
                        const Point& oriPt,
                        PolarContour& evenPolarSeq);

void IpPolarSeqEvenlyCW(const PtInPolarSeq& ascThetaSeq, int numInterval,
                        const Point& oriPt,
                        PolarContour& evenPolarSeq);

// -------------------------------------------------------------------------------------------

// extract the primary and coarse polar sequence on the contour
void CalcPolarSeqOnCt(const CONTOUR& spCt, const Point2i& eyeCP,
                      PolarContour& polarCont);

// build a new version of polar point sequence that with even intervals of theta.
void BuildEvenPolarSeq(const PolarContour& rawPolarSeq,
                       int num_interval, // how many intervals from 0 to 2*Pi
                       PolarContour& evenPolarSeq);

void SmClosedPolarSeq(const PolarContour& evenPolarSeq,
                      int mwLen, //length of moving window
                      PolarContour& smoothPolarSeq);

// 针对的是不封闭的弧线
void SmOpenPolarSeqV2(const PolarContour& evenPolarSeq,
                      int mwLen, //length of moving window
                      PolarContour& smoothPolarSeq);

// 在两个端点处，以复制的方式来凑足窗口所需点数
void SmOpenPolarSeqV3(const PolarContour& evenPolarSeq,
                      int mwLen,
                      PolarContour& smoothPolarSeq);

Point PolarPt2CartPt(const PtInPolarCd& polarPt, const Point& oriPt); //, Point& cartPt);

void PolarPtSeq2CartPtSeq(const PolarContour& polarPtSeq,
                          CONTOUR& cartCont);

#endif /* end of POLAR_PT_SEQ_HPP */
