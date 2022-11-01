#ifndef ACTIVE_CONTOURS_H
#define ACTIVE_CONTOURS_H

// Contains #define for algo info
#include "tsparams.h"

#include <set>
#include <vector>
#include <iterator>
#include <iostream>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "sobel.h"
#include "../Common.hpp"


/*
    E = ∫(αEcont + βEcurv + γEimage)ds
*/

/*
    Self point-insertion methods,
    not required to run
*/
// #define USE_ANGLE_INSERTION 1
// #define USE_AVERAGE_LENGTH_BISECTION 1

/*
    Angle intersection inserts
    a new point if the angle generated
    by i-1, i, i+1 is > lower bound
    and < upper bound
*/
#ifdef USE_ANGLE_INSERTION
    #define ANGLE_WAIT_SEC 1
    #define ANGLE_LOWER_BOUND 0.42
    #define ANGLE_UPPER_BOUND 1.2
#endif

/*
    Average length inserts the a new point
    between two if the length between them
    are
*/

namespace cvalg
{

using namespace cv;
using namespace std;

class ActiveContours
{
public:
    ActiveContours();
    void init(int i_width, int i_height);
    void setInitCont(const CONTOUR& initCont);
    
    void optimize(const Mat& inImg, const Mat& cornerField,
                  int viewRadius, int iterTimes);

    bool minimumRunReqSet();
    Mat drawSnake(Mat frame);
    
    void setParams(AlgoParams* params);

    bool previously_reset;
    
    vector<Point> getOptimizedCont();

private:
    AlgoParams* _params;

    int _w, _h;
    Point _center;
    CONTOUR _points;

    float _alpha;
    float _beta;
    float _gamma;
    float _avgDist;

    Point2i GetPrevPt(int ptIndex);
    Point2i GetNextPt(int ptIndex);

    Point updatePos(int ptIndex, Point start, Point end,
                    const Mat& edgeImage, const Mat& cornerField,
                    float MaxEcont, float MaxEcurv,
                    vector<float>& EcontRec,
                    vector<float>& EcurvRec,
                    vector<float>& EedgeRec,
                    vector<float>& EcorRec,
                    vector<float>& EtotalRec);
    
    Point updatePosV2(int ptIndex, Point start, Point end,
                    const Mat& edgeImage, const Mat& cornerField,
                    float MaxEcont, float MaxEcurv,
                    float& destEcont, float& destEcurv,
                    float& destEedge, float& destEcor);
    
    //void AvgPointDist();
    
    // 估算E的各分量的大致取值（是各点取最大值，还是取平均值？）
    // 这个操作在优化开始前利用初始数据来完成
    void estiEnergeComponents(float& MaxEcont, float& MaxEcurv);
    
    float CalcEcont(const Point2i& prevPt,
                    const Point2i& neibPt, float avgDist);
    
    // Ecurv = (x[i-1] - 2x[i] + x[i+1])^2 + (y[i-1] - 2y[i] + y[i+1])^2
    float CalcEcurv(const Point2i& prevPt,
                    const Point2i& neibPt, const Point2i& nextPt);
    
    void ShowECompData(vector<float>& EcontRec,
                       vector<float>& EcurvRec,
                       vector<float>& EedgeRec,
                       vector<float>& EcorRec);
    
    void BuildNeigArea(int ptIndex, int viewRadius,
                       Point2i& startPt, Point2i& endPt);

};
}

void ShowMaxMinInRec(const std::vector<float>& rec,
                     const std::string& title);


#endif // ACTIVE_CONTOURS_H

