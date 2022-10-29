#ifndef ACTIVE_CONTOURS_H
#define ACTIVE_CONTOURS_H

// Contains #define for algo info
#include "tsparams.h"

#ifdef ACTIVE_CONTOUR_ALG
#include <set>
#include <vector>
#include <iterator>
#include <iostream>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "sobel.h"
#include "bresenham.h"
#include "gaussarea.h"


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
#ifdef USE_AVERAGE_LENGTH_BISECTION
    #define BISECT_WAIT 4
#endif

namespace cvalg
{

using namespace cv;
using namespace std;

class ActiveContours
{
public:
    ActiveContours();
    void init(int i_width, int i_height);
    void insertPoint(Point p);
    
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
    vector<Point> _points;

    float _alpha;
    float _beta;
    float _gamma;
    float _avgDist;

    Point updatePos(int pointIndex, Point start, Point end,
                    const Mat& edgeImage, const Mat& cornerField);
    void AvgPointDist();

    // Splits snake by a couple of different
    // methods
#ifdef USE_ANGLE_INSERTION
    clock_t _lastAngle;
    void angleInsertion();
#endif

#ifdef USE_AVERAGE_LENGTH_BISECTION
    clock_t _lastBisect;
    void selfInsertion();
#endif

};
}

#endif // IFDEF ACTIVE_CONTOUR_ALG

#endif // ACTIVE_CONTOURS_H

