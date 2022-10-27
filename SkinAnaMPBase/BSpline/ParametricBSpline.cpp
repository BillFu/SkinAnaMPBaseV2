//
//  ParametricBSpline.cpp

/*******************************************************************************
本模块的功能在于，在github.com/ttk592/spline的基础上，将parametric bspline进行封装改造，
使之很容易地生成闭合、光滑的BSpline曲线。

Author: Fu Xiaoqiang
Date:   2022/9/16

********************************************************************************/

#include "spline.h"
#include "ParametricBSpline.hpp"

double sqr(double x)
{
    return x*x;
}

// this function is copied from github.com/ttk592/spline
void create_time_grid(std::vector<double>& T, double& tmin, double& tmax,
                      std::vector<double>& X, std::vector<double>& Y, bool is_closed_curve)
{

    assert(X.size()==Y.size() && X.size()>2);

    // hack for closed curves (so that it closes smoothly):
    //  - append the same grid points a few times so that the spline
    //    effectively runs through the closed curve a few times
    //  - then we only use the last loop
    //  - if periodic boundary conditions were implemented then
    //    simply setting periodic bd conditions for both x and y
    //    splines is sufficient and this hack would not be needed
    int idx_first=-1, idx_last=-1;
    if(is_closed_curve) {
        // remove last point if it is identical to the first
        if(X[0]==X.back() && Y[0]==Y.back()) {
            X.pop_back();
            Y.pop_back();
        }

        const int num_loops=3;  // number of times we go through the closed loop
        std::vector<double> Xcopy, Ycopy;
        for(int i=0; i<num_loops; i++) {
            Xcopy.insert(Xcopy.end(), X.begin(), X.end());
            Ycopy.insert(Ycopy.end(), Y.begin(), Y.end());
        }
        idx_last  = (int)Xcopy.size()-1;
        idx_first = idx_last - (int)X.size();
        X = Xcopy;
        Y = Ycopy;

        // add first point to the end (so that the curve closes)
        X.push_back(X[0]);
        Y.push_back(Y[0]);
    }

    // setup a "time variable" so that we can interpolate x and y
    // coordinates as a function of time: (X(t), Y(t))
    T.resize(X.size());
    T[0]=0.0;
    for(size_t i=1; i<T.size(); i++) {
        // time is proportional to the distance, i.e. we go at a const speed
        T[i] = T[i-1] + sqrt( sqr(X[i]-X[i-1]) + sqr(Y[i]-Y[i-1]) );
    }
    if(idx_first<0 || idx_last<0) {
        tmin = T[0] - 0.0;
        tmax = T.back() + 0.0;
    } else {
        tmin = T[idx_first];
        tmax = T[idx_last];
    }
}

//-------------------------------------------------------------------------------------------

/*******************************************************************************************
 本函数的作用是，传入粗糙的折线多边形，构造出光滑、封闭的多边形。
 csPolygon: closed, opened BSpline polygon
 csNumPoint: how many points should be output to present csPolygon
 *******************************************************************************************/

void DenseSmoothPolygon(const POLYGON& contours, int csNumPoint,
                        POLYGON& csPolygon, bool is_closed_curve)
{
    std::vector<double> inX;
    std::vector<double> inY;
    
    for(Point2i inPt: contours)
    {
        inX.push_back(inPt.x);
        inY.push_back(inPt.y);
    }
    
    tk::spline::spline_type type = tk::spline::cspline;
    bool make_monotonic = false;

    // setup auxiliary "time grid"
    double tmin = 0.0, tmax = 0.0;
    std::vector<double> T;
    create_time_grid(T, tmin, tmax, inX, inY, is_closed_curve);

    // define a spline for each coordinate x, y
    tk::spline sx, sy;
    sx.set_points(T, inX, type);
    sy.set_points(T, inY, type);
    if(make_monotonic)
    {
        // adjusts spline coeffs to be piecewise monotonic where possible
        sx.make_monotonic();
        sy.make_monotonic();
    }

    for(int i=0; i<csNumPoint; i++)
    {
        double t = tmin + (double)i*(tmax-tmin)/(csNumPoint - 1);
        
        int outX = (int)sx(t);
        int outY = (int)sy(t);
        csPolygon.push_back(Point2i(outX, outY));
    }
}

//-------------------------------------------------------------------------------------------
