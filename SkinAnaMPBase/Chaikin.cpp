//
//  Chaikin.cpp

/*******************************************************************************
本模块使用chaikin算法对折线多边形进行光滑处理。
// https://zhuanlan.zhihu.com/p/380519655

Author: Fu Xiaoqiang
Date:   2022/11/2

********************************************************************************/

#include "Geometry.hpp"
#include "Chaikin.hpp"

//------目前版本只针对封闭多边形进行光滑，断开的折线不在目前的考虑范围之内-----------------------

void GenQRPtsOnArc(const Point2i& p1, const Point2i& p2, Point2i& Q, Point2i& R)
{
    Q.x = (int)(0.75 * p1.x + 0.25 * p2.x);
    Q.y = (int)(0.75 * p1.y + 0.25 * p2.y);
    
    R.x = (int)(0.25 * p1.x + 0.75 * p2.x);
    R.y = (int)(0.25 * p1.y + 0.75 * p2.y);
}


void SmoothOnceCK(const CONTOUR& inCt,
                  int shortThresh,
                  CONTOUR& outCt)
{
    int inNumPt = static_cast<int>(inCt.size());
    outCt.reserve(inNumPt * 2);
    
    for(int i=0; i<=inNumPt-1; i++)
    {
        int j = (i+1) % inNumPt;
        
        Point2i Q, R;
        GenQRPtsOnArc(inCt[i], inCt[j], Q, R);
        
        // if the distance of Q and R is very short, just keep one,
        // the midway of Q and R can be used.
        float d = DisBetw2Pts(Q, R);
        if(d <= shortThresh)
        {
            Point M = (Q + R) / 2 ;
            outCt.push_back(M);
        }
        else
        {
            outCt.push_back(Q);
            outCt.push_back(R);
        }
    }
}

// CK: chaikin
void SmoothContourCK(const CONTOUR& oriCt,
                     int shortThresh, int iterTimes, CONTOUR& smCt)
{
    CONTOUR inCt(oriCt);
    CONTOUR outCt;
    for(int k=1; k<=iterTimes-1; k++)
    {
        SmoothOnceCK(inCt, shortThresh, outCt);
    }
    
    SmoothOnceCK(outCt, shortThresh, smCt);
}
