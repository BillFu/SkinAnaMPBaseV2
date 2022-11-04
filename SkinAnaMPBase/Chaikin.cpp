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
    // Q接近P1, R接近P2
    Q.x = (int)(0.75 * p1.x + 0.25 * p2.x);
    Q.y = (int)(0.75 * p1.y + 0.25 * p2.y);
    
    R.x = (int)(0.25 * p1.x + 0.75 * p2.x);
    R.y = (int)(0.25 * p1.y + 0.75 * p2.y);
}


void SmoothClosedOnceCK(const CONTOUR& inCt,
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
void SmoothClosedContourCK(const CONTOUR& oriCt,
                     int shortThresh, int iterTimes, CONTOUR& smCt)
{
    CONTOUR inCt(oriCt);
    CONTOUR outCt;
    for(int k=1; k<=iterTimes-1; k++)
    {
        SmoothClosedOnceCK(inCt, shortThresh, outCt);
        inCt = outCt;
        outCt.clear();
    }
    
    SmoothClosedOnceCK(outCt, shortThresh, smCt);
}

void SmUnclosedContCK(const CONTOUR& oriCt,
                     int iterTimes, CONTOUR& smCt)
{
    CONTOUR inCt; //(oriCt);
    
    for(int i=0; i<oriCt.size(); )
    {
        inCt.push_back(oriCt[i]);
        i += 2;
    }
    
    CONTOUR outCt;
    for(int k=1; k<=iterTimes-1; k++)
    {
        SmUnclosedOnceCK(inCt, outCt, k);
        inCt.clear();
        
        for(int i=0; i<outCt.size(); )
        {
            inCt.push_back(outCt[i]);
            i += 2;
        }
            
        outCt.clear();
    }
    
    SmUnclosedOnceCK(inCt, smCt, iterTimes);
}


void SmUnclosedOnceCK(const CONTOUR& inCt,
                  CONTOUR& outCt, int K)
{
    int inNumPt = static_cast<int>(inCt.size());
    
    //float avgDis = AvgPointDist(inCt);
    //float shortThresh = avgDis * 0.5;
    //if(shortThresh < 3)
    //    shortThresh = 3;
    
    //outCt.reserve(inNumPt * 2);
    //outCt.push_back(inCt[0]);

    for(int i=0; i<=inNumPt-2; i++)
    {
        int j = i+1;
        
        //float d0 = DisBetw2Pts(inCt[i], inCt[j]);
        ///if(d0 <= shortThresh)
            //continue;

        Point2i Q, R;
        GenQRPtsOnArc(inCt[i], inCt[j], Q, R);
        
        outCt.push_back(Q);
        outCt.push_back(R);
    }
    
    //outCt.push_back(inCt[inNumPt-1]);

    Mat canvas(3264, 2448, CV_8UC1, Scalar(0));
    for(int i = 0; i < outCt.size(); i++)
    {
        cv::circle(canvas, outCt[i], 1, Scalar(150), cv::FILLED);
    }
    cv::putText(canvas, to_string(0), outCt[0],
                FONT_HERSHEY_SIMPLEX, 1, Scalar(150), 1);
    cv::putText(canvas, to_string(outCt.size()-1), outCt[outCt.size()-1],
                FONT_HERSHEY_SIMPLEX, 1, Scalar(150), 1);
    
    string filename = to_string(K) + ".png";
    imwrite(filename, canvas);
}
