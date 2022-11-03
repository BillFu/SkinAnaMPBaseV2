//
//  SmEyePg.cpp

/*******************************************************************************
本模块采用二次多项式拟合的方式，来光滑眼睛的轮廓线。
代码在EyebrowMaskV5模块的部分内容上“翻新”而来。
  
Author: Fu Xiaoqiang
Date:   2022/11/2
********************************************************************************/
#include <algorithm>
#include "Geometry.hpp"
#include "Utils.hpp"
#include "SmEyePg.hpp"
#include "polyitems_fit.hpp"

//-------------------------------------------------------------------------------
/*
 std::vector<int> v = {5, 2, 8, 10, 9};
 auto minmax = std::minmax_element(v.begin(), v.end());

 std::cout << "minimum element: " << *minmax.first << '\n';
 std::cout << "maximum element: " << *minmax.second << '\n';
 */

void FindNearestPtOnCt(const CONTOUR& EyeCont, const Point2i& refPt,
                       int& nearestIdx, Point2i& nearestPt)
{
    int numPt = static_cast<int>(EyeCont.size());
    
    vector<float> disList;
    disList.reserve(numPt);
    
    for(Point2i pt: EyeCont)
    {
        float dis = DisBetw2Pts(pt, refPt);
        disList.push_back(dis);
    }
    
    nearestIdx = min_element(disList.begin(), disList.end()) - disList.begin();
    nearestPt = EyeCont[nearestIdx];
}

/*
将眼睛轮廓线分割为上、下弧线。
原始数据必须在同一坐标系下。输入的左右角点不一定就在轮廓线上，需要找出距离它们最近的两点作为分裂点。
 */
void SplitEyeCt(const CONTOUR& EyeCont,
                const Point2i& refLCorPt,
                const Point2i& refRCorPt,
                CONTOUR& upEyeCurve,
                CONTOUR& lowEyeCurve)
{
    Point2i actLCorPt, actRCorPt; // act: actual
    int   actLCorIdx, actRCorIdx;
    FindNearestPtOnCt(EyeCont, refLCorPt, actLCorIdx, actLCorPt);
    FindNearestPtOnCt(EyeCont, refRCorPt, actRCorIdx, actRCorPt);
        
    int numPts = static_cast<int>(EyeCont.size());

    // OpenCV中，findContours()返回的轮廓点序列：外轮廓为逆时针方向，内轮廓为顺时针方向
    // 左右角点排在轮廓线中index次序与他们的y值大小有关，谁的小，谁排在前面。
    // !!! 无论哪种情形，i都是增大的。
    if(actLCorIdx > actRCorIdx)
    {
        // rightCorner --> leftCorner, upper curve
        for(int i=actRCorIdx; i<= actLCorIdx; i++)
        {
            lowEyeCurve.push_back(EyeCont[i]);
        }
        
        // leftCorner --> rightCorner, lower curve
        for(int i=actLCorIdx; i<=actRCorIdx+numPts; i++)
        {
            int j = i % numPts;
            upEyeCurve.push_back(EyeCont[j]);
        }
    }
    else // actLCorIdx < actRCorIdx
    {
        // rightCorner --> leftCorner, upper curve
        for(int i=actRCorIdx; i<= actLCorIdx+numPts; i++)
        {
            int j = i % numPts;
            lowEyeCurve.push_back(EyeCont[j]);
        }
        
        // leftCorner --> rightCorner, lower curve
        for(int i=actLCorIdx; i>=actRCorIdx; i++)
        {
            upEyeCurve.push_back(EyeCont[i]);
        }
    }
}

void DelExtremCurvPtsOnCurve(const CONTOUR& EyeCurve, CONTOUR& trimCurve)
{
    float meanCurv, stdevCurv;
    vector<float> curvList;
    EstMeanStdevCurvateOfCt(EyeCurve, meanCurv, stdevCurv, curvList);

    //auto minmax = std::minmax_element(curvList.begin(), curvList.end());
    float maxK = *max_element(curvList.begin(), curvList.end());
    float minK = *min_element(curvList.begin(), curvList.end());
    cout << "minK: " << minK << ", maxK: " << maxK << endl;
    cout << "meanCurv: " << meanCurv << ", stdevCurv: " << stdevCurv << endl;

    float curvUpTh = meanCurv + stdevCurv;
    float curvLowTh = 0.010; //meanCurv - stdevCurv;

    int N = static_cast<int>(EyeCurve.size());
    for(int i=1; i<=N-2; i++)
    {
        if(curvList[i-1] > curvLowTh && curvList[i-1] < curvUpTh )
        {
            trimCurve.push_back(EyeCurve[i]);
        }
    }
}

void SmCurveByFit(const CONTOUR& EyeCurve, CONTOUR& smCont)
{
    // frist step: calculate the mean of curvature, then
    // remove the points at both end of contour, which have high curvature.
    CONTOUR trimCurve;
    DelExtremCurvPtsOnCurve(EyeCurve, trimCurve);

    //CONTOUR smC
    SmoothCtByPIFitOrder2(trimCurve, smCont);
}

