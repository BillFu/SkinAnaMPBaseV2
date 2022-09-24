//
//  ForeheadMask.cpp

/*******************************************************************************
 
本模块最初的功能是，将额头的轮廓线适当地抬高一些。
更名后，增加了ForeheadMask的构建。

Author: Fu Xiaoqiang
Date:   2022/9/23

********************************************************************************/
#include <algorithm>
#include "ForeheadMask.hpp"
#include "Geometry.hpp"
#include "../BSpline/ParametricBSpline.hpp"
#include "FundamentalMask.hpp"


// 前额顶部轮廓线由9个lm点组成。这9个点组成第0排点集，比它们低一些的9个点组成第1排点集。
// 抬高后获得的9个点组成-1排点集。
vector<int> one_row_indices{68, 104, 69, 108, 151, 337, 299, 333, 298};
vector<int> zero_row_indices{54, 103, 67, 109, 10,  338, 297, 332, 284};  // 第0排才是MP提取出的前额顶部轮廓线

// 如果点在前额顶部轮廓线上，返回它在轮廓线点集中的index；否则返回-1
int getPtIndexOfFHCurve(int ptIndex)
{
    vector<int>& vec = zero_row_indices; // 第0排才是MP提取出的前额顶部轮廓线
    
    auto it = std::find(vec.begin(), vec.end(), ptIndex);
    if (it != vec.end())
        return (int)distance(vec.begin(), it);
    else
        return -1;
}

//-------------------------------------------------------------------------------------------

/******************************************************************************************

 *******************************************************************************************/

void RaiseupForeheadCurve(const int lm_2d[468][2], int raisedFhCurve[9][2], float alpha)
{
    // 前额顶部轮廓线由9个lm点组成。这9个点组成第0排点集，比它们低一些的9个点组成第1排点集。
    // 抬高后获得的9个点组成-1排点集。
    // 所有的lm indices以0为起始（id为0的点是“人中”穴位的最低点，V字沟的谷底）
    // 边缘的抬升效果要逐渐减弱，还要考虑x值向中央收拢一些。
    
    //int raisedPtsY[9]; // the Y values of the -1 row
    //int raisedPtsX[9]; // the X values of the -1 row
    for(int i = 0; i<9; i++)
    {
        int id_row0 = zero_row_indices[i];
        int id_row1 = one_row_indices[i];
        
        int yi_0 = lm_2d[id_row0][1];
        int xi_0 = lm_2d[id_row0][0];
        int yi_1 = lm_2d[id_row1][1];

        // the difference of y values between the 0 row and the 1 row.
        // delta_y = row1.y - row0.y; and delta_y > 0
        int delta_y = yi_1 - yi_0;
        
        // now not consider the attenuation effect when to be far away from the center
        raisedFhCurve[i][1] = int(yi_0 - delta_y* alpha);
        raisedFhCurve[i][0] = xi_0; // now x just keep unchanged
    }
}

void RaiseupForeheadCurve(const int lm_2d[468][2], Point2i raisedFhCurve[9], float alpha)
{
    // 前额顶部轮廓线由9个lm点组成。这9个点组成第0排点集，比它们低一些的9个点组成第1排点集。
    // 抬高后获得的9个点组成-1排点集。
    // 所有的lm indices以0为起始（id为0的点是“人中”穴位的最低点，V字沟的谷底）
    // 边缘的抬升效果要逐渐减弱，还要考虑x值向中央收拢一些。
    
    //int raisedPtsY[9]; // the Y values of the -1 row
    //int raisedPtsX[9]; // the X values of the -1 row
    for(int i = 0; i<9; i++)
    {
        int id_row0 = zero_row_indices[i];
        int id_row1 = one_row_indices[i];
        
        int yi_0 = lm_2d[id_row0][1];
        int xi_0 = lm_2d[id_row0][0];
        int yi_1 = lm_2d[id_row1][1];

        // the difference of y values between the 0 row and the 1 row.
        // delta_y = row1.y - row0.y; and delta_y > 0
        int delta_y = yi_1 - yi_0;
        
        // now not consider the attenuation effect when to be far away from the center
        raisedFhCurve[i].y = int(yi_0 - delta_y* alpha);
        raisedFhCurve[i].x = xi_0; // now x just keep unchanged
    }
    
    //把10点再升高一点点， 把109和338稍微升一点
    raisedFhCurve[4].y -= raisedFhCurve[4].y* 0.08;
    raisedFhCurve[3].y -= raisedFhCurve[3].y* 0.04;
    raisedFhCurve[5].y -= raisedFhCurve[5].y* 0.04;
}
//-------------------------------------------------------------------------------------------

/**********************************************************************************************

***********************************************************************************************/
void ForgeForeheadPg(const FaceInfo& faceInfo, POLYGON& outPolygon)
{
    // 点的索引针对468个general landmark而言
    /*
    int fhPtIndices[] = { // 顺时针计数
        67*, 67r, 109r, 10r, 338r, 297r, 297*,  // up line
        333, 334, 296, 336, 9, 107, 66, 105, 104  // bottom line
            ---------- 285, 8, 55 --------- expanded alternative sub-path
    };
    67*: 在67和103之间插值出来的。
    297*: 在297和332之间插值出来的。
    67r, 109r, 10r, 338r, 297r: 这5个点是原来点抬高后的版本。
    */
    
    //Point2i Interpolate(int x1, int y1, int x2, int y2, float t);
    //Point2i Interpolate(const Point2i& p1, const Point2i& p2, float t);
    // asterisk, 星号
    Point2i pt67a = IpGLmPtWithPair(faceInfo, 67, 103, 0.60);
    Point2i pt297a = IpGLmPtWithPair(faceInfo, 297, 332, 0.60);
    
    Point2i raisedFhPts[9];
    RaiseupForeheadCurve(faceInfo.lm_2d, raisedFhPts, 0.8);

    Point2i pt67r = raisedFhPts[2];
    Point2i pt109r = raisedFhPts[3];
    Point2i pt10r = raisedFhPts[4];
    Point2i pt338r = raisedFhPts[5];
    Point2i pt297r = raisedFhPts[6];
    
    outPolygon.push_back(pt67a);
    outPolygon.push_back(pt67r);
    outPolygon.push_back(pt109r);
    outPolygon.push_back(pt10r);
    outPolygon.push_back(pt338r);
    outPolygon.push_back(pt297r);
    outPolygon.push_back(pt297a);

    //int botLinePts[] = {333, 334*, 296*, 336, 285, 8, 55, 107, 66*, 105*, 104};
    outPolygon.push_back(getPtOnGLm(faceInfo, 333));

    Point2i pt334a = IpGLmPtWithPair(faceInfo, 334, 333, 0.65);
    Point2i pt296a = IpGLmPtWithPair(faceInfo, 296, 299, 0.50);
    outPolygon.push_back(pt334a);
    outPolygon.push_back(pt296a);

    outPolygon.push_back(getPtOnGLm(faceInfo, 336));
    outPolygon.push_back(getPtOnGLm(faceInfo, 285));
    outPolygon.push_back(getPtOnGLm(faceInfo, 8));
    outPolygon.push_back(getPtOnGLm(faceInfo, 55));
    outPolygon.push_back(getPtOnGLm(faceInfo, 107));

    Point2i pt66a = IpGLmPtWithPair(faceInfo, 66, 69, 0.5);
    Point2i pt105a = IpGLmPtWithPair(faceInfo, 105, 104, 0.65);
    outPolygon.push_back(pt66a);
    outPolygon.push_back(pt105a);
    
    outPolygon.push_back(getPtOnGLm(faceInfo, 104));
}

/**********************************************************************************************

***********************************************************************************************/
void ForgeForeheadMask(const FaceInfo& faceInfo, Mat& outMask)
{
    POLYGON coarsePolygon, refinedPolygon;

    ForgeForeheadPg(faceInfo, coarsePolygon);
    
    int csNumPoint = 80;
    CloseSmoothPolygon(coarsePolygon, csNumPoint, refinedPolygon);

    DrawContOnMask(faceInfo.imgWidth, faceInfo.imgHeight, refinedPolygon, outMask);
}
