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
 前额顶部轮廓线由9个lm点组成。
 Input: lm_2d
 Output: raisedForeheadCurve
 alpha: [0.0 1.0]，the greater this value is, the more raised up
 *******************************************************************************************/
void RaiseupForeheadCurve(const Point2i lm_2d[468], int raisedFhCurve[NUM_PT_TOP_FH][2], float alpha)
{
    // 前额顶部轮廓线由9个lm点组成。这9个点组成第0排点集，比它们低一些的9个点组成第1排点集。
    // 抬高后获得的9个点组成-1排点集。
    // 所有的lm indices以0为起始（id为0的点是“人中”穴位的最低点，V字沟的谷底）
    // 边缘的抬升效果要逐渐减弱，还要考虑x值向中央收拢一些。
    
    for(int i = 0; i<NUM_PT_TOP_FH; i++)
    {
        int id_row0 = zero_row_indices[i];
        int id_row1 = one_row_indices[i];
        
        int yi_0 = lm_2d[id_row0].y;
        int xi_0 = lm_2d[id_row0].x;
        int yi_1 = lm_2d[id_row1].y;

        // the difference of y values between the 0 row and the 1 row.
        // delta_y = row1.y - row0.y; and delta_y > 0
        int delta_y = yi_1 - yi_0;
        
        // now not consider the attenuation effect when to be far away from the center
        raisedFhCurve[i][1] = int(yi_0 - delta_y* alpha);
        raisedFhCurve[i][0] = xi_0; // now x just keep unchanged
    }
}

/******************************************************************************************
 前额顶部轮廓线由9个lm点组成。
 Input: lm_2d
 Output: raisedForeheadCurve
 alpha: [0.0 1.0]，the greater this value is, the more raised up
 *******************************************************************************************/
void RaiseupFhCurve(const Point2i lm_2d[468],
                          Point2i raisedFhCurve[NUM_PT_TOP_FH],
                          int raisedPtIndices[NUM_PT_TOP_FH],
                          float alpha)
{
    // 前额顶部轮廓线由9个lm点组成。这9个点组成第0排点集，比它们低一些的9个点组成第1排点集。
    // 抬高后获得的9个点组成-1排点集。
    // 所有的lm indices以0为起始（id为0的点是“人中”穴位的最低点，V字沟的谷底）
    // 边缘的抬升效果要逐渐减弱，还要考虑x值向中央收拢一些。
    
    for(int i = 0; i<NUM_PT_TOP_FH; i++)
    {
        int id_row0 = zero_row_indices[i];
        int id_row1 = one_row_indices[i];
        raisedPtIndices[i] = id_row0;
        
        int yi_0 = lm_2d[id_row0].y;
        int xi_0 = lm_2d[id_row0].x;
        int yi_1 = lm_2d[id_row1].y;

        // the difference of y values between the 0 row and the 1 row.
        // delta_y = row1.y - row0.y; and delta_y > 0
        int delta_y = yi_1 - yi_0;
        
        // now not consider the attenuation effect when to be far away from the center
        raisedFhCurve[i].y = int(yi_0 - delta_y* alpha);
        raisedFhCurve[i].x = xi_0; // now x just keep unchanged
    }
    
    //把10点再升高一点点， 把109和338稍微升一点
    raisedFhCurve[4].y -= raisedFhCurve[4].y* 0.06;
    raisedFhCurve[3].y -= raisedFhCurve[3].y* 0.04; // 109
    raisedFhCurve[5].y -= raisedFhCurve[5].y* 0.04; // 338
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
    
    Point2i raisedFhPts[NUM_PT_TOP_FH];
    int raisedPtIndices[NUM_PT_TOP_FH];
    RaiseupFhCurve(faceInfo.lm_2d, raisedFhPts, raisedPtIndices, 0.7);

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

    Point2i pt334a = IpGLmPtWithPair(faceInfo, 334, 333, 0.4);
    //Point2i pt296a = IpGLmPtWithPair(faceInfo, 296, 299, 0.15);
    outPolygon.push_back(pt334a);
    //outPolygon.push_back(pt296a);
    Point2i pt296 = getPtOnGLm(faceInfo, 296);
    outPolygon.push_back(pt296);

    Point2i pt336a = IpGLmPtWithPair(faceInfo, 336, 151, -0.1);
    outPolygon.push_back(pt336a);
    
    Point2i pt285a = IpGLmPtWithPair(faceInfo, 285, 55, 0.05);
    outPolygon.push_back(pt285a);
    
    //outPolygon.push_back(getPtOnGLm(faceInfo, 8));
    
    Point2i pt8a = IpGLmPtWithPair(faceInfo, 8, 168, 0.4);
    outPolygon.push_back(pt8a);
    
    Point2i pt55a = IpGLmPtWithPair(faceInfo, 55, 285, 0.05);
    outPolygon.push_back(pt55a);

    Point2i pt107a = IpGLmPtWithPair(faceInfo, 107, 151, -0.1);
    outPolygon.push_back(pt107a);

    Point2i pt105a = IpGLmPtWithPair(faceInfo, 105, 104, 0.4);
    Point2i pt66 = getPtOnGLm(faceInfo, 66);
    outPolygon.push_back(pt66);
    outPolygon.push_back(pt105a);
    
    outPolygon.push_back(getPtOnGLm(faceInfo, 104));
}

/**********************************************************************************************

***********************************************************************************************/
void ForgeForeheadMask(const FaceInfo& faceInfo, const Mat& fbBiLab, Mat& outMask)
{
    POLYGON coarsePolygon, refinedPolygon;

    ForgeForeheadPg(faceInfo, coarsePolygon);
    
    int csNumPoint = 80;
    CloseSmoothPolygon(coarsePolygon, csNumPoint, refinedPolygon);

    DrawContOnMask(faceInfo.imgWidth, faceInfo.imgHeight, refinedPolygon, outMask);
    
    outMask = outMask & fbBiLab;
}


//-------------------------------------------------------------------------------------------

/**********************************************************************************************
扩展版的前额区域，向下扩展到部分鼻梁区域。
***********************************************************************************************/
void ForgeExpFhPg(const FaceInfo& faceInfo, POLYGON& outPolygon)
{
    // 点的索引针对468个general landmark而言
    // 顺时针计数
    
    // asterisk, 星号
    Point2i pt67a = IpGLmPtWithPair(faceInfo, 67, 103, 0.60);
    Point2i pt297a = IpGLmPtWithPair(faceInfo, 297, 332, 0.60);
    
    Point2i raisedFhPts[NUM_PT_TOP_FH];
    int raisedPtIndices[NUM_PT_TOP_FH];
    RaiseupFhCurve(faceInfo.lm_2d, raisedFhPts, raisedPtIndices, 0.7);

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

    Point2i pt334a = IpGLmPtWithPair(faceInfo, 334, 333, 0.4);
    outPolygon.push_back(pt334a);
    Point2i pt296 = getPtOnGLm(faceInfo, 296);
    outPolygon.push_back(pt296);

    Point2i pt336a = IpGLmPtWithPair(faceInfo, 336, 151, -0.1);
    outPolygon.push_back(pt336a);
    
    outPolygon.push_back(getPtOnGLm(faceInfo, 285));
    outPolygon.push_back(getPtOnGLm(faceInfo, 417));
    outPolygon.push_back(getPtOnGLm(faceInfo, 465));
    outPolygon.push_back(getPtOnGLm(faceInfo, 357));
    outPolygon.push_back(getPtOnGLm(faceInfo, 277));
    outPolygon.push_back(getPtOnGLm(faceInfo, 437));
    outPolygon.push_back(getPtOnGLm(faceInfo, 195));
    outPolygon.push_back(getPtOnGLm(faceInfo, 217));
    outPolygon.push_back(getPtOnGLm(faceInfo, 47));
    outPolygon.push_back(getPtOnGLm(faceInfo, 128));
    outPolygon.push_back(getPtOnGLm(faceInfo, 245));
    outPolygon.push_back(getPtOnGLm(faceInfo, 193));
    outPolygon.push_back(getPtOnGLm(faceInfo, 55));

    Point2i pt107a = IpGLmPtWithPair(faceInfo, 107, 151, -0.1);
    outPolygon.push_back(pt107a);

    Point2i pt105a = IpGLmPtWithPair(faceInfo, 105, 104, 0.4);
    Point2i pt66 = getPtOnGLm(faceInfo, 66);
    outPolygon.push_back(pt66);
    outPolygon.push_back(pt105a);
    
    outPolygon.push_back(getPtOnGLm(faceInfo, 104));
}

void ForgeExpFhMask(const FaceInfo& faceInfo, const Mat& fbBiLab, Mat& outMask)
{
    POLYGON coarsePg, refinedPg;
    ForgeExpFhPg(faceInfo, coarsePg);
    
    int csNumPoint = 100;
    CloseSmoothPolygon(coarsePg, csNumPoint, refinedPg);

    DrawContOnMask(faceInfo.imgWidth, faceInfo.imgHeight, refinedPg, outMask);
    
    outMask = outMask & fbBiLab;
}
