//
//  SkinMask.cpp

/*******************************************************************************
Author: Fu Xiaoqiang
Date:   2022/10/8

********************************************************************************/

#include "SkinMask.hpp"
#include "../BSpline/ParametricBSpline.hpp"
#include "ForeheadMask.hpp"
#include "../Geometry.hpp"
#include "FundamentalMask.hpp"

//-------------------------------------------------------------------------------------------

/**********************************************************************************************
额头顶部轮廓得到了提升。
***********************************************************************************************/
void ForgeSkinPgV3(const FaceInfo& faceInfo, POLYGON& skinPolygon,
                        Point2i raisedFhCurve[NUM_PT_TOP_FH], // for debugging, output
                        int raisedPtIndices[NUM_PT_TOP_FH]    //for debugging, output
                        )
{
#define NUM_PT_SILH  34
    // 36 points in silhouette
    int silhouette[] = {
        10,  338, 297, 332, 301, 368, 454, 323, 361, 288,
        397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
        172, 58,  132, 93,  234, 139, 71,  54,  103, 67,  109
    };
    
    int num_pts = sizeof(silhouette) / sizeof(int);
    
    //Point2i raisedFhCurve[NUM_PT_TOP_FH];
    RaiseupFhCurve(faceInfo.lm_2d, raisedFhCurve, raisedPtIndices, 0.8);

    // calculate the new version of coordinates of Points on face silhouette.
    Point2i newSilhouPts[NUM_PT_SILH];
    for(int i = 0; i<num_pts; i++)
    {
        int index = silhouette[i];
        int indexInFHC = getPtIndexOfFHCurve(index);
        
        if(indexInFHC == -1) // Not on Forehead Curve
        {
            newSilhouPts[i] = faceInfo.lm_2d[index];
        }
        else // on Forehead Curve, need update the coordinates, i.e. raising up
        {
            newSilhouPts[i] = raisedFhCurve[indexInFHC];
        }
    }
    
    for(int i = 0; i<num_pts-4; i++) // from 10, 338 ... to 139, 71
    {
        skinPolygon.push_back(newSilhouPts[i]);
    }
    
    Point2i pt103 = faceInfo.lm_2d[103]; // pass throght the original No.103 point.
    skinPolygon.push_back(pt103);
    
    Point2i pt103r = newSilhouPts[num_pts-3];
    Point2i pt67r  = newSilhouPts[num_pts-2];

    Point2i pt103rp = Interpolate(pt103r, pt67r, 0.4);
    skinPolygon.push_back(pt103rp);

    skinPolygon.push_back(newSilhouPts[num_pts-2]); // raised 67
    skinPolygon.push_back(newSilhouPts[num_pts-1]); // raised 109

}

//-------------------------------------------------------------------------------------------

/**********************************************************************************************
本函数构建skinMask，挖掉眉毛、嘴唇、眼睛等区域。
***********************************************************************************************/
void ForgeSkinMaskV3(const FaceInfo& faceInfo,
                     const Mat& mouthMask,
                     const Mat& eyebrowMask,
                     const Mat& eyeMask,
                     Mat& outMask,
                     Point2i raisedFhCurve[NUM_PT_TOP_FH],
                     int raisedPtIndices[NUM_PT_TOP_FH])
{
    POLYGON coarsePolygon, refinedPolygon;

    ForgeSkinPgV3(faceInfo, coarsePolygon,
                       raisedFhCurve, raisedPtIndices);
    
    int csNumPoint = 200;
    CloseSmoothPolygon(coarsePolygon, csNumPoint, refinedPolygon);

    DrawContOnMask(faceInfo.imgWidth, faceInfo.imgHeight, refinedPolygon, outMask);
    
    outMask = outMask & (~mouthMask) & (~eyebrowMask) & (~eyeMask);
}

//-------------------------------------------------------------------------------------------
