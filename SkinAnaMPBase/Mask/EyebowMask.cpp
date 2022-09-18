//
//  EyebowMask.cpp

/*******************************************************************************
Author: Fu Xiaoqiang
Date:   2022/9/11

********************************************************************************/

#include "EyebowMask.hpp"
#include "../BSpline/ParametricBSpline.hpp"
#include "DetectRegion.hpp"

//-------------------------------------------------------------------------------------------

void ForgeOneEyebowPolygon(const FaceInfo& faceInfo, EyeID eyeID, POLYGON& eyebowPolygon)
{
    // 采用Eye Refine Region的点，左眉毛、右眉毛的坐标索引是相同的！
    int eyeBowPtIndices[] = {69, 68, 66, 65, 64, 50, 43, 45};
    //int eyeBowPtIndices[] = {69, 68, 67, 66, 65, 64, 50, 43, 44, 45};

    POLYGON coarseEyebowPolygon;
    int num_pts = sizeof(eyeBowPtIndices) / sizeof(int);
    for(int i = 0; i<num_pts; i++)
    {
        int index = eyeBowPtIndices[i];
        
        int x, y;
        
        if(eyeID == LEFT_EYE)
        {
            x = faceInfo.leftEyeRefinePts[index][0];
            y = faceInfo.leftEyeRefinePts[index][1];
        }
        else  //RightEyeID
        {
            x = faceInfo.rightEyeRefinePts[index][0];
            y = faceInfo.rightEyeRefinePts[index][1];
        }
        
        coarseEyebowPolygon.push_back(Point2i(x, y));
    }
    
    // then convert the corse to the refined
    int csNumPoint = 100;
    ForgeClosedSmoothPolygon(coarseEyebowPolygon, csNumPoint, eyebowPolygon);
}

/******************************************************************************************
*******************************************************************************************/
void ForgeTwoEyebowsMask(const FaceInfo& faceInfo, Mat& outMask)
{
    POLYGON leftEyeBowPolygon, rightEyeBowPolygon;
    ForgeOneEyebowPolygon(faceInfo, LEFT_EYE, leftEyeBowPolygon);
    ForgeOneEyebowPolygon(faceInfo, RIGHT_EYE, rightEyeBowPolygon);
    
    POLYGON_GROUP polygonGroup;
    polygonGroup.push_back(leftEyeBowPolygon);
    polygonGroup.push_back(rightEyeBowPolygon);
    
    outMask = ContourGroup2Mask(faceInfo.imgWidth, faceInfo.imgHeight, polygonGroup);
}

//-------------------------------------------------------------------------------------------
