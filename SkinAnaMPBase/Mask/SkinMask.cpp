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
#include "Utils.hpp"

//-------------------------------------------------------------------------------------------
#define NUM_PT_SILH  34

/**********************************************************************************************
额头顶部轮廓得到了提升。
***********************************************************************************************/
void ForgeSkinPgV3(const FaceInfo& faceInfo, POLYGON& skinPolygon,
                        Point2i raisedFhCurve[NUM_PT_TOP_FH], // for debugging, output
                        int raisedPtIndices[NUM_PT_TOP_FH]    //for debugging, output
                        )
{
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
    
    // 修正下巴底缘的三点：377, 152, 148, 使其坐标位于脸部之内
    
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
                     const Mat& lowFaceMask,
                     Mat& outMask,
                     Point2i raisedFhCurve[NUM_PT_TOP_FH],
                     int raisedPtIndices[NUM_PT_TOP_FH])
{
    POLYGON coarsePolygon, refinedPolygon;

    ForgeSkinPgV3(faceInfo, coarsePolygon,
                       raisedFhCurve, raisedPtIndices);
    
    int csNumPoint = 200;
    CloseSmoothPolygon(coarsePolygon, csNumPoint, refinedPolygon);

    DrawContOnMask(refinedPolygon, outMask);
    
    // calculate the lower cut line:
    Point2i pt61 = faceInfo.lm_2d[61];
    Point2i pt291 = faceInfo.lm_2d[291];
    int y = (pt61.y + pt291.y) / 2;
    
    int w = faceInfo.srcImgS.width;
    int h = faceInfo.srcImgS.height - y;
    Rect lowCutRect(0, y, w, h);
    lowFaceMask(lowCutRect).copyTo(outMask(lowCutRect));
    
    outMask = outMask & (~mouthMask) & (~eyebrowMask) & (~eyeMask);
}

//-------------------------------------------------------------------------------------------

bool isPtInMask(const Point2i& pt,
                const Mat& mask)
{
    if(mask.at<uchar>(pt) == 255)
        return true;
    else
        return false;
}

void raisePtONJaw(Point2i& oriPt,
                  const Mat& faceMask,
                  int step)
{
    while(!isPtInMask(oriPt, faceMask))
    {
        oriPt.y -= step;
    }
}

void ForgeSkinPgV4(const FaceInfo& faceInfo,
                   const Mat& lowFaceMask,
                   POLYGON& skinPolygon,
                   Point2i raisedFhCurve[NUM_PT_TOP_FH], // for debugging, output
                   int raisedPtIndices[NUM_PT_TOP_FH]    //for debugging, output
                        )
{
    // 36 points in silhouette
    int silhouette[] = {
        10,  338, 297, 332, 301, 368, 454, 323, 361, 288,
        397, 365, 379, 378, 369, 377, 152, 148, 176, 149,
        150, 136, 172, 58,  132, 93,  234, 139, 71,  54,
        103, 67,  109
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
    
    int upStep = faceInfo.srcImgS.width / 1000;
    // 修正下巴底缘的三点：377, 152, 148, 使其坐标位于脸部之内
    raisePtONJaw(newSilhouPts[15], // pt377
                 lowFaceMask, upStep);
    raisePtONJaw(newSilhouPts[16], // pt152
                 lowFaceMask, upStep);
    raisePtONJaw(newSilhouPts[17], // pt148
                 lowFaceMask, upStep);
    
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

/**********************************************************************************************
本函数构建skinMask，挖掉眉毛、嘴唇、眼睛等区域。
***********************************************************************************************/
void ForgeSkinMaskV4(const FaceInfo& faceInfo,
                     const Mat& mouthMask,
                     const Mat& eyebrowMask,
                     const Mat& eyeMask,
                     const Mat& lowFaceMask,
                     Mat& outMask,
                     Point2i raisedFhCurve[NUM_PT_TOP_FH],
                     int raisedPtIndices[NUM_PT_TOP_FH])
{
    POLYGON coarsePolygon, refinedPolygon;

    ForgeSkinPgV4(faceInfo, lowFaceMask, coarsePolygon,
                       raisedFhCurve, raisedPtIndices);
    
    int csNumPoint = 200;
    CloseSmoothPolygon(coarsePolygon, csNumPoint, refinedPolygon);

    DrawContOnMask(refinedPolygon, outMask);
    
    /*
    // calculate the lower cut line:
    Point2i pt61 = faceInfo.lm_2d[61];
    Point2i pt291 = faceInfo.lm_2d[291];
    int y = (pt61.y + pt291.y) / 2;
    
    int w = faceInfo.imgWidth;
    int h = faceInfo.imgHeight - y;
    Rect lowCutRect(0, y, w, h);
    lowFaceMask(lowCutRect).copyTo(outMask(lowCutRect));
    */
    
    outMask = outMask & (~mouthMask) & (~eyebrowMask) & (~eyeMask);
}

//-------------------------------------------------------------------------------------------

// 新思路：先有一组关键点连接起来，构成脸部外轮廓的雏形，而后，若它们不在分割出的脸部Mask之内，则向内收缩，
// 直到进入分割出的脸部Mask之内。收缩的方向垂直于该点的法线，而法线方向是依据当前点的左右邻居连线而估计出来的。
// 目前我们的分割算法，在眼睛以下的脸部较为准确，在额头部位受到头发的干扰而降低了精准度。
// 因此，眼睛以下采用分割结果引导关键点收缩的思路，眼睛以上则要另谋出路。

// move Point on silhouette into the area of face given by segmentation
// eta: control the step size of each moving
void MovePt2SegFaceMask(const Point2i& curOriPt, const Point2i& prevOriPt, const Point2i& nextOriPt,
                        const Mat& faceMask, float eta, Point2i& movedCurPt)
{
    // 逐点进行的顺序是：preOriPt --> curOriPt --> nextOriPt
    // 推导出的计算公式严重依赖于这种顺序安排。
    
    movedCurPt = curOriPt;
    
    int dx = nextOriPt.x - prevOriPt.x;
    int dy = nextOriPt.y - prevOriPt.y;
    
    // 前进方向为（-dy, dx）
    int move_times = 0;
    while(!isPtInMask(movedCurPt, faceMask)
          && isInImg(movedCurPt, faceMask.cols, faceMask.rows)
          && move_times <= 5)
    {
        int xa = movedCurPt.x - eta*dy;  //apostrophe，撇号
        int ya = movedCurPt.y + eta*dx;
        movedCurPt = Point2i(xa, ya);
        
        move_times++;
    }
}

//Pg: polygon
void ForgeSkinPgV5(const FaceInfo& faceInfo,
                   const Mat& lowFaceMask,
                   POLYGON& skinPolygon,
                   Point2i raisedFhCurve[NUM_PT_TOP_FH], // for debugging, output
                   int raisedPtIndices[NUM_PT_TOP_FH]    //for debugging, output
)
{
#define NUM_PT_SILH_36  36
    
    // 36 points in silhouette
    int silhouette[] = {
        /*
        10,  338, 297, 332, 301, 368, 454, 323, 361, 288,
        397, 365, 379, 378, 369, 377, 152, 148, 176, 149,
        150, 136, 172, 58,  132, 93,  234, 139, 71,  54,
        103, 67,  109
        */
        //这些点以顺时针的方向排列。编号为10的点，位于额顶中央。
        
        10,  338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
        397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
        172, 58,  132, 93,  234, 127, 162, 21,  54,  103, 67,  109
    };
    
    int num_pts = sizeof(silhouette) / sizeof(int);
    assert(num_pts == NUM_PT_SILH_36);
    
    //Point2i raisedFhCurve[NUM_PT_TOP_FH];
    RaiseupFhCurve(faceInfo.lm_2d, raisedFhCurve, raisedPtIndices, 0.8);

    // calculate the basic version of coordinates of Points on face silhouette.
    Point2i silhouPts[NUM_PT_SILH_36];
    for(int i = 0; i<num_pts; i++)
    {
        int index = silhouette[i];
        int indexInFHC = getPtIndexOfFHCurve(index);
        
        if(indexInFHC == -1) // Not on Forehead Curve
        {
            silhouPts[i] = faceInfo.lm_2d[index];
        }
        else // on Forehead Curve, to raise up the original point
        {
            silhouPts[i] = raisedFhCurve[indexInFHC];
        }
    }
    
    // move the points on the silhouette torward inside to make sure that
    // every point in the area of face given by the segmentation processing.
    Point2i movedSilhouPts[NUM_PT_SILH_36];

    float eta = 1.0 / 80.0;
    for(int i = 0; i<num_pts; i++)
    {
        Point2i curOriPt, prevOriPt, nextOriPt;
        
        curOriPt = silhouPts[i];
        if(i == 0)
        {
            nextOriPt = silhouPts[i+1];
            prevOriPt = silhouPts[num_pts-1];
        }
        else if(i == num_pts - 1)
        {
            nextOriPt = silhouPts[0];
            prevOriPt = silhouPts[i-1];
        }
        else // normal case
        {
            nextOriPt = silhouPts[i+1];
            prevOriPt = silhouPts[i-1];
        }
        
        MovePt2SegFaceMask(curOriPt, prevOriPt, nextOriPt,
                           lowFaceMask, eta, movedSilhouPts[i]);
    }
    
    for(int i = 0; i<num_pts-1; i++)
    {
        skinPolygon.push_back(movedSilhouPts[i]);
    }
}

void ForgeSkinMaskV5(const FaceInfo& faceInfo,
                     const Mat& mouthMask,
                     const Mat& eyebrowMask,
                     const Mat& eyeMask,
                     const Mat& lowFaceMask,
                     Mat& outMask,
                     //---the following items are used for debugging, output
                     Point2i raisedFhCurve[NUM_PT_TOP_FH],
                     int raisedPtIndices[NUM_PT_TOP_FH])
{
    POLYGON coarsePolygon, refinedPolygon;

    ForgeSkinPgV5(faceInfo, lowFaceMask, coarsePolygon,
                       raisedFhCurve, raisedPtIndices);
    
    int csNumPoint = 200;
    CloseSmoothPolygon(coarsePolygon, csNumPoint, refinedPolygon);
    DrawContOnMask(refinedPolygon, outMask);
    //DrawContOnMask(faceInfo.imgWidth, faceInfo.imgHeight, coarsePolygon, outMask);
    
    outMask = outMask & (~mouthMask) & (~eyebrowMask) & (~eyeMask);
}



/*
void MovePt2SegFaceMaskTest(const Point2i& curOriPt,
                        const Point2i& prevOriPt, const Point2i& nextOriPt,
                        const Mat& faceMask, float eta, Point2i& movedCurPt,
                        const Mat& srcImage)
{
    // 逐点进行的顺序是：preOriPt --> curOriPt --> nextOriPt
    // 推导出的计算公式严重依赖于这种顺序安排。
    Mat canvas = srcImage.clone();
    
    movedCurPt = curOriPt;
    
    int dx = nextOriPt.x - prevOriPt.x;
    int dy = nextOriPt.y - prevOriPt.y;
    
    cv::Scalar yellow(0, 255, 255); // (B, G, R)
    cv::Scalar blue(255, 0, 0);
    
    // 前进方向为（-dy, dx）
    while(!isPtInMask(movedCurPt, faceMask)
          && isInImg(movedCurPt, faceMask.cols, faceMask.rows))
    {
        int xa = movedCurPt.x - eta*dy;  //apostrophe，撇号
        int ya = movedCurPt.y + eta*dx;
        movedCurPt = Point2i(xa, ya);
        
        //cv::// cv::Point center(faceInfo.lm_2d[i].x, faceInfo.lm_2d[i].y);
        cv::circle(canvas, movedCurPt, 5, blue, cv::FILLED);
    }
    
    imwrite("test.png", canvas);
}


//Pg: polygon
void TestPgV5(const FaceInfo& faceInfo,
              const Mat& lowFaceMask,
              POLYGON& skinPolygon,
              const Mat& srcImage)
{
//#define NUM_PT_SILH_36  36
    Point2i raisedFhCurve[NUM_PT_TOP_FH]; // for debugging, output
    int raisedPtIndices[NUM_PT_TOP_FH];
    
    // 36 points in silhouette
    int silhouette[] = {
        //这些点以顺时针的方向排列。编号为10的点，位于额顶中央。
        10,  338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
        397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
        172, 58,  132, 93,  234, 127, 162, 21,  54,  103, 67,  109
    };
    
    int num_pts = sizeof(silhouette) / sizeof(int);
    assert(num_pts == NUM_PT_SILH_36);
    
    //Point2i raisedFhCurve[NUM_PT_TOP_FH];
    RaiseupFhCurve(faceInfo.lm_2d, raisedFhCurve, raisedPtIndices, 0.8);

    // calculate the basic version of coordinates of Points on face silhouette.
    Point2i silhouPts[NUM_PT_SILH_36];
    for(int i = 0; i<num_pts; i++)
    {
        int index = silhouette[i];
        int indexInFHC = getPtIndexOfFHCurve(index);
        
        if(indexInFHC == -1) // Not on Forehead Curve
        {
            silhouPts[i] = faceInfo.lm_2d[index];
        }
        else // on Forehead Curve, to raise up the original point
        {
            silhouPts[i] = raisedFhCurve[indexInFHC];
        }
    }
    
    // move the points on the silhouette torward inside to make sure that
    // every point in the area of face given by the segmentation processing.
    Point2i movedSilhouPt;

    float eta = 1.0 / 80.0;
    
    Point2i curOriPt, prevOriPt, nextOriPt;
        
    curOriPt = silhouPts[4];
    nextOriPt = silhouPts[5];
    prevOriPt = silhouPts[3];
    MovePt2SegFaceMaskTest(curOriPt, prevOriPt, nextOriPt,
                        lowFaceMask, eta, movedSilhouPt, srcImage);

}

void TestMaskV5(const FaceInfo& faceInfo,
            const Mat& lowFaceMask,
            const Mat& srcImage)
{
    POLYGON coarsePolygon;

    TestPgV5(faceInfo, lowFaceMask, coarsePolygon, srcImage);
}
*/
