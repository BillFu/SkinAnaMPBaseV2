//
//  DetectRegion.cpp

/*******************************************************************************
Author: Fu Xiaoqiang
Date:   2022/9/15

********************************************************************************/

#include "FundamentalMask.hpp"
#include "../BSpline/ParametricBSpline.hpp"
#include "ForeheadMask.hpp"
#include "Geometry.hpp"

//-------------------------------------------------------------------------------------------
void expanMask(const Mat& inMask, int expandSize, Mat& outMask)
{
    Mat element = getStructuringElement(MORPH_ELLIPSE,
                           Size(2*expandSize + 1, 2*expandSize+1),
                           Point(expandSize, expandSize));
    
    dilate(inMask, outMask, element);
}

// !!!调用这个函数前，outMask必须进行过初始化，或者已有内容在里面！！！
void DrawContOnMask(int img_width, int img_height, const POLYGON& contours, Mat& outMask)
{
    cv::fillPoly(outMask, contours, cv::Scalar(255));
}

Mat ContourGroup2Mask(int img_width, int img_height, const POLYGON_GROUP& contoursGroup)
{
    cv::Mat mask(img_height, img_width, CV_8UC1, cv::Scalar(0));
    
    for(auto contours: contoursGroup)
    {
        cv::fillPoly(mask, contours, cv::Scalar(255));
    }
    
    return mask;
}

//-------------------------------------------------------------------------------------------

/**********************************************************************************************
额头顶部轮廓得到了提升。
***********************************************************************************************/
void ForgeSkinPolygon(const FaceInfo& faceInfo, POLYGON& skinPolygon)
{
    // the indices for lm in meadiapipe mesh from
    // https://github.com/tensorflow/tfjs-models/blob/838611c02f51159afdd77469ce67f0e26b7bbb23/face-landmarks-detection/src/mediapipe-facemesh/keypoints.ts
    
    int silhouette[] = {
        10,  338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
        397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
        172, 58,  132, 93,  234, 127, 162, 21,  54,  103, 67,  109};
    
    int num_pts = sizeof(silhouette) / sizeof(int);
    
    int raisedFhCurve[9][2];
    RaiseupForeheadCurve(faceInfo.lm_2d, raisedFhCurve, 0.8);

    // calculate the new version of coordinates of Points on face silhouette.
    int newSilhouPts[36][2];
    for(int i = 0; i<num_pts; i++)
    {
        int index = silhouette[i];
        int indexInFHC = getPtIndexOfFHCurve(index);
        
        if(indexInFHC == -1) // Not on Forehead Curve
        {
            newSilhouPts[i][0] = faceInfo.lm_2d[index].x;
            newSilhouPts[i][1] = faceInfo.lm_2d[index].y;
        }
        else // on Forehead Curve, need update the coordinates, i.e. raising up
        {
            newSilhouPts[i][0] = raisedFhCurve[indexInFHC][0];
            newSilhouPts[i][1] = raisedFhCurve[indexInFHC][1];
        }
    }
    
    for(int i = 0; i<num_pts; i++)
    {
        int x = newSilhouPts[i][0];
        int y = newSilhouPts[i][1];
        skinPolygon.push_back(Point2i(x, y));
    }
}

// face mask below the eyes, 在鼻子部位向上凸出，接近额头
void ForgeFaceLowThEyePg(const FaceInfo& faceInfo, POLYGON& skinPolygon)
{
    // the indices for lm in meadiapipe mesh from
    int outlinePts[] = { // in counterclockwise order
    //234, // 234, the most left and top point
    116, 137, 177, 147, 172, 136, 150, 149,
        176, 148, 199, 428,
    //400, 378, 379, 365, 397, 288, 376, 366, 323, //454,  // 454, the most right and top point
    400, 431, 430, 416, 288, 376, 366, 447,  
    // top limit line from right to left
    346, 349, 453, 417, 336, 107, 193, 233, 120, 117
    };
    
    int num_pts = sizeof(outlinePts) / sizeof(int);
    
    // calculate the new version of coordinates of Points on face silhouette.
    for(int i = 0; i<num_pts; i++)
    {
        int index = outlinePts[i];
        skinPolygon.push_back(faceInfo.lm_2d[index]);
    }
}
//-------------------------------------------------------------------------------------------

void ForgeMouthPolygon(const FaceInfo& faceInfo,
                       int& mouthWidth, int& mouthHeight,
                       POLYGON& mouthPolygon)
{
    // the indices for lm in meadiapipe mesh from
    //https://github.com/tensorflow/tfjs-models/blob/838611c02f51159afdd77469ce67f0e26b7bbb23/face-landmarks-detection/src/mediapipe-facemesh/keypoints.ts
    
    // 采用Lip Refine Region的点！
    int lipsOuterPtIndices[] = {
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,  // 下外轮廓线，从左到右
        19, 18, 17, 16, 15, 14, 13, 12, 11 // 上外轮廓线，从右到左
    };
    
    int num_pts = sizeof(lipsOuterPtIndices) / sizeof(int);
    
    for(int i = 0; i<num_pts; i++)
    {
        int index = lipsOuterPtIndices[i];
        mouthPolygon.push_back(faceInfo.lipRefinePts[index]);
    }
    
    // the following five points come from the general lms
    Point2i pt37 = getPtOnGLm(faceInfo, 37);
    Point2i pt267 = getPtOnGLm(faceInfo, 267) ;
    Point2i pt57 = getPtOnGLm(faceInfo, 57);
    Point2i pt287 = getPtOnGLm(faceInfo, 287);
    Point2i pt17 = getPtOnGLm(faceInfo, 17);

    mouthHeight = pt17.y - (pt37.y + pt267.y) / 2;
    mouthWidth  = pt287.x - pt57.x;
    
    cout << "mouthHeight: " << mouthHeight << endl;
    cout << "mouthWidth: " << mouthWidth << endl;
}

//-------------------------------------------------------------------------------------------
void ForgeSkinMask(const FaceInfo& faceInfo, Mat& outMask)
{
    POLYGON coarsePolygon, refinedPolygon;

    ForgeSkinPolygon(faceInfo, coarsePolygon);
    
    int csNumPoint = 200;
    CloseSmoothPolygon(coarsePolygon, csNumPoint, refinedPolygon);

    DrawContOnMask(faceInfo.imgWidth, faceInfo.imgHeight, refinedPolygon, outMask);
}

// face mask below the eyes, 在鼻子部位向上凸出，接近额头
void ForgeFaceLowThEyeMask(const FaceInfo& faceInfo, Mat& outMask)
{
    POLYGON coarsePolygon, refinedPolygon;

    ForgeFaceLowThEyePg(faceInfo, coarsePolygon);
    int csNumPoint = 160;
    CloseSmoothPolygon(coarsePolygon, csNumPoint, refinedPolygon);

    DrawContOnMask(faceInfo.imgWidth, faceInfo.imgHeight, refinedPolygon, outMask);
}

//expanRatio: expansion width toward outside / half of mouth height
void ForgeMouthMask(const FaceInfo& faceInfo, float expanRatio, Mat& outFinalMask)
{
    POLYGON coarsePolygon, refinedPolygon;

    int mouthW, mouthH;
    ForgeMouthPolygon(faceInfo, mouthW, mouthH, coarsePolygon);
    
    int csNumPoint = 60;
    CloseSmoothPolygon(coarsePolygon, csNumPoint, refinedPolygon);

    // when to construct a Mat, Height first, and then Width!
    Mat basicMask(faceInfo.imgHeight, faceInfo.imgWidth, CV_8UC1, cv::Scalar(0));
    DrawContOnMask(faceInfo.imgWidth, faceInfo.imgHeight, refinedPolygon, basicMask);
    
    int expandSize = expanRatio * mouthH / 2;
    expanMask(basicMask, expandSize, outFinalMask);
}

//-------------------------------------------------------------------------------------------
// Pg: Polygon
void ForgeOneEyeFullPg(const Point2i eyeRefinePts[71], POLYGON& outPolygon)
{
    // 采用Lip Refine Region的点！
    int fullEyeOuterPtIndices[] = { // 顺时针计数
        70, 68, 66, 65, 64, 54,  // 下外轮廓线，从左到右
        56, 57, 58, 59, 60, 61, 62   // 上外轮廓线，从右到左
    };
    
    int num_pts = sizeof(fullEyeOuterPtIndices) / sizeof(int);
    
    for(int i = 0; i<num_pts; i++)
    {
        int index = fullEyeOuterPtIndices[i];
        outPolygon.push_back(eyeRefinePts[index]);
    }
}

// EeyeFullMask包含眼睛、眉毛、眼袋的大范围区域
void ForgeOneEyeFullMask(const FaceInfo& faceInfo, EyeID eyeID, Mat& outMask)
{
    POLYGON coarsePolygon, refinedPolygon;
    
    if(eyeID == LEFT_EYE)
        ForgeOneEyeFullPg(faceInfo.lEyeRefinePts, coarsePolygon);
    else
        ForgeOneEyeFullPg(faceInfo.rEyeRefinePts, coarsePolygon);
    
    int csNumPoint = 50; //200;
    CloseSmoothPolygon(coarsePolygon, csNumPoint, refinedPolygon);

    DrawContOnMask(faceInfo.imgWidth, faceInfo.imgHeight, refinedPolygon, outMask);
}

void ForgeTwoEyesFullMask(const FaceInfo& faceInfo, Mat& outEyesFullMask)
{
    cv::Mat outMask(faceInfo.imgHeight, faceInfo.imgWidth, CV_8UC1, cv::Scalar(0));

    ForgeOneEyeFullMask(faceInfo, LEFT_EYE, outMask);
    ForgeOneEyeFullMask(faceInfo, RIGHT_EYE, outMask);
    
    /*
    int dila_size = 20;
    Mat element = getStructuringElement(MORPH_ELLIPSE,
                           Size(2*dila_size + 1, 2*dila_size+1),
                           Point(dila_size, dila_size));
    
    dilate(outMask, outEyesFullMask, element);
    */
    
    expanMask(outMask, 20, outEyesFullMask);
}

//-------------------------------------------------------------------------------------------
void ForgeNosePolygon(const FaceInfo& faceInfo, POLYGON& outPolygon)
{
    // 采用Lip Refine Region的点！
    int nosePtIndices[] = { // 逆时针
    55, 193, 245, 114, 142, 203, 98, 97, // No.55 point near left eye
        2,
    326, 327, 423, 371, 343, 465, 417, 285 };
    
    int num_pts = sizeof(nosePtIndices) / sizeof(int);
    
    for(int i = 0; i<num_pts; i++)
    {
        int index = nosePtIndices[i];
        outPolygon.push_back(faceInfo.lm_2d[index]);
    }
}

void ForgeNoseMask(const FaceInfo& faceInfo, Mat& outMask)
{
    POLYGON coarsePolygon, refinedPolygon;

    ForgeNosePolygon(faceInfo, coarsePolygon);
    
    int csNumPoint = 100;
    CloseSmoothPolygon(coarsePolygon, csNumPoint, refinedPolygon);

    DrawContOnMask(faceInfo.imgWidth, faceInfo.imgHeight, refinedPolygon, outMask);
}
//-------------------------------------------------------------------------------------------

void OverlayMaskOnImage(const Mat& srcImg, const Mat& mask,
                        const string& maskName,
                        const char* out_filename)
{
    vector<Mat> blue_mask_chs;

    Mat zero_chan(srcImg.size(), CV_8UC1, Scalar(0));
    blue_mask_chs.push_back(zero_chan);
    blue_mask_chs.push_back(zero_chan);
    blue_mask_chs.push_back(mask);

    Mat blueMask;
    merge(blue_mask_chs, blueMask);
    
    Mat outImg;
    addWeighted(srcImg, 0.70, blueMask, 0.3, 0.0, outImg);
    
    Scalar blueColor(255, 0, 0);  // BGR
    putText(outImg, "SkinAnaMPBase: " + maskName, Point(100, 100),
                    FONT_HERSHEY_SIMPLEX, 2, blueColor, 2);
    //bool isSucceeded =
    imwrite(out_filename, outImg);
}
