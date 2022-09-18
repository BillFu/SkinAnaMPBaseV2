//
//  DetectRegion.cpp

/*******************************************************************************
Author: Fu Xiaoqiang
Date:   2022/9/15

********************************************************************************/

#include "DetectRegion.hpp"
#include "BSpline/ParametricBSpline.hpp"
#include "ForeheadCurve.hpp"

//-------------------------------------------------------------------------------------------

/**********************************************************************************************

***********************************************************************************************/

Mat contour2mask(int img_width, int img_height, const POLYGON& contours)
{
    cv::Mat mask(img_height, img_width, CV_8UC1, cv::Scalar(0));
    cv::fillPoly(mask, contours, cv::Scalar(255));
    
    return mask;
}

void Contour2Mask(int img_width, int img_height, const POLYGON& contours, Mat& outMask)
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
            newSilhouPts[i][0] = faceInfo.lm_2d[index][0];
            newSilhouPts[i][1] = faceInfo.lm_2d[index][1];
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

void ForgeMouthPolygon(const FaceInfo& faceInfo, POLYGON& mouthPolygon)
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
        
        int x = faceInfo.lipRefinePts[index][0];
        int y = faceInfo.lipRefinePts[index][1];
        mouthPolygon.push_back(Point2i(x, y));
    }
}

void ForgeOneEyebowPolygon(const FaceInfo& faceInfo, EyeID eyeID, POLYGON& eyebowPolygon)
{
    // 采用Eye Refine Region的点，左眉毛、右眉毛的坐标索引是相同的！
    int eyeBowPtIndices[] = {69, 68, 67, 66, 65, 64, 50, 43, 44, 45};
    
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
    int csNumPoint = 200;
    ForgeClosedSmoothPolygon(coarseEyebowPolygon, csNumPoint, eyebowPolygon);
}

//-------------------------------------------------------------------------------------------
void ForgeSkinMask(const FaceInfo& faceInfo, Mat& outMask)
{
    POLYGON skinPolygon;
    ForgeSkinPolygon(faceInfo, skinPolygon);
    outMask = contour2mask(faceInfo.imgWidth, faceInfo.imgHeight, skinPolygon);
}


void ForgeMouthMask(const FaceInfo& faceInfo, Mat& outMask)
{
    POLYGON polygon;
    ForgeMouthPolygon(faceInfo, polygon);
    outMask = contour2mask(faceInfo.imgWidth, faceInfo.imgHeight, polygon);
}


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

void ForgeOneEyeFullPolygon(const int eyeRefinePts[71][2], POLYGON& outPolygon)
{
    // 采用Lip Refine Region的点！
    int fullEyeOuterPtIndices[] = {
        // 顺时针计数
        70, 68, 67, 66, 65, 64, 63, 54,  // 下外轮廓线，从左到右
        55, 56, 57, 58, 59, 60, 61, 62   // 上外轮廓线，从右到左
    };
    
    int num_pts = sizeof(fullEyeOuterPtIndices) / sizeof(int);
    
    for(int i = 0; i<num_pts; i++)
    {
        int index = fullEyeOuterPtIndices[i];
        
        int x = eyeRefinePts[index][0];
        int y = eyeRefinePts[index][1];
        outPolygon.push_back(Point2i(x, y));
    }
}

// EeyeFullMask包含眼睛、眉毛、眼袋的大范围区域
void ForgeOneEyeFullMask(const FaceInfo& faceInfo, EyeID eyeID, Mat& outMask)
{
    POLYGON coarsePolygon, refinedPolygon;
    
    if(eyeID == LEFT_EYE)
        ForgeOneEyeFullPolygon(faceInfo.leftEyeRefinePts, coarsePolygon);
    else
        ForgeOneEyeFullPolygon(faceInfo.rightEyeRefinePts, coarsePolygon);
    
    
    
    int csNumPoint = 50; //200;
    ForgeClosedSmoothPolygon(coarsePolygon, csNumPoint, refinedPolygon);

    Contour2Mask(faceInfo.imgWidth, faceInfo.imgHeight, refinedPolygon, outMask);
}

Mat ForgeTwoEyesFullMask(const FaceInfo& faceInfo)
{
    cv::Mat outMask(faceInfo.imgHeight, faceInfo.imgWidth, CV_8UC1, cv::Scalar(0));

    ForgeOneEyeFullMask(faceInfo, LEFT_EYE, outMask);
    ForgeOneEyeFullMask(faceInfo, RIGHT_EYE, outMask);
    
    /*
    int dila_size = 30;
    Mat element = getStructuringElement(MORPH_ELLIPSE,
                           Size(2*dila_size + 1, 2*dila_size+1),
                           Point(dila_size, dila_size));
    
    cv::Mat outExpandedMask(faceInfo.imgHeight, faceInfo.imgWidth, CV_8UC1, cv::Scalar(0));
    dilate(outMask, outExpandedMask, element);
    
    return outExpandedMask;
    */
    return outMask;
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
    bool isSucceeded = imwrite(out_filename, outImg);
}
