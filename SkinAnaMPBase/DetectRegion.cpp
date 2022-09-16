//
//  DetectRegion.cpp

/*******************************************************************************
Author: Fu Xiaoqiang
Date:   2022/9/15

********************************************************************************/

#include "DetectRegion.hpp"
#include "BSpline/ParametricBSpline.hpp"

//-------------------------------------------------------------------------------------------

/**********************************************************************************************

***********************************************************************************************/

Mat contour2mask(int img_width, int img_height, const POLYGON& contours)
{
    cv::Mat mask(img_height, img_width, CV_8UC1, cv::Scalar(0));
    cv::fillPoly(mask, contours, cv::Scalar(255));
    
    return mask;
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

***********************************************************************************************/
void ForgeSkinPolygon(const FaceInfo& faceInfo, POLYGON& skinPolygon)
{
    // the outer contour of face in front view.
    /*
    int face_contour_pts_indices[] = {103, 67, 109, 10, 338, 297,
        332, 298, 300, 383, 372, 345, 352, 376, 433, 416, 364,
        430, 431, 369, 400, 396, 175, 171, 176, 140, 149,
        170, 150, 169, 136, 135, 138, 215, 177, 137, 227,
        234, 156, 46, 53, 52, 65, 107, 66, 105, 63, 70, 71, 68
    };
    */
    
    // the indices for lm in meadiapipe mesh from
    // https://github.com/tensorflow/tfjs-models/blob/838611c02f51159afdd77469ce67f0e26b7bbb23/face-landmarks-detection/src/mediapipe-facemesh/keypoints.ts
    
    int silhouette[] = {
        10,  338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
        397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
        172, 58,  132, 93,  234, 127, 162, 21,  54,  103, 67,  109};
    
    int num_pts = sizeof(silhouette) / sizeof(int);
    
    for(int i = 0; i<num_pts; i++)
    {
        int index = silhouette[i];
        
        int x = faceInfo.lm_2d[index][0];
        int y = faceInfo.lm_2d[index][1];
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
        
        if(eyeID == LeftEyeID)
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
void ForgeSkinMask(int img_width, int img_height,
                   const FaceInfo& faceInfo, Mat& outMask)
{
    POLYGON skinPolygon;
    ForgeSkinPolygon(faceInfo, skinPolygon);
    outMask = contour2mask(img_width, img_height, skinPolygon);
}


void ForgeMouthMask(int img_width, int img_height,
                   const FaceInfo& faceInfo, Mat& outMask)
{
    POLYGON polygon;
    ForgeMouthPolygon(faceInfo, polygon);
    outMask = contour2mask(img_width, img_height, polygon);
}


void ForgeTwoEyebowsMask(int img_width, int img_height,
                    const FaceInfo& faceInfo, Mat& outMask)
{
    POLYGON leftEyeBowPolygon, rightEyeBowPolygon;
    ForgeOneEyebowPolygon(faceInfo, LeftEyeID, leftEyeBowPolygon);
    ForgeOneEyebowPolygon(faceInfo, RightEyeID, rightEyeBowPolygon);
    
    POLYGON_GROUP polygonGroup;
    polygonGroup.push_back(leftEyeBowPolygon);
    polygonGroup.push_back(rightEyeBowPolygon);
    
    outMask = ContourGroup2Mask(img_width, img_height, polygonGroup);
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
