//
//  DetectRegion.cpp

/*******************************************************************************
Author: Fu Xiaoqiang
Date:   2022/9/15

********************************************************************************/

#include "DetectRegion.hpp"


//-------------------------------------------------------------------------------------------

/**********************************************************************************************

***********************************************************************************************/

Mat contour2mask(int img_width, int img_height, const POLYGON& contours)
{
    cv::Mat mask(img_height, img_width, CV_8UC1, cv::Scalar(0));
    cv::fillPoly(mask, contours, cv::Scalar(255));
    
    return mask;
}

//-------------------------------------------------------------------------------------------

/**********************************************************************************************

***********************************************************************************************/
void ForgeSkinPolygon(const FaceInfo& faceInfo, POLYGON& skinPolygon)
{
    // the outer contour of face in front view.
    int face_contour_pts_indices[] = {103, 67, 109, 10, 338, 297,
        332, 298, 300, 383, 372, 345, 352, 376, 433, 416, 364,
        430, 431, 369, 400, 396, 175, 171, 176, 140, 149,
        170, 150, 169, 136, 135, 138, 215, 177, 137, 227,
        234, 156, 46, 53, 52, 65, 107, 66, 105, 63, 70, 71, 68
    };
    
    int num_pts = sizeof(face_contour_pts_indices) / sizeof(int);
    
    for(int i = 0; i<num_pts; i++)
    {
        int index = face_contour_pts_indices[i];
        
        int x = faceInfo.lm_2d[index][0];
        int y = faceInfo.lm_2d[index][1];
        skinPolygon.push_back(Point2i(x, y));
    }
}

//-------------------------------------------------------------------------------------------
void ForgeSkinMask(int img_width, int img_height,
                   const FaceInfo& faceInfo, Mat& outMask)
{
    POLYGON skinPolygon;
    ForgeSkinPolygon(faceInfo, skinPolygon);
    outMask = contour2mask(img_width, img_height, skinPolygon);
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
