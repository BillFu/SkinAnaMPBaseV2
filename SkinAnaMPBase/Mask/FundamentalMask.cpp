//
//  DetectRegion.cpp

/*******************************************************************************
Author: Fu Xiaoqiang
Date:   2022/9/15

********************************************************************************/

#include "FundamentalMask.hpp"
#include "../BSpline/ParametricBSpline.hpp"
#include "ForeheadMask.hpp"
#include "../Geometry.hpp"

//-------------------------------------------------------------------------------------------
void expanMask(const Mat& inMask, int expandSize, Mat& outMask)
{
    Mat element = getStructuringElement(MORPH_ELLIPSE,
                           Size(2*expandSize + 1, 2*expandSize+1),
                           Point(expandSize, expandSize));
    
    dilate(inMask, outMask, element);
}

// !!!调用这个函数前，outMask必须进行过初始化，或者已有内容在里面！！！
void DrawContOnMask(const POLYGON& contours, Mat& outMask)
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

Mat ContourGroup2Mask(Size imgS, const POLYGON_GROUP& contoursGroup)
{
    Mat rst = ContourGroup2Mask(imgS.width, imgS.height, contoursGroup);
    return rst;
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
}

//-------------------------------------------------------------------------------------------

//expanRatio: expansion width toward outside / half of mouth height
void ForgeMouthMask(const FaceInfo& faceInfo, float expanRatio, Mat& outFinalMask)
{
    POLYGON coarsePolygon, refinedPolygon;

    int mouthW, mouthH;
    ForgeMouthPolygon(faceInfo, mouthW, mouthH, coarsePolygon);
    
    int csNumPoint = 60;
    DenseSmoothPolygon(coarsePolygon, csNumPoint, refinedPolygon);

    // when to construct a Mat, Height first, and then Width!
    Mat basicMask(faceInfo.srcImgS, CV_8UC1, cv::Scalar(0));
    DrawContOnMask(refinedPolygon, basicMask);
    
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
    DenseSmoothPolygon(coarsePolygon, csNumPoint, refinedPolygon);

    DrawContOnMask(refinedPolygon, outMask);
}

void ForgeEyesFullMask(const FaceInfo& faceInfo, Mat& outEyesFullMask)
{
    cv::Mat outMask(faceInfo.srcImgS, CV_8UC1, cv::Scalar(0));

    ForgeOneEyeFullMask(faceInfo, LEFT_EYE, outMask);
    ForgeOneEyeFullMask(faceInfo, RIGHT_EYE, outMask);
    
    outEyesFullMask = outMask;
    //expanMask(outMask, 20, outEyesFullMask);
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
    DenseSmoothPolygon(coarsePolygon, csNumPoint, refinedPolygon);

    DrawContOnMask(refinedPolygon, outMask);
}
//-------------------------------------------------------------------------------------------

void ForgeNoseBellPg(const FaceInfo& faceInfo, POLYGON& outPg)
{
    outPg.push_back(getPtOnGLm(faceInfo, 197));
    outPg.push_back(getPtOnGLm(faceInfo, 196));
    //Point2i pt236a = IpGLmPtWithPair(faceInfo, 236, 217, 0.15);
    //outPg.push_back(pt236a);
    
    outPg.push_back(getPtOnGLm(faceInfo, 174));
    outPg.push_back(getPtOnGLm(faceInfo, 198));
    outPg.push_back(getPtOnGLm(faceInfo, 48));
    outPg.push_back(getPtOnGLm(faceInfo, 92));
    outPg.push_back(getPtOnGLm(faceInfo, 169));
    
    // add two special points and 152, the lowest point
    Point2i pt169 = getPtOnGLm(faceInfo, 169);
    Point2i pt152 = getPtOnGLm(faceInfo, 152);
    Point2i pt394 = getPtOnGLm(faceInfo, 394);
    
    Point2i sp1 =  getRectCornerPt(pt169, pt152);
    outPg.push_back(sp1);
    outPg.push_back(pt152);
    Point2i sp2 =  getRectCornerPt(pt394, pt152);
    outPg.push_back(sp2);
    
    outPg.push_back(getPtOnGLm(faceInfo, 394));
    outPg.push_back(getPtOnGLm(faceInfo, 322));
    outPg.push_back(getPtOnGLm(faceInfo, 278));
    //Point2i pt456a = IpGLmPtWithPair(faceInfo, 456, 437, 0.15);
    //outPg.push_back(pt456a);
    
    outPg.push_back(getPtOnGLm(faceInfo, 420));
    outPg.push_back(getPtOnGLm(faceInfo, 399));
    outPg.push_back(getPtOnGLm(faceInfo, 419));
}

void ForgeNoseBellMask(const FaceInfo& faceInfo, Mat& outMask)
{
    POLYGON coarsePg, refinedPg;

    ForgeNoseBellPg(faceInfo, coarsePg);
    
    int csNumPoint = 80;
    DenseSmoothPolygon(coarsePg, csNumPoint, refinedPg);

    DrawContOnMask(refinedPg, outMask);
}

//-------------------------------------------------------------------------------------------

void OverlayMaskOnImage(const Mat& srcImg, const Mat& mask,
                        const string& maskName,
                        const char* out_filename,
                        Scalar drawColor)
{
    vector<Mat> blue_mask_chs;

    Mat zero_chan(srcImg.size(), CV_8UC1, Scalar(0));
    blue_mask_chs.push_back(zero_chan);
    blue_mask_chs.push_back(zero_chan);
    blue_mask_chs.push_back(mask);

    Mat blueMask;
    merge(blue_mask_chs, blueMask);
    
    Mat outImg = Mat::zeros(srcImg.size(), CV_8UC3);
    addWeighted(srcImg, 0.70, blueMask, 0.3, 0.0, outImg);
    
    double stdScale = 2.0;
    int    stdWidth = 2000;
    double fontScale = srcImg.cols * stdScale / stdWidth;
    
    //Scalar redColor(0, 0, 255);  // BGR
    cv::putText(outImg, "SkinAnaMPBase: " + maskName, Point(100, 100),
                    FONT_HERSHEY_SIMPLEX, fontScale, drawColor, 2);

    imwrite(out_filename, outImg);
}

void OverMaskOnCanvas(Mat& canvas, const Mat& mask,
                      const Scalar& drawColor)
{
    vector<Mat> mask_chs;

    //Mat zero_chan(srcImg.size(), CV_8UC1, Scalar(0));
    Mat blueCh = mask*drawColor[0]/255;
    Mat greenCh = mask*drawColor[1]/255;
    Mat redCh = mask*drawColor[2]/255;

    mask_chs.push_back(blueCh);
    mask_chs.push_back(greenCh);
    mask_chs.push_back(redCh);

    Mat coloredMask;
    merge(mask_chs, coloredMask);
    
    addWeighted(canvas, 0.70, coloredMask, 0.3, 0.0, canvas);
}
