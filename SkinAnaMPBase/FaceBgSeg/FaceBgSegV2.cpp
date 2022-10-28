//
//  FaceBgSegV2.cpp

/*******************************************************************************
Imported from FaceBgSeg Project.

Author: Fu Xiaoqiang
Date:   2022/10/10

********************************************************************************/

#include <iostream>
#include <string>
#include <fstream>
#include <sstream>
#include <vector>
#include <sys/stat.h>

#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>

#include "nlohmann/json.hpp"
#include "FaceBgSegV2.hpp"
#include "../Utils.hpp"
#include "../Geometry.hpp"

using namespace std;
using namespace cv;

using json = nlohmann::json;

//-------------------------------------------------------------------------------------------

FaceBgSegmentor::FaceBgSegmentor()
{
    
}

FaceBgSegmentor::~FaceBgSegmentor()
{

}

/******************************************************************************************
 *******************************************************************************************/

void FaceBgSegmentor::SegInfer(const Mat& srcImage,
                               FaceSegRst& segResult)
{
    srcImgH = srcImage.rows;
    srcImgW = srcImage.cols;
    segResult.srcImgS = srcImage.size();
    
    Mat srcImg_RGB;
    cv::cvtColor(srcImage, srcImg_RGB, cv::COLOR_BGR2RGB);
    
    // 预处理和前向推理
    cv::Mat blob = dnn::blobFromImage(srcImg_RGB, 1.0 / 255.0,
                                      cv::Size(SEG_NET_INPUT_SIZE, SEG_NET_INPUT_SIZE),
                                      Scalar(0, 0, 0), false, false, CV_32F);
    segNet.setInput(blob);
    
    Mat score = segNet.forward();
    
    // 取出推理结果进行后处理
    // 输出也是方阵，512*512
    const int rows = score.size[2];
    const int cols = score.size[3];
    const int chans = score.size[1]; // the channels here are classes that a pixel belongs to
    //segLabels = Mat(rows, cols, CV_8UC1, Scalar(0));
    Mat maxVal(rows, cols, CV_32FC1, Scalar(-1.0));
    
    // 推理之后，会计算出每个像素属于每个类别的隶属度
    // 检索出某个像素的最大类别隶属度，就判定这个像素属于这个类别
    // 在这种情形下，图像分割也就是像素分类。
    for (int c = 0; c < chans; c++)
    {
        for (int row = 0; row < rows; row++)
        {
            const float* ptrScore = score.ptr<float>(0, c, row);
            uchar* ptrSegClass = segResult.segLabels.ptr<uchar>(row);
            float* ptrMaxVal = maxVal.ptr<float>(row);
            for (int col = 0; col < cols; col++)
            {
                if (ptrScore[col] > ptrMaxVal[col])
                {
                    ptrMaxVal[col] = ptrScore[col];
                    ptrSegClass[col] = (uchar)c;
                }
            }
        }
    }
}

//-------------------------------------------------------------------------------------------
void FaceBgSegmentor::ScaleUpBBox(const Rect& inBBox, Rect& outBBox)
{
    Point2i p1 = inBBox.tl();
    Point2i p2 = inBBox.br();
    
    int outX1 = p1.x * srcImgW / SEG_NET_OUTPUT_SIZE;
    int outY1 = p1.y * srcImgH / SEG_NET_OUTPUT_SIZE;
        
    int outX2 = p2.x * srcImgW / SEG_NET_OUTPUT_SIZE;
    int outY2 = p2.y * srcImgH / SEG_NET_OUTPUT_SIZE;

    if((outX2 - outX1) % 2 != 0) //make width a even number
        outX2 += 1;
    if((outY2 - outY1) % 2 != 0) //make height a even number
        outY2 += 1;
    
    outBBox = Rect(Point(outX1, outY1), Point(outX2, outY2));
}

void FaceBgSegmentor::ScaleUpPoint(const Point2i& inPt, Point2i& outPt)
{
    int outX1 = inPt.x * srcImgW / SEG_NET_OUTPUT_SIZE;
    int outY1 = inPt.y * srcImgH / SEG_NET_OUTPUT_SIZE;
        
    outPt = Point2i(outX1, outY1);
}

void FaceBgSegmentor::ScaleUpPointSet(const Point2i inPts[], int numPt, Point2i outPts[])
{
    for(int i=0; i<numPt; i++)
        ScaleUpPoint(inPts[i], outPts[i]);
}

// NOS: Net Output Space
// the coordinate of Contour measured in NOS
// the return BBox is measured in SP, i.e., Source Space
Rect FaceBgSegmentor::CalcBBoxSPforContInNOS(const CONTOUR& ctInNOS)
{
    CONTOUR approxPoly;
    approxPolyDP(ctInNOS, approxPoly, 3, true );
    Rect nosBBox = boundingRect(approxPoly);
    
    // in the end, the result will be transformed into the space of source image.
    Rect spBBox;
    ScaleUpBBox(nosBBox, spBBox);
    
    return spBBox;
}

Rect FaceBgSegmentor::CalcBBoxNOSforContInNOS(const CONTOUR& ctInNOS)
{
    CONTOUR approxPoly;
    approxPolyDP(ctInNOS, approxPoly, 3, true );
    Rect nosBBox = boundingRect(approxPoly);
    return nosBBox;
}


// crop mask by using contour, i.e., change mask from the global coordinate into local coordinate
void FaceBgSegmentor::CropMaskByCont(const CONTOUR& contour, const Mat& maskGC,
                    SPACE_DEF space, SegMask& segMask)
{
    segMask.space = space;
    if(space == NET_OUT_SPACE)
    {
        segMask.bbox = CalcBBoxNOSforContInNOS(contour);
        maskGC(segMask.bbox).copyTo(segMask.mask);
    }
    else
    {
        segMask.bbox = CalcBBoxSPforContInNOS(contour);
        maskGC(segMask.bbox).copyTo(segMask.mask);
    }
}

//----------------------------------------------------------------------------------------------
// Bounding Box and Face Center Point in the coordinate system of the source image.
// Face Center Point must not be the center of BBox,
// It refers to the center of the line connecting the centers of wo eyes.
// FBEB_Mask: mask covering face and its sub-elements
void FaceBgSegmentor::CalcFaceBBox(const Mat& FBEB_Mask,
                                   FaceSegRst& segResult)
{
    // firstly, calculation is carried out in the space of seg labels, i.e., 512*512
    // in the end, the result will be transformed into the space of source image.
    
    CONTOURS contours;
    findContours(FBEB_Mask, contours,
            cv::noArray(), RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    if(contours.size() == 0)
    {
        cout << "No Face found in source image!" << endl;
        return;
    }
    
    std::sort(contours.begin(), contours.end(),
        [](const CONTOUR& a, const CONTOUR& b){return a.size() > b.size();});
    
    segResult.faceBBox = CalcBBoxSPforContInNOS(contours[0]);
}

//-------------------------------------------------------------------------------------------

// formula: ratio = abs(a1-a2) / max(a1, a2)
float FaceBgSegmentor::calcEyeAreaDiffRatio(int a1, int a2)
{
    int diff = abs(a1 - a2);
    int maxV = max(a1, a2);
    
    float r = (float)diff / (float)maxV;
    return r;
}

// this function includes: calcuate the center point of two eys, area of two eyes,
// and the ratio of area difference, final determine the face is in front view
// or profile view.
// The center point of face refers to the center of the line connecting the centers of wo eyes.
// in the profile view, the CP of face esitmated cannot be used for the bad precision.
void FaceBgSegmentor::CalcEyesInfo(const Mat& eyesMask, FaceSegRst& segResult)
{
    CONTOURS contours;
    //CHAIN_APPROX_SIMPLE会导致轮廓点稀疏，不紧密相挨，导致后面我们的算法失败
    findContours(eyesMask, contours,
            cv::noArray(), RETR_EXTERNAL, CHAIN_APPROX_NONE);

    if(contours.size() < 2)
    {
        cout << "some error happened when to extract eyes contours!" << endl;
        return;
    }
    
    std::sort(contours.begin(), contours.end(),
        [](const CONTOUR& a, const CONTOUR& b){return a.size() > b.size();});
    
    // Get the mass centers:
    vector<cv::Point2i> mc;
    for( auto& c : contours )
    {
        auto m = moments(c, true);
        //mu.push_back(m); // only if you need the mu vector
        int cx = (int)(m.m10/m.m00);
        int cy = (int)(m.m01/m.m00);
        mc.push_back(cv::Point2i(cx, cy));
    }
    
    Point2i eyeSegCP0, eyeSegCP1;
    ScaleUpPoint(mc[0], eyeSegCP0);
    ScaleUpPoint(mc[1], eyeSegCP1);
    
    double areaEye0 = contourArea(contours[0]);
    double areaEye1 = contourArea(contours[1]);

    float scaleX = (float)srcImgW / (float)SEG_NET_OUTPUT_SIZE;
    float scaleY = (float)srcImgH / (float)SEG_NET_OUTPUT_SIZE;
    
    int eyeArea0 = (int)(areaEye0 * scaleX * scaleY);
    int eyeArea1 = (int)(areaEye1 * scaleX * scaleY);
    
    if(eyeSegCP0.x < eyeSegCP1.x)  // 0 is left, 1 is right
    {
        segResult.lEyeSegCP = eyeSegCP0;
        segResult.rEyeSegCP = eyeSegCP1;
        
        segResult.leftEyeArea = eyeArea0;
        segResult.rightEyeArea = eyeArea1;
        
        CropMaskByCont(contours[0], eyesMask,
                       NET_OUT_SPACE, segResult.lEyeMaskNOS);
        CropMaskByCont(contours[1], eyesMask,
                       NET_OUT_SPACE, segResult.rEyeMaskNOS);
        
        // 0 is left, 1 is right
        CalcEyeCtFPs(srcImgW, srcImgH, contours[0], segResult.lBrowCP_NOS,
                     mc[0], segResult.lEyeFPsNOS, segResult.lEyeFPs); // left
        CalcEyeCtFPs(srcImgW, srcImgH, contours[1], segResult.rBrowCP_NOS,
                     mc[1], segResult.rEyeFPsNOS, segResult.rEyeFPs); // right
    }
    else // 1 is left, 0 is right
    {
        segResult.lEyeSegCP = eyeSegCP1;
        segResult.rEyeSegCP = eyeSegCP0;
        
        segResult.leftEyeArea = eyeArea1;
        segResult.rightEyeArea = eyeArea0;
        
        CropMaskByCont(contours[1], eyesMask,
                       NET_OUT_SPACE, segResult.lEyeMaskNOS);
        CropMaskByCont(contours[0], eyesMask,
                       NET_OUT_SPACE, segResult.rEyeMaskNOS);
        
        // 1 is left, 0 is right
        CalcEyeCtFPs(srcImgW, srcImgH, contours[1], segResult.lBrowCP_NOS,
                     mc[1], segResult.lEyeFPsNOS, segResult.lEyeFPs); // left
        CalcEyeCtFPs(srcImgW, srcImgH, contours[0], segResult.rBrowCP_NOS,
                     mc[0], segResult.rEyeFPsNOS, segResult.rEyeFPs); // right
    }
    
    segResult.eyeAreaDiffRatio = calcEyeAreaDiffRatio(eyeArea0, eyeArea1);
    
    if(segResult.eyeAreaDiffRatio > EyeAreaDiffRation_TH)
    {
        segResult.isFrontView = false;
        // facePriInfo.faceCP = (eyeCP1 + eyeCP2) / 2;
        int totalEyeArea = eyeArea0 + eyeArea1;
        float t = (float)eyeArea0 / (float)totalEyeArea;
        segResult.faceCP =  Interpolate(eyeSegCP0, eyeSegCP1, t);
    }
    else
    {
        segResult.isFrontView = true;
        segResult.faceCP = (eyeSegCP0 + eyeSegCP1) / 2;
    }
}

//-------------------------------------------------------------------------------------------
// extract all masks that covered by the facial elements,
// all of them are measured in Net Output Space
void FaceBgSegmentor::ParseSegLab(FaceSegRst& segResult,
                                  Mat& FBEB_Mask,
                                  Mat& browsMask,
                                  Mat& eyesMask,
                                  Mat& beardMask)
{
    // roadmap:
    // bg --> face --> brows --> eyes --> beard
        
    //in FBEB_Mask, 0 for background, 255 for face and its elments
    cv::threshold(segResult.segLabels, FBEB_Mask, SEG_BG_LABEL, 255, cv::THRESH_BINARY);
    //imwrite("FBEB_Mask.png", FBEB_Mask);
    
    Mat BEB_Mask; // 255 for brows, eyes, beard
    cv::threshold(segResult.segLabels, BEB_Mask, SEG_FACE_LABEL, 255, cv::THRESH_BINARY);
    
    Mat EB_Mask; // 255 for eyes and beard
    cv::threshold(segResult.segLabels, EB_Mask, SEG_EYEBROW_LABEL, 255, cv::THRESH_BINARY);
    
    cv::threshold(segResult.segLabels, beardMask, SEG_EYE_LABEL, 255, cv::THRESH_BINARY);
    
    eyesMask = EB_Mask & (~beardMask);
    //imwrite("eysMask.png", eyesMask);

    browsMask = BEB_Mask & (~EB_Mask);
    //imwrite("browsMask.png", browsMask);
}

void FaceBgSegmentor::CalcBrowsInfo(const Mat& browsMask,
                                    FaceSegRst& segResult)
{
    CONTOURS contours;
    //CHAIN_APPROX_SIMPLE会导致轮廓点稀疏，不紧密相挨
    findContours(browsMask, contours,
            cv::noArray(), RETR_EXTERNAL, CHAIN_APPROX_NONE);

    if(contours.size() < 2)
    {
        cout << "some error happened when to extract brows contours!" << endl;
        return;
    }
    
    std::sort(contours.begin(), contours.end(),
        [](const CONTOUR& a, const CONTOUR& b){return a.size() > b.size();});
    
    // Get the mass centers:
    vector<cv::Point2i> mc;
    for( auto& c : contours )
    {
        auto m = moments(c, true);
        //mu.push_back(m); // only if you need the mu vector
        int cx = (int)(m.m10/m.m00);
        int cy = (int)(m.m01/m.m00);
        mc.push_back(cv::Point2i(cx, cy));
    }
    
    Point2i browCP0, browCP1;
    ScaleUpPoint(mc[0], browCP0);
    ScaleUpPoint(mc[1], browCP1);

    if(browCP0.x < browCP1.x)
    {
        segResult.leftBrowCP = browCP0;
        segResult.rightBrowCP = browCP1;
        segResult.lBrowCP_NOS = mc[0];
        segResult.rBrowCP_NOS = mc[1];
        
        CropMaskByCont(contours[0], browsMask,
                       NET_OUT_SPACE, segResult.leftBrowMask);
        CropMaskByCont(contours[1], browsMask,
                       NET_OUT_SPACE, segResult.rightBrowMask);
    }
    else
    {
        segResult.leftBrowCP = browCP1;
        segResult.rightBrowCP = browCP0;
        segResult.lBrowCP_NOS = mc[1];
        segResult.rBrowCP_NOS = mc[0];
        
        CropMaskByCont(contours[1], browsMask,
                       NET_OUT_SPACE, segResult.leftBrowMask);
        CropMaskByCont(contours[0], browsMask,
                       NET_OUT_SPACE, segResult.rightBrowMask);
    }
}

//-------------------------------------------------------------------------------------------
void FaceBgSegmentor::SegImage(const Mat& srcImage, FaceSegRst& segResult)
{
    SegInfer(srcImage, segResult);
    
    Mat FBEB_MaskNOS, browsMaskNOS, eyesMaskNOS, beardMaskNOS; // measured in Net Output Space
    // extract all masks that covered by the facial elements
    ParseSegLab(segResult, FBEB_MaskNOS, browsMaskNOS, eyesMaskNOS, beardMaskNOS);
    
    CalcFaceBBox(FBEB_MaskNOS, segResult);

    // brows firstly, and then eyes
    CalcBrowsInfo(browsMaskNOS, segResult);
    
    CalcEyesInfo(eyesMaskNOS, segResult);

}
