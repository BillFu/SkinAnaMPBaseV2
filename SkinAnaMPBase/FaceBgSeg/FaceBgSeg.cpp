//
//  FaceBgSeg.cpp

/*******************************************************************************
Imported from FaceBgSeg Project.

Author: Fu Xiaoqiang
Date:   2022/9/29

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
#include "FaceBgSeg.hpp"
#include "../Utils.hpp"
#include "../Geometry.hpp"

using namespace std;
using namespace cv;

using json = nlohmann::json;

bool FaceBgSegmentor::isNetLoaded = false;
dnn::Net FaceBgSegmentor::segNet;
vector<Vec3b> FaceBgSegmentor::classColorTable;

// 这个函数在程序初始化时要调用一次，并确保返回true之后，才能往下进行
bool FaceBgSegmentor::LoadSegModel(const string& modelFileName)
{
    try
    {
        segNet = dnn::readNetFromONNX(modelFileName);
        cout << "the Model file for segmentation has been Successfully loaded!" << endl;
        isNetLoaded = true;
        return true;
    }
    catch (cv::Exception& e)
    {
        cout << "Failed to load the Face/Bg Segment model file: " << modelFileName << endl;
        isNetLoaded = false;
        return false;
    }
}

// return true if OK; otherwise return false
bool FaceBgSegmentor::LoadClassColorTable(const string& classColorFileName)
{
    std::vector<cv::Vec3b> colors;
    ifstream fp(classColorFileName.c_str());
    if (!fp.is_open())
    {
        cout << "can not open the class color table file: " << classColorFileName << endl;
        return false;
    }
    
    string line;
    while (!fp.eof())
    {
        getline(fp, line);
        if (line.length())
        {
            stringstream ss(line);
            string name;
            ss >> name;
            int r, g, b;
            ss >> r >> g >> b;
            Vec3b color(r, g, b);
            classColorTable.push_back(color);
        }
    }

    return true;
}

//-------------------------------------------------------------------------------------------

FaceBgSegmentor::FaceBgSegmentor()
{
    
}

FaceBgSegmentor::~FaceBgSegmentor()
{

}

/******************************************************************************************
 *******************************************************************************************/

void FaceBgSegmentor::Segment(const Mat& srcImage)
{
    srcImgH = srcImage.rows;
    srcImgW = srcImage.cols;
    
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
    segLabels = Mat(rows, cols, CV_8UC1);
    Mat maxVal(rows, cols, CV_32FC1);
    
    // 推理之后，会计算出每个像素属于每个类别的隶属度
    // 检索出某个像素的最大类别隶属度，就判定这个像素属于这个类别
    // 在这种情形下，图像分割也就是像素分类。
    for (int c = 0; c < chans; c++)
    {
        for (int row = 0; row < rows; row++)
        {
            const float* ptrScore = score.ptr<float>(0, c, row);
            uchar* ptrSegClass = segLabels.ptr<uchar>(row);
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


// 将分割结果以彩色Table渲染出来，并放大到原始图像尺度
Mat FaceBgSegmentor::RenderSegLabels()
{
    // mapping from label to corresponding color
    cv::Mat segLabelImg = Mat::zeros(SEG_NET_OUTPUT_SIZE, SEG_NET_OUTPUT_SIZE, CV_8UC3);
    for (int row = 0; row < SEG_NET_OUTPUT_SIZE; row++)
    {
        const uchar* ptrMaxCl = segLabels.ptr<uchar>(row);
        cv::Vec3b* ptrColor = segLabelImg.ptr<cv::Vec3b>(row);
        for (int col = 0; col < SEG_NET_OUTPUT_SIZE; col++)
        {
            ptrColor[col] = classColorTable[ptrMaxCl[col]];
        }
    }
    
    resize(segLabelImg, segLabelImg, Size(srcImgW, srcImgH));
    return segLabelImg;
}

//-------------------------------------------------------------------------------------------

// blend segment labels image with source iamge:
// result = alpha * segLabels + (1-alpha) * srcImage
// alpha lies in [0.0 1.0]
void OverlaySegOnImage(const Mat& segLabel, const Mat& srcImg,
                       float alpha,
                       const char* outImgFileName)
{
    Mat outImg;
    addWeighted(srcImg, 1.0 - alpha, segLabel, alpha, 0.0, outImg);
    imwrite(outImgFileName, outImg);
}

// blend segment labels image with source iamge:
// result = alpha * segLabels + (1-alpha) * srcImage
// alpha lies in [0.0 1.0]
void OverlaySegOnImageV2(const Mat& segLabel, const Mat& srcImg,
                       float alpha, const Rect& faceBBox,
                       const char* outImgFileName)
{
    Mat outImg;
    addWeighted(srcImg, 1.0 - alpha, segLabel, alpha, 0.0, outImg);
    
    cv::Scalar colorBox(255, 0, 0); // (B, G, R)
    rectangle(outImg, faceBBox, colorBox, 2, LINE_8);
    
    imwrite(outImgFileName, outImg);
}

void DrawSegOnImage(const Mat& segLabel, const Mat& srcImg,
                       float alpha, const FacePrimaryInfo& facePriInfo,
                       const char* outImgFileName)
{
    Mat outImg;
    addWeighted(srcImg, 1.0 - alpha, segLabel, alpha, 0.0, outImg);
    
    cv::Scalar colorBox(255, 0, 0); // (B, G, R)
    rectangle(outImg, facePriInfo.faceBBox, colorBox, 2, LINE_8);

    cv::Scalar yellow(0, 255, 255); // (B, G, R)
    for(auto cp: facePriInfo.eyeCPs)
    {
        //cv::Point center(x, y);
        cv::circle(outImg, cp, 10, yellow, cv::FILLED);
    }
    
    cv::circle(outImg, facePriInfo.faceCP, 12, yellow, cv::FILLED);


    Scalar redColor(0, 0, 255);  // BGR
    cv::putText(outImg, "eye diff ratio: " + to_string(facePriInfo.eyeAreaDiffRatio),
                Point(500, 250),
                FONT_HERSHEY_SIMPLEX, 4, redColor, 2);
    
    string viewStr = (facePriInfo.isFrontView) ? "Front View" : "Profile View";
    cv::putText(outImg, viewStr,
                Point(500, 350),
                FONT_HERSHEY_SIMPLEX, 4, redColor, 2);
    
    imwrite(outImgFileName, outImg);
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

    outBBox = Rect(Point(outX1, outY1), Point(outX2, outY2));
}

void FaceBgSegmentor::ScaleUpPoint(const Point2i& inPt, Point2i& outPt)
{
    int outX1 = inPt.x * srcImgW / SEG_NET_OUTPUT_SIZE;
    int outY1 = inPt.y * srcImgH / SEG_NET_OUTPUT_SIZE;
        
    outPt = Point2i(outX1, outY1);
}

// Bounding Box and Face Center Point in the coordinate system of the source image.
// Face Center Point must not be the center of BBox,
// It refers to the center of the line connecting the centers of wo eyes.
void FaceBgSegmentor::CalcFaceBBox(Rect& faceBBox)
{
    // firstly, calculation is carried out in the space of seg labels, i.e., 512*512
    // in the end, the result will be transformed into the space of source image.
    
    // 1. binary the seg labels image into two classes:
    // background and face(including its sub-component)
    
    cv::Mat labelsBi;
    cv::threshold(segLabels, labelsBi, SEG_BG_LABEL, 255, cv::THRESH_BINARY);
    //imwrite("labelsBi.png", labelsBi);
    
    CONTOURS contours;
    findContours(labelsBi, contours,
            cv::noArray(), RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    if(contours.size() == 0)
    {
        cout << "No Face found in source image!" << endl;
        return;
    }
    
    std::sort(contours.begin(), contours.end(),
        [](const CONTOUR& a, const CONTOUR& b){return a.size() > b.size();});
    
    CONTOUR faceContour = contours[0]; // get thr first one and the biggset one
    CONTOUR approxPoly;
    approxPolyDP(faceContour, approxPoly, 3, true );
    Rect smallBBox = boundingRect(approxPoly);
    
    // in the end, the result will be transformed into the space of source image.
    ScaleUpBBox(smallBBox, faceBBox);
}

void FaceBgSegmentor::CalcFaceBBox(FacePrimaryInfo& facePriInfo)
{
    CalcFaceBBox(facePriInfo.faceBBox);
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
void FaceBgSegmentor::CalcEyePts(FacePrimaryInfo& facePriInfo)
{
    // roadmap:
    // 1. calculate the beard mask
    // 2. calculate the beard_eyes mask
    // 3. eyes_mask = beard_eyes mask - beard mask
    
    Mat beardMask;
    // when one pixel meet: label > SEG_EYE_LABEL, then it is accepted as Beard
    threshold(segLabels, beardMask, SEG_EYE_LABEL, 255, cv::THRESH_BINARY);
    
    Mat eyesBeardMask;  // including both eyes and beard
    // when one pixel meet: label > SEG_EYEBROW_LABEL, then it is accepted as eyes or Beard
    threshold(segLabels, eyesBeardMask, SEG_EYEBROW_LABEL, 255, cv::THRESH_BINARY);
    
    Mat eyesMask = eyesBeardMask & (~beardMask);
    //imwrite("eysMask.png", eyesMask);
    
    CONTOURS contours;
    findContours(eyesMask, contours,
            cv::noArray(), RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

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
    
    Point2i eyeCP1, eyeCP2;
    ScaleUpPoint(mc[0], eyeCP1);
    ScaleUpPoint(mc[1], eyeCP2);
    
    facePriInfo.eyeCPs[0] = eyeCP1;
    facePriInfo.eyeCPs[1] = eyeCP2;
    
    double areaEye1 = contourArea(contours[0]);
    double areaEye2 = contourArea(contours[1]);

    float scaleX = (float)srcImgW / (float)SEG_NET_OUTPUT_SIZE;
    float scaleY = (float)srcImgH / (float)SEG_NET_OUTPUT_SIZE;
    
    facePriInfo.eyeAreas[0] = (int)(areaEye1 * scaleX * scaleY);
    facePriInfo.eyeAreas[1] = (int)(areaEye2 * scaleX * scaleY);
    
    facePriInfo.eyeAreaDiffRatio = calcEyeAreaDiffRatio(
            facePriInfo.eyeAreas[0], facePriInfo.eyeAreas[1]);
    
    if(facePriInfo.eyeAreaDiffRatio > EyeAreaDiffRation_TH)
    {
        facePriInfo.isFrontView = false;
        // facePriInfo.faceCP = (eyeCP1 + eyeCP2) / 2;
        int totalEyeArea = facePriInfo.eyeAreas[0] + facePriInfo.eyeAreas[1];
        float t = (float)facePriInfo.eyeAreas[0] / (float)totalEyeArea;
        facePriInfo.faceCP =  Interpolate(facePriInfo.eyeCPs[0], facePriInfo.eyeCPs[1], t);
    }
    else
    {
        facePriInfo.isFrontView = true;
        facePriInfo.faceCP = (eyeCP1 + eyeCP2) / 2;
    }
}

