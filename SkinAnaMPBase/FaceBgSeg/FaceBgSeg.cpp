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

bool FaceBgSegmentor::Initialize(const string& segModelFile, const string& classColorFile)
{
    // return true if OK; otherwise return false
    bool isOK = LoadClassColorTable(classColorFile);
    if(!isOK)
    {
        cout << "Failed to load Class Color Table: " << classColorFile << endl;
        return false;
    }
    
    // 这个函数在程序初始化时要调用一次，并确保返回true之后，才能往下进行
    isOK = LoadSegModel(segModelFile);
    if(!isOK)
    {
        cout << "Failed to load Image Segment Model: " << segModelFile << endl;
        return false;
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

void FaceBgSegmentor::Segment(const Mat& srcImage,
                              FaceSegResult& segResult)
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


// 将分割结果以彩色Table渲染出来，并放大到原始图像尺度
Mat FaceBgSegmentor::RenderSegLabels(const Size& imgSize, const Mat& segLabels)
{
    // mapping from label to corresponding color
    // 切记：初始化一个Mat时，尽可能提供最多已知的信息！
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
    
    resize(segLabelImg, segLabelImg, imgSize, INTER_NEAREST);
    return segLabelImg;
}

//-------------------------------------------------------------------------------------------

void DrawSegOnImage(const Mat& srcImg, float alpha,
                    const FaceSegResult& segResult,
                    const char* outImgFileName)
{
    Mat segColorLabel = FaceBgSegmentor::RenderSegLabels(
            srcImg.size(), segResult.segLabels);

    //Mat outImg(srcImg.size(), CV_32FC3);
    Mat outImg = Mat::zeros(srcImg.size(), CV_8UC3);
    addWeighted(srcImg, 1.0 - alpha, segColorLabel, alpha, 0.0, outImg);

    string outImg_DataType = openCVType2str(outImg.type());
    cout << "outImg_DataType: " << outImg_DataType << endl;

    cv::Scalar colorBox(255, 0, 0); // (B, G, R)
    rectangle(outImg, segResult.faceBBox, colorBox, 2, LINE_8);

    cv::Scalar yellow(0, 255, 255); // (B, G, R)
    for(auto cp: segResult.eyeCPs)
    {
        //cv::Point center(x, y);
        cv::circle(outImg, cp, 10, yellow, cv::FILLED);
    }
    
    cv::circle(outImg, segResult.faceCP, 12, yellow, cv::FILLED);

    double stdScale = 2.0;
    int    stdWidth = 2000;
    double fontScale = srcImg.cols * stdScale / stdWidth;
    
    Scalar redColor(0, 0, 255);  // BGR
    cv::putText(outImg, "eye diff ratio: " + to_string(segResult.eyeAreaDiffRatio),
                Point(100, 250),
                FONT_HERSHEY_SIMPLEX, fontScale, redColor, 2);
    
    string viewStr = (segResult.isFrontView) ? "Front View" : "Profile View";
    cv::putText(outImg, viewStr,
                Point(100, 350),
                FONT_HERSHEY_SIMPLEX, fontScale, redColor, 2);
    
    Point tlPt = segResult.faceBBox.tl();
    stringstream tlPtSS;
    tlPtSS << "BBox.tl: (x=" << tlPt.x << ",y=" << tlPt.y << ")";
    cv::putText(outImg, tlPtSS.str(), Point(100, 450),
                FONT_HERSHEY_SIMPLEX, fontScale, redColor, 2);
    
    int BBoxW = segResult.faceBBox.width;
    int BBoxH = segResult.faceBBox.height;
    stringstream bboxWHSS;
    bboxWHSS << "BBox.WH: (w=" << BBoxW << ",h=" << BBoxH << ")";
    cv::putText(outImg, bboxWHSS.str(), Point(100, 550),
                FONT_HERSHEY_SIMPLEX, fontScale, redColor, 2);
    
    stringstream cpSS;
    cpSS << "faceCP: (x=" << segResult.faceCP.x << ",y=" << segResult.faceCP.y << ")";
    cv::putText(outImg, cpSS.str(), Point(100, 650),
                FONT_HERSHEY_SIMPLEX, fontScale, redColor, 2);
    
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

// Bounding Box and Face Center Point in the coordinate system of the source image.
// Face Center Point must not be the center of BBox,
// It refers to the center of the line connecting the centers of wo eyes.
void FaceBgSegmentor::CalcFaceBBoxImpl(const Mat& segLabels, Rect& faceBBox)
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

void FaceBgSegmentor::CalcFaceBBox(FaceSegResult& segResult)
{
    CalcFaceBBoxImpl(segResult.segLabels, segResult.faceBBox);
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
void FaceBgSegmentor::CalcEyePts(FaceSegResult& segResult)
{
    // roadmap:
    // 1. calculate the beard mask
    // 2. calculate the beard_eyes mask
    // 3. eyes_mask = beard_eyes mask - beard mask
    
    Mat beardMask;
    // when one pixel meet: label > SEG_EYE_LABEL, then it is accepted as Beard
    threshold(segResult.segLabels, beardMask, SEG_EYE_LABEL, 255, cv::THRESH_BINARY);
    
    Mat eyesBeardMask;  // including both eyes and beard
    // when one pixel meet: label > SEG_EYEBROW_LABEL, then it is accepted as eyes or Beard
    threshold(segResult.segLabels, eyesBeardMask, SEG_EYEBROW_LABEL, 255, cv::THRESH_BINARY);
    
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
    
    segResult.eyeCPs[0] = eyeCP1;
    segResult.eyeCPs[1] = eyeCP2;
    
    double areaEye1 = contourArea(contours[0]);
    double areaEye2 = contourArea(contours[1]);

    float scaleX = (float)srcImgW / (float)SEG_NET_OUTPUT_SIZE;
    float scaleY = (float)srcImgH / (float)SEG_NET_OUTPUT_SIZE;
    
    segResult.eyeAreas[0] = (int)(areaEye1 * scaleX * scaleY);
    segResult.eyeAreas[1] = (int)(areaEye2 * scaleX * scaleY);
    
    segResult.eyeAreaDiffRatio = calcEyeAreaDiffRatio(
            segResult.eyeAreas[0], segResult.eyeAreas[1]);
    
    if(segResult.eyeAreaDiffRatio > EyeAreaDiffRation_TH)
    {
        segResult.isFrontView = false;
        // facePriInfo.faceCP = (eyeCP1 + eyeCP2) / 2;
        int totalEyeArea = segResult.eyeAreas[0] + segResult.eyeAreas[1];
        float t = (float)segResult.eyeAreas[0] / (float)totalEyeArea;
        segResult.faceCP =  Interpolate(segResult.eyeCPs[0], segResult.eyeCPs[1], t);
    }
    else
    {
        segResult.isFrontView = true;
        segResult.faceCP = (eyeCP1 + eyeCP2) / 2;
    }
}

//-------------------------------------------------------------------------------------------

// return a binary labels image: 0 for background, and 255 for face
// (including all its components), with the same size as the source image
Mat FaceBgSegmentor::CalcFaceBgBiLabel(const FaceSegResult& segResult) 
{
    // NOTE: be careful with there are two coordinate system!
    // 1. binary the seg labels image into two classes:
    // background and face(including its sub-component)
    Mat labelsBi;
    cv::threshold(segResult.segLabels, labelsBi, SEG_BG_LABEL, 255, cv::THRESH_BINARY);
    resize(labelsBi, labelsBi, segResult.srcImgS, INTER_NEAREST);
    
    return labelsBi;
}


// FB: face and background
Mat FaceBgSegmentor::CalcFBBiLabExBeard(const FaceSegResult& segResult)
{
    // NOTE: be careful with there are two coordinate system!
    // 1. binary the seg labels image into two classes:
    // background and face(including its sub-component)
    Mat labelsBi;
    cv::threshold(segResult.segLabels, labelsBi, SEG_BG_LABEL, 255, cv::THRESH_BINARY);
    
    Mat beardBilab;
    cv::threshold(segResult.segLabels, beardBilab, SEG_EYE_LABEL, 255, cv::THRESH_BINARY);
    labelsBi = labelsBi & (~beardBilab);
    
    resize(labelsBi, labelsBi, segResult.srcImgS, INTER_NEAREST);
    
    return labelsBi;
}

//-------------------------------------------------------------------------------------------

void SegImage(const Mat& srcImage, FaceSegResult& segResult)
{
    FaceBgSegmentor segmentor;
    segmentor.Segment(srcImage, segResult);
    
    segmentor.CalcFaceBBox(segResult);
    segmentor.CalcEyePts(segResult);
}
