//
//  SegHelper.cpp

/*******************************************************************************
把一些FaceBgSegV2.cpp中的内容切分出来，放在此处.

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
    
    // show the CPs of eyes
    cv::circle(outImg, segResult.leftEyeCP, 10, yellow, cv::FILLED);
    cv::circle(outImg, segResult.rightEyeCP, 10, yellow, cv::FILLED);
    
    // show the CPs of brows
    cv::circle(outImg, segResult.leftBrowCP, 10, yellow, cv::FILLED);
    cv::circle(outImg, segResult.rightBrowCP, 10, yellow, cv::FILLED);
    
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

