//
//  Common.hpp
//
//
/*
本模块提供一些基础性的公共定义，供各模块使用。
 
Author: Fu Xiaoqiang
Date:   2022/9/11
*/

#ifndef COMMON_HPP
#define COMMON_HPP

#include<opencv2/opencv.hpp>

#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/string_util.h"
#include "tensorflow/lite/examples/label_image/get_top_n.h"
#include "tensorflow/lite/model.h"

using namespace std;
using namespace cv;
using namespace tflite;

//#define TEST_RUN  // !!! 发布编译时，必须把这个定义给注释掉 ！！！

#define TEST_RUN2  // !!! 发布编译时，必须把这个定义给注释掉 ！！！

#ifdef TEST_RUN2
 extern string wrkOutDir;
extern string wrkMaskOutDir;
#endif

typedef unique_ptr<tflite::Interpreter> INTERPRETER;
typedef unique_ptr<FlatBufferModel> TF_LITE_MODEL;

typedef vector<Point2i> SPLINE;
typedef vector<Point2i> POLYGON;
typedef vector<POLYGON> POLYGON_GROUP;

typedef vector<Point> CONTOUR;
typedef vector<CONTOUR> CONTOURS;

#define SEG_NET_INPUT_SIZE   512
#define SEG_NET_OUTPUT_SIZE  512

enum EyeID
{
    LEFT_EYE,
    RIGHT_EYE
};

struct HeadPose
{
    float pitch;  // rotate with x-axis
    float yaw;    // rotate with y-axis
    float roll;   // rotate with z-axis
    
    HeadPose()
    {
        pitch = 0.0;
        yaw = 0.0;
        roll = 0.0;
    }
};

enum SPACE_DEF
{
    SOURCE_SPACE,
    NET_OUT_SPACE  // 512*512
};

enum CLOCK_DIR
{
    CLOCK_WISE,
    CCLOCK_WISE
};

struct PtInPolarCd  //Point in polar coordinate system
{
    double r;
    double theta;
    
    PtInPolarCd():
    r(0.0), theta(0.0)
    {
        
    }
    
    PtInPolarCd(double r0, double theta0):
    r(r0), theta(theta0)
    {
        
    }
};

typedef vector<PtInPolarCd>  PtInPolarSeq; //Seq: sequence

struct PolarContour
{
    Point2i oriPt;
    PtInPolarSeq ptSeq;
};

struct SegMask
{
    SPACE_DEF space;
    Rect bbox;
    Mat  mask; // sub-matrix cropped from the NOS or SP, its location specified by bbox
};

// bbox和mask具体针对的是哪个坐标系，由使用者灵活决定和负责。
struct DetectRegion  
{
    Rect bbox;
    Mat  mask;
    
    DetectRegion()
    {
        bbox = Rect(0, 0, 1, 1);
        mask = Mat(1, 1, CV_8UC1, Scalar(0));
    }
    
    DetectRegion(const Rect& bbox0, const Mat& mask0):
    bbox(bbox0), mask(mask0)
    {
        
    }
};

// the group of detcting regions for wrinkle
struct WrkRegGroup
{
    DetectRegion fhReg; // forehead
    DetectRegion glabReg; // glabella

    DetectRegion lEyeBagReg;
    DetectRegion rEyeBagReg;

    DetectRegion lNagvReg; //
    DetectRegion rNagvReg;

    DetectRegion lCheekReg;
    DetectRegion rCheekReg;

    DetectRegion lCrowFeetReg;
    DetectRegion rCrowFeetReg;
};

// 各类皮肤特征的检测区域汇总
struct DetRegPack
{
    Mat poreMask;
    Mat wrkFrgiMask; // used for frangi filtering to detect wrinkle
    WrkRegGroup  wrkRegGroup;
};

struct SegEyeFPsNOS
{
    Point lCorPtNOS;
    Point rCorPtNOS;
    Point mTopPtNOS;
    Point mBotPtNOS;
};

struct EyeSegFPs  // in source space
{
    Point lCorPt;
    Point rCorPt;
    Point mTopPt;
    Point mBotPt;
};

// store the information about face which has been refined out from the segment labels
// Rst: result
struct FaceSegRst
{
    // all the coordinates and sizes are measured in the space of the source image.
    Size        srcImgS;
    Rect        faceBBox;
    Point2i     faceCP;         // CP: Center Point;
    float       eyeAreaDiffRatio;  // ratio = abs(a1-a2) / max(a1, a2)
    
    //关于lEyeSegCP和rEyeSegCP的说明：
    // Seg表示来自人脸/背景分割的结果；没有加NOS的修饰，表示坐标是Source Space中的。
    // 该规则也适用于其他字段。
    Point2i     lEyeSegCP;    // in source space
    Point2i     rEyeSegCP;
    
    //Eye FP: feature points of eye, P1, P2, P3, P4。P1: 内侧上角点；P2: 外侧上角点；P3: 下弧线中点；P4: 上弧线中点。
    SegEyeFPsNOS  lEyeFPsNOS;
    SegEyeFPsNOS  rEyeFPsNOS;
    EyeSegFPs        lEyeFPs;
    EyeSegFPs        rEyeFPs;

    int         leftEyeArea;  // in source space
    int         rightEyeArea;

    SegMask     lEyeMaskNOS;
    SegMask     rEyeMaskNOS;
    
    Point2i     leftBrowCP;     // center point of left eyebrow in source space
    Point2i     rightBrowCP;    // center point of right eyebrow in source space
    Point2i     lBrowCP_NOS;  // in Net Output Space
    Point2i     rBrowCP_NOS;

    SegMask     leftBrowMask;
    SegMask     rightBrowMask;
    bool        isFrontView;
    Mat         segLabels;      // 512 * 512, one channel
    
    FaceSegRst()
    {
        segLabels = Mat(512, 512, CV_8UC1, Scalar(0));
    }
    
    friend ostream &operator<<(ostream &output, const FaceSegRst &fpi )
    {
        output << "FacePrimaryInfo{" << endl;
        
        output << "faceBBox: " << fpi.faceBBox << endl;
        output << "faceCP: " << fpi.faceCP << endl;
        //output << "eyeCPs1: " << fpi.eyeCPs[0] << endl;
        //output << "eyeCPs2: " << fpi.eyeCPs[1] << endl;
        output << "isFrontView: " << fpi.isFrontView << endl;

        output << "}" << endl;
        return output;
    }
};

#define NUM_PT_GENERAL_LM       468
#define NUM_PT_EYE_REFINE_GROUP  71
#define NUM_PT_LIP_REFINE_GROUP  80

struct FaceInfo
{
    //int imgWidth;  // source image
    //int imgHeight;
    Size  srcImgS;
    
    float confidence; // 这张人脸存在的可信度
    
    //[n][0] for x, [n][1] for y
    // measured in source iamge coordinate system
    float lm_3d[NUM_PT_GENERAL_LM][3];  // x, y, z, Not used
    Point2i   lm_2d[NUM_PT_GENERAL_LM];  // x, y，与上面的lm_3d中相同，只是数据类型不同
    
    Point2i lEyeRefinePts[NUM_PT_EYE_REFINE_GROUP];
    Point2i rEyeRefinePts[NUM_PT_EYE_REFINE_GROUP];
    
    Point2i lipRefinePts[NUM_PT_LIP_REFINE_GROUP];
    
    HeadPose headPose;
    
    FaceInfo()
    {
        srcImgS = Size(0, 0);
        confidence = 0.0;
    }
};

#endif /* end of COMMON_HPP */
