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


typedef unique_ptr<tflite::Interpreter> INTERPRETER;
typedef unique_ptr<FlatBufferModel> TF_LITE_MODEL;

typedef vector<Point2i> POLYGON;
typedef vector<POLYGON> POLYGON_GROUP;

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
};

#define NUM_PT_GENERAL_LM       468
#define NUM_PT_EYE_REFINE_GROUP  71
#define NUM_PT_LIP_REFINE_GROUP  80

struct FaceInfo
{
    int imgWidth;  // source image
    int imgHeight;
    
    //[n][0] for x, [n][1] for y
    // measured in source iamge coordinate system
    float lm_3d[NUM_PT_GENERAL_LM][3];  // x, y, z, Not used
    int   lm_2d[NUM_PT_GENERAL_LM][2];  // x, y，与上面的lm_3d中相同，只是数据类型不同
    
    int leftEyeRefinePts[NUM_PT_EYE_REFINE_GROUP][2];
    int rightEyeRefinePts[NUM_PT_EYE_REFINE_GROUP][2];
    
    int lipRefinePts[NUM_PT_LIP_REFINE_GROUP][2];
    
    HeadPose headPose;
};

#endif /* end of COMMON_HPP */
