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


enum EyeID
{
    LeftEyeID,
    RightEyeID
};

struct FaceInfo
{
    //[n][0] for x, [n][1] for y
    // measured in source iamge coordinate system
    float lm_3d[468][3];  // x, y, z
    int   lm_2d[468][2];  // x, y，与上面的lm_3d中相同，只是数据类型不同
    
    int leftEyeRefinePts[71][2];
    int rightEyeRefinePts[71][2];
    
    int lipRefinePts[80][2];
    
    float pitch;  // rotate with x-axis
    float yaw;    // rotate with y-axis
    float roll;   // rotate with z-axis

};

#endif /* end of COMMON_HPP */
