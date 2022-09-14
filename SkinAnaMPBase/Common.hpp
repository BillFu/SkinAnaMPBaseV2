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



#endif /* end of COMMON_HPP */
