//
//  FaceDetection.hpp
//
/*
该模块的原始版本来自
https://github.com/cuongvng/Face-Detection-TFLite-JNI-Android
 
本模块的功能是，利用tensorflow lite C++ API来驱动MediaPipe中内含的Face Detect深度学习网络，
来推理获取人脸图像中的Face Box（包括坐标和置信度）.
目前的推理是一次性的，即从模型加载到解释器创建，到推理，到结果提取，这一流程只负责完成一副人脸影像的处理。
 
Author: Fu Xiaoqiang
Date:   2022/9/12
*/

#ifndef FACE_DETECTION_HPP
#define FACE_DETECTION_HPP

// #include <cstdio>
#include <iostream>
#include <opencv2/opencv.hpp>
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/optional_debug_tools.h"

enum{N_FACE_ATTB=5}; // number of attributes of the following struct:

struct FaceBoxInfo{
    float x1;
    float y1;
    float x2;
    float y2;
    float score;
};

class FaceDetector{
private:
    char* mModelBuffer = nullptr;
    long mModelSize;
    bool mModelQuantized = false;
    const int OUTPUT_WIDTH = 640;
    const int OUTPUT_HEIGHT = 480;
    std::unique_ptr<tflite::FlatBufferModel> mModel;
    std::unique_ptr<tflite::Interpreter> mInterpreter;
    TfLiteTensor* mInputTensor = nullptr;
    TfLiteTensor* mOutputHeatmap = nullptr;
    TfLiteTensor* mOutputScale = nullptr;
    TfLiteTensor* mOutputOffset = nullptr;

    int d_h;
    int d_w;
    float d_scale_h;
    float d_scale_w;
    float scale_w ;
    float scale_h ;
    int image_h;
    int image_w;

public:
    FaceDetector(char* buffer, long size, bool quantized=false);
    ~FaceDetector();
    void detect(cv::Mat img, std::vector<FaceBoxInfo>& faces,
            float scoreThresh, float nmsThresh);

private:
    void loadModel();
    
    void dynamic_scale(float in_w, float in_h);
    
    void postProcess(float* heatmap, float* scale, float* offset,
                     std::vector<FaceBoxInfo>& faces,
                     float heatmapThreshold, float nmsThreshold);
    
    void nms(std::vector<FaceBoxInfo>& input, std::vector<FaceBoxInfo>& output,
            float nmsThreshold);
    
    std::vector<int> filterHeatmap(float* heatmap, int h, int w, float thresh);
    
    void getBox(std::vector<FaceBoxInfo>& faces);
};

#endif //FACE_DETECTION_HPP
