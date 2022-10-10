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


// store the information about face which has been refined out from the segment labels
struct FaceSegResult
{
    // all the coordinates and sizes are measured in the space of the source image.
    Size        srcImgS;
    Rect        faceBBox;
    Point2i     faceCP;         // CP: Center Point;
    Point2i     eyeCPs[2];      // the area size of No.0 in image is bigger than No.1
    int         eyeAreas[2];    // in pixels in source image space
    float       eyeAreaDiffRatio;  // ratio = abs(a1-a2) / max(a1, a2)
    Point2i     leftBrowCP;     // center point of left eyebrow
    Point2i     rightBrowCP;    // center point of right eyebrow
    bool        isFrontView;
    Mat         segLabels;      // 512 * 512, one channel
    
    FaceSegResult()
    {
        segLabels = Mat(512, 512, CV_8UC1, Scalar(0));
    }
    
    Point2i getLeftEyeCP() const
    //the trailing const makes the "this" parameter const,
    //meaning that you can invoke the method on const objects of the class type,
    //and that the method cannot modify the object on which it was invoked
    //(at least, not via the normal channels).
    {
        if(eyeCPs[0].x < eyeCPs[1].x)
            return eyeCPs[0];
        else
            return eyeCPs[1];
    }
    
    Point2i getRightEyeCP() const
    //the trailing const makes the "this" parameter const,
    //meaning that you can invoke the method on const objects of the class type,
    //and that the method cannot modify the object on which it was invoked
    //(at least, not via the normal channels).
    {
        if(eyeCPs[0].x > eyeCPs[1].x)
            return eyeCPs[0];
        else
            return eyeCPs[1];
    }
    
    friend ostream &operator<<(ostream &output, const FaceSegResult &fpi )
    {
        output << "FacePrimaryInfo{" << endl;
        
        output << "faceBBox: " << fpi.faceBBox << endl;
        output << "faceCP: " << fpi.faceCP << endl;
        output << "eyeCPs1: " << fpi.eyeCPs[0] << endl;
        output << "eyeCPs2: " << fpi.eyeCPs[1] << endl;
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
    int imgWidth;  // source image
    int imgHeight;
    
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
        imgWidth = 0;
        imgHeight = 0;
        
        confidence = 0.0;
    }
};

#endif /* end of COMMON_HPP */
