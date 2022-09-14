//
//  FaceLmExtract.hpp
//
//
/*
本模块的功能是，利用tensorflow lite C++ API来驱动MediaPipe中内含的Face Detection深度学习网络，
来推理获取人脸图像中的Face Landmarks.
目前的推理是一次性的，即从模型加载到解释器创建，到推理，到结果提取，这一流程只负责完成一副图像的处理。
 
Author: Fu Xiaoqiang
Date:   2022/9/11
*/

#ifndef FACE_DETECT_HPP
#define FACE_DETECT_HPP



#include<opencv2/opencv.hpp>
#include "Common.hpp"

using namespace std;
using namespace cv;


//-----------------------------------------------------------------------------------------

/******************************************************************************************
该函数的功能是，加载Face Mesh模型，生成深度网络，创建解释器并配置它。
return true if all is well done, otherwise reurn false and give the error reason.
numThreads: 解释器推理时可以使用的线程数量，最低为1.
*******************************************************************************************/

TF_LITE_MODEL LoadFaceDetectModel(const char* faceDetectModelFileName);

//-----------------------------------------------------------------------------------------

/******************************************************************************************
if all is OK, return true; otherwise return false, and errorMsg will told you what has happened.

hasFace: indicate whether a face appeared or not.
*******************************************************************************************/

bool DetectFace(const TF_LITE_MODEL& faceDetectModel, const Mat& srcImage,
                bool& hasFace, float& confidence,
                   string& errorMsg);


#endif /* end of FACE_DETECT_HPP */
