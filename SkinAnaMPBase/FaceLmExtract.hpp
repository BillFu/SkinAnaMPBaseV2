//
//  FaceLmAttenExtract.hpp
//
//
/*
本模块的功能是，利用tensorflow lite C++ API来驱动MediaPipe中内含的Face Mesh With Attention
深度学习网络，来推理获取人脸图像中的Face Landmarks（包含的内容比较丰富）。
目前的推理是一次性的，即从模型加载到解释器创建，到推理，到结果提取，只负责完成一副人脸的LM提取。
 
Author: Fu Xiaoqiang
Date:   2022/9/22
*/

#ifndef FACE_LM_EXTRACT_HPP
#define FACE_LM_EXTRACT_HPP

#include <opencv2/opencv.hpp>
#include "Common.hpp"

using namespace std;
using namespace cv;
using namespace tflite;

//-----------------------------------------------------------------------------------------

/******************************************************************************************
该函数的功能是，加载Face Mesh Attention模型，生成深度网络，创建解释器并配置它。
return true if all is well done, otherwise reurn false and give the error reason.
numThreads: 解释器推理时可以使用的线程数量，最低为1.
*******************************************************************************************/

TF_LITE_MODEL LoadFaceMeshAttenModel(const char* faceMeshModelFileName);

//-----------------------------------------------------------------------------------------

/******************************************************************************************
 目前的推理是一次性的，即从模型加载到解释器创建，到推理，到结果提取，这一流程只负责完成一副人脸的LM提取。
 以后要让前半段的结果长期存活，用于连续推理，以提高效率。
 
if confidence >= confTh, then a face will be confirmed, hasFace will be assigned with true.
Note: after invoking this function, return value and hasFace must be check!
*******************************************************************************************/
bool ExtractFaceLm(const TF_LITE_MODEL& face_lm_model, const Mat& srcImage,
                    float confTh, bool& hasFace, float& confidence,
                    FaceInfo& faceInfo, string& errorMsg);



//-----------------------------------------------------------------------------------------

/******************************************************************************************
convert the coordinates of LM extracted from the Padded image into the coordinates
of source image space.
dummyFI: the coordiantes measured in padded image space.
srcSpaceFI: the coordinates measured in the source iamge space.
alpha: deltaH / srcH
*******************************************************************************************/
void padCoord2SrcCoord(int srcW, int srcH, float alpha,
                       const FaceInfo& dummyFI, FaceInfo& srcSpaceFI);

#endif /* end of FACE_LM_EXTRACT_HPP */
