//
//  FaceLmExtractV2.hpp
//
//
/*
本模块的功能是，利用tensorflow lite C++ API来驱动MediaPipe中内含的Face Mesh With Attention
深度学习网络，来推理获取人脸图像中的Face Landmarks（包含的内容比较丰富）。
目前的推理是一次性的，即从模型加载到解释器创建，到推理，到结果提取，只负责完成一副人脸的LM提取。
相比上一个版本，变化的是Shifting&Padding的方式。方法是，按照FaceBBox（由人脸/背景分割模块提供）
将脸部图像从源图像中挖出来，再给四周各扩充宽、高的25%，送入推理引擎。获得Lm之后，再变换到原始图像空间。
在Padding之前（或之后）再加上旋转，使两眼连线为水平。
Author: Fu Xiaoqiang
Date:   2022/10/6
*/

#ifndef FACE_LM_EXTRACT_V3_HPP
#define FACE_LM_EXTRACT_V3_HPP

#include <opencv2/opencv.hpp>
#include "Common.hpp"

using namespace std;
using namespace cv;
using namespace tflite;

#define FACE_MESH_NET_INPUT_W 192
#define FACE_MESH_NET_INPUT_H 192

struct NormalLmSet
{
    double normal_lm_2d[NUM_PT_GENERAL_LM][2];
    double LNorEyeBowPts[NUM_PT_EYE_REFINE_GROUP][2];
    double RNorEyeBowPts[NUM_PT_EYE_REFINE_GROUP][2];
    double NorLipRefinePts[NUM_PT_LIP_REFINE_GROUP][2];
};

//-----------------------------------------------------------------------------------------

extern TF_LITE_MODEL tfLiteFMMoel;  // FM: face mesh

/******************************************************************************************
该函数的功能是，加载Face Mesh Attention模型，生成深度网络，创建解释器并配置它。
return true if all is well done, otherwise reurn false and give the error reason.
numThreads: 解释器推理时可以使用的线程数量，最低为1.
*******************************************************************************************/

bool LoadFaceMeshModel(const char* faceMeshModelFileName, string& errorMsg);

//-----------------------------------------------------------------------------------------

/******************************************************************************************
目前的推理是一次性的，即从模型加载到解释器创建，到推理，到结果提取，这一流程只负责完成一副人脸的LM提取。
以后要让前半段的结果长期存活，用于连续推理，以提高效率。
 
if confidence >= confTh, then a face will be confirmed, hasFace will be assigned with true.
Note: after invoking this function, return value and hasFace must be check!
 
if needPadding is true, padding would be applied in this function, and vertPadRatio
specify how much ratio of height should be expanded vertically.
More, if padding on, the final padded image would be a square.
*******************************************************************************************/
bool ExtractFaceLm(const Mat& srcImage,
                    float confTh, const FaceSegResult& segResult,
                    bool& hasFace,
                    FaceInfo& faceInfo, string& errorMsg);


//-----------------------------------------------------------------------------------------

/******************************************************************************************
convert the coordinates of LM extracted from the padded image space into the coordinates
of source image space.
Cd: coordinate
*******************************************************************************************/
/*
void PadCd2SrcCd_All(const cv::Size& padImgS, int TP, int LP,
                     const Point2i& bboxTL,
                     const NormalLmSet& normalLmSet, //Normal means [0.0 1.0]
                     FaceInfo& srcSpaceFI);
*/



//TP, LP: the sizes of Top Padding and Left Padding
//bboxTL: the top and left corner of face bbox meansured in the source image space.
void CalcLmsInFixSpace(const cv::Size& fixImgS,
                         const NormalLmSet& normalLmSet,
                       FaceInfo& fixSpaceFI);


#endif /* end of FACE_LM_EXTRACT_V3_HPP */
