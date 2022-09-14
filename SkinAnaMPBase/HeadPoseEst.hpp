//
//  HeadPoseEst.hpp
//
//
/*

Author: Fu Xiaoqiang
Date:   2022/9/10

本模块采用14个点人脸关键点，利用PnP算法来计算人脸的位姿。
2D关键点来自MediaPipe中的face mesh模型的推理结果（实现是利用Tensorflow Lite），
3D人脸参考模型来自网络：

*/

#ifndef HEAD_POSE_EST_HPP
#define HEAD_POSE_EST_HPP

#include<opencv2/opencv.hpp>

using namespace std;
using namespace cv;


/******************************************************************************************
srcImgWidht, srcImgHeight: the width and height of the input source image.
lm_2d: as the input argument, extracted from the source image, and measured in 
the coordinate system of the source image. 
pitch, yaw, roll: are output arguments, measured in degrees.
*******************************************************************************************/
void EstHeadPose(int srcImgWidht, int srcImgHeight, 
	int lm_2d[468][2], float& pitch, float& yaw, float& roll);

#endif /* end of HEAD_POSE_EST_HPP */
