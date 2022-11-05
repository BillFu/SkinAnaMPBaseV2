//
//  Wrinkle.hpp
//
//
/*
本模块负责检测皱纹和相关处理。
 
Author: Fu Xiaoqiang
Date:   2022/11/1
*/

#ifndef WRINKLE_HPP
#define WRINKLE_HPP

#include "opencv2/opencv.hpp"

using namespace std;
using namespace cv;

#include "../Common.hpp"

void DetectWrinkle(const Mat& inImg, const Rect& faceRect);

#endif /* end of WRINKLE_HPP */
