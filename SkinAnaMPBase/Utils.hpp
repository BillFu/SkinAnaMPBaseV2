//
//  Utils.hpp
//
//
/*
本模块提供一些简单的辅助性函数。
 
Author: Fu Xiaoqiang
Date:   2022/9/11
*/

#ifndef UTILS_HPP
#define UTILS_HPP

#include "opencv2/opencv.hpp"

using namespace std;
using namespace cv;

/**********************************************************************************************
将缩小版(大小为H*W)的影像“喂”给TF Lite网络的输入端，图像采用BGR通道次序。
同时，在“喂”之前，对像素值Normalization，使之变为Float，取值范围为[0.0 1.0]。
imgDataPtr已经是缩小版的输入影像了。
***********************************************************************************************/
void FeedInputWithNormalizedImage(uint8_t* imgDataPtr, float* netInputBuffer, int H, int W, int C);

//-------------------------------------------------------------------------------------------

/**********************************************************************************************
将缩小版(大小为H*W)的影像“喂”给TF Lite网络的输入端，图像采用BGR通道次序。
同时，在“喂”之前，对像素值Quantization，使之变为Float，取值范围为[-1.0 1.0]。
imgDataPtr已经是缩小版的输入影像了。
***********************************************************************************************/
void FeedInputWithQuantizedImage(uint8_t* imgDataPtr, float* netInputBuffer, int H, int W, int C);

//-------------------------------------------------------------------------------------------

// return a string that present a float with 2 decimal digits.
// for example, return "3.14" for 3.1415927
string convertFloatToStr2DeciDigits(float value);


/**********************************************************************************************
采用Padding的方式，将原始图像扩充为一个正方形。
上下左右都扩展。
先扩充上下，newH = srcH * alpha, alpha取[0.2 0.4]
依据newH，再扩展左右，使之成为一个正方形
正方形的边长等于newH，原始像素矩阵在正方形中居中。
填充的像素取黑色。
***********************************************************************************************/
void MakeSquareImageV2(const Mat& srcImg, float deltaHRatio, Mat& squareImg);
//-------------------------------------------------------------------------------------------

/**********************************************************************************************
采用Shifting & Padding的方式，将原始图像进行几何方面的修正，使之适合于输入Face Mesh With Attention模型提取关键点。
基本的要求：
1. Face CP should be located in the center of the newly forged image.
2. 25% margin should be remained at both sides of the output image. 25% for each side.
3. In the output image, the ratio of Height vs Widht should be 1.4.
4. the raw content of source image should be contained completely in the out image.
5. scale not changed.
FV: front view
BBox: Bounding Box
CP: center point, i.e., the midway of the line connecting the center points of two eyes.
***********************************************************************************************/
void GeoFixFVSrcImg(const Mat& srcImg, const Rect& faceBBox,
                    const Point2i& faceCP, float alpha, Mat& outImg);
//-------------------------------------------------------------------------------------------



#endif /* end of UTILS_HPP */
