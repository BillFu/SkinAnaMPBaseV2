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

#include <filesystem>

#include "opencv2/opencv.hpp"

using namespace std;
using namespace cv;

namespace fs = std::filesystem;

string openCVType2str(int type);

bool isInImg(const Point2i& pt, int cols, int rows);

/**********************************************************************************************
RC: rows and cols
the rows and cols of the out image can be divied by 4
***********************************************************************************************/
void PadImgWithRC4Div(Mat& srcImg); //, Mat& outImg);


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
void FeedInWithQuanImage(uint8_t* imgDataPtr, float* netInputBuffer, int H, int W, int C);

//-------------------------------------------------------------------------------------------
// another implementation of FeedInWithQuanImage()
void FeedPadImgToNet(const cv::Mat& resizedPadImg, float* inTensorBuf);

// return a string that present a float with 2 decimal digits.
// for example, return "3.14" for 3.1415927
string convertFloatToStr2DeciDigits(float value);

// FP: full path
string BuildOutImgFileName(const fs::path& outDir,
                         const string& fileNameBone,
                         const string& outPrefix);

// File Bone Name: no path and no extension
// the bone name of "images/JPN/cross_2.jpg" is "cross_2"
string GetFileBoneName(string fileName);

//-------------------------------------------------------------------------------------------
int convSegNetY2SrcY(int srcImgH, int segNetY);
int convSegNetX2SrcX(int srcImgW, int segNetX);

int convSrcY2SegNetY(int srcImgH, int srcY);
int convSrcX2SegNetX(int srcImgW, int srcX);

Point2i convSegNetPt2SrcPt(const Size& srcImgS, const Point2i& snPt);
Point2i convSrcPt2SegNetPt(const Size& srcImgS, const Point2i& srcPt);

//-------------------------------------------------------------------------------------------
template<typename T>
T stddev(std::vector<T> const & func);

#endif /* end of UTILS_HPP */
