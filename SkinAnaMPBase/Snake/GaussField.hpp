/*
GaussField.hpp

本模块用于生成高斯场。
 
// https://docs.opencv.org/3.4/d4/d7d/tutorial_harris_detector.html
// https://docs.opencv.org/3.4/d8/dd8/tutorial_good_features_to_track.html

Author: Fu Xiaoqiang
Date:   2022/10/28
*/

#ifndef GAUSSIAN_FIELD_HPP
#define GAUSSIAN_FIELD_HPP

#include "opencv2/opencv.hpp"

using namespace std;
using namespace cv;

#include "../Common.hpp"

// void DetectCorner_Harris();
//void DetectCorner_ST(const Mat& srcImg); // shi-tomasi

Mat BuildGaussField(int fieldW, int fieldH, int sigma,
                     const vector<Point2i>& peaks);

#endif /* end of GAUSSIAN_FIELD_HPP */
