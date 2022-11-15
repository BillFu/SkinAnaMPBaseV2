//
//  ImgProc.h
//  MCSkinAnaLib
//
//  Created by Fu on 2022/6/21.
//  Copyright © 2022 MeiceAlg. All rights reserved.
//
/*
本模块来自JPN_V4。
 
Author: Fu Xiaoqiang
Date:   2022/11/1
*/

#ifndef IMG_PROC_H
#define IMG_PROC_H

#include "opencv2/opencv.hpp"

using namespace cv;

#include "Common.hpp"

// functions moved here from Old version of MC_SkinAnalyze(.h&.mm)
void connectEdge(cv::Mat& src);

void removeBurrs(const cv::Mat & src, cv::Mat &dst);

// 给二值图像中的粗黑线“瘦身”
int BlackLineThinInBiImg(uchar *lpBits, int Width, int Height);

Mat Integral_2(const Mat& image);

//计算(2*d+1) * (2*d+1)领域里的像素灰度值的均值、方差。
void Local_MeanStd(const Mat& image, Mat &mean, Mat &_std, int d);

// ACE可以表示Automatic color equalization，或者Adaptive contrast enhancement
// 这里的ACE从实现代码来看，代表的是Adaptive Contrast Enhancement
// OpenCV图像处理专栏十四 | 基于Retinex成像原理的自动色彩均衡算法(ACE).
// https://pythontechworld.com/article/detail/TbevLKxeIEjS
// OpenCV图像处理专栏五 | ACE算法论文解读及实现
// https://cloud.tencent.com/developer/article/1552856
// 自适应对比度增强（ACE）算法原理及实现
// https://codeleading.com/article/65423852945/
void ACE(const Mat& image, Mat& result, int d, /* 邻域半径 */
         float Scale, float MaxCG);

void FindGradient1(cv::Mat& InputImage, cv::Mat& OutputImage);

void FindGradient(cv::Mat& InputImage, cv::Mat& OutputImage);

Mat worldGray(const cv::Mat& src);

// Cvt: convert
Mat CvtFtImgTo8U_MinMax(Mat& ftImg);

// negative values will be changed into 0,
// positive values will be scaled
Mat CvtFtImgTo8U_NoNega(Mat& ftImg);

void ApplyCLAHE(const Mat& inImg,
                int gridSize,
                Mat& outImg);

// 返回8邻域不为0的像素的数目
int getConnectDegree(const std::vector<int>& neibValues);
int getTurnNum(const std::vector<int>& neibValues);

POINT_SET get8NeibCoordinates(const Point2i& pt, const Rect& imgRect);

void get8NeibValues(const Point2i& pt, const Mat& img, const Rect& imgRect,
                    vector<int>& neibValues);

POINT_SET get16NeibCoordinates(const Point2i& pt, const Rect& imgRect);

void get16NeibValues(const Point2i& pt, const Mat& img, const Rect& imgRect,
                    vector<int>& neibValues);

// opencv 骨架提取/图像细化
// csdn: 徐大大平凡之路
void chao_thinimage(Mat &srcimage); //单通道、二值化后的图像

void DelDupPtOnCont(CONTOUR& aWrk);

#endif /* IMG_PROC_H */
