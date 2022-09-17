//
//  LM_Exporter.hpp
//
//
/*
本模块将提取出的全部LM数据导出为4个文本文件。
 
Author: Fu Xiaoqiang
Date:   2022/9/17
*/

#ifndef LM_EXPORTER_HPP
#define LM_EXPORTER_HPP

#include "opencv2/opencv.hpp"

using namespace std;
using namespace cv;

#include "Common.hpp"


void Export_lm_3d(FaceInfo& faceInfo, const char* outFileName);

void Export_lm_2d(FaceInfo& faceInfo, const char* outFileName);

void Export_Lip_RefinePts(FaceInfo& faceInfo, const char* outFileName);

void Export_TwoEyes_RefinePts(FaceInfo& faceInfo, const char* outFileName);

// 将目前的所有数据（也包含位姿数据）导出到一个文本文件中
// 数据在导出文件中分布的安排如下：
// 1. 按在FaceInfo定义时的数据成员先后来排序。
// 2. 每段数据前加一行文本标题，便于人工浏览。
void ExportLM_FullData(FaceInfo& faceInfo, const char* outFileName);

#endif /* end of LM_EXPORTER_HPP */
