//
//  Wrinkle.hpp
//
//
/*
本模块负责检测皱纹和相关处理。
 
Author: Fu Xiaoqiang
Date:   2022/11/15
*/

#ifndef WRINKLE_V2_HPP
#define WRINKLE_V2_HPP

#include "opencv2/opencv.hpp"

using namespace std;
using namespace cv;

#include "../Common.hpp"


void PreprocGrImg(const Mat& grSrcImg,
                  Mat& outImg);

// wrkGaborRespMap: 输出，记录Gabor滤波的结果，大小和位置由Face_Rect来限定
void DetectWrinkle(const Mat& inImg, const Rect& faceRect,
                   WrkRegGroup& wrkRegGroup,
                   CONTOURS& deepWrkConts,
                   CONTOURS& lightWrkConts, // light, 轻的
                   int& numLongWrk, int& numShortWrk,
                   int& numDeepWrk, int& numLightWrk);

// 把检测出的浅皱纹和深皱纹在背景图像上画出来
// 返回的标注图像是4通道，即RGBA。
Mat forgeWrkAnno(const Size& mapSize,
                   const CONTOURS& LightWrkConts,
                   const CONTOURS& DeepWrkConts);


// Sp: superposition
Mat SpWrkOnSrcImg(const Mat srcImg,
                  const CONTOURS& LightWrkConts,
                  const CONTOURS& DeepWrkConts);

#endif /* end of WRINKLE_V2_HPP */
