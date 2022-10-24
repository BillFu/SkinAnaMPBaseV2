/*
EyebrowMaskV5.hpp

本模块用于计算Eyebrow Mask。
基本思路：
1. 每只眼睛单独处理。
2. 利用一个眼睛的左、右角点，将这个眼睛的轮廓线分为上弧线、下弧线。整个轮廓的点序列顺序为逆时针，
   上弧线从右角点到左角点，下弧线由左角点到右角点。
3. 将上、下弧线分别进行二次曲线拟合，而后重新采样获得光滑的上、下弧线。
4. 光滑后的上、下弧线组合成新的眼睛轮廓线。
 
Author: Fu Xiaoqiang
Date:   2022/10/24
*/

#ifndef EYEBROW_MASK_V5_HPP
#define EYEBROW_MASK_V5_HPP

#include "opencv2/opencv.hpp"

using namespace std;
using namespace cv;

#include "../Common.hpp"


// Ip: interpolate
Point2i IpPtInERG(const Point2i eyeRefPts[NUM_PT_EYE_REFINE_GROUP],
                  int pIndex1, int pIndex2, float t);


//ERPs: eye refined points
// 用分割出的眉毛中心点来修正ERPs的坐标
// transform the eye refine points in face mesh space into the segment space
void FixERPsBySegBrowCP(const Point2i eyeRefPts[NUM_PT_EYE_REFINE_GROUP], // input
                       const Point2i& segCP, // eye center point in segment space
                       Point2i fixedEyeRefPts[NUM_PT_EYE_REFINE_GROUP]   // output
);
/**********************************************************************************************

***********************************************************************************************/
// Note: Only forge One!!!
void ForgeBrowPg(const Point2i eyeRefPts[NUM_PT_EYE_REFINE_GROUP],
                 const Point2i& segBrowCP,POLYGON& browPg);

void ForgeBrowsMask(const FaceInfo& faceInfo,
                    const FaceSegRst& segResult, // input, const
                    Mat& outMask);

/**********************************************************************************************
only cover the two eyes
***********************************************************************************************/

/*
void ForgeEyePgBySegRst(Size srcImgS, const SegMask& eyeSegMask, const Point2i& eyeCP,
                        POLYGON& eyePg);
*/

// 用分割的结果构造出一只眼睛的轮廓多边形
void ForgeEyePgBySegRstV2(Size srcImgS,
                          const SegMask& eyeSegMask,
                          const SegEyeFPsNOS& eyeFPsNOS,
                          POLYGON& eyePg);


void ForgeEyesMask(const FaceInfo& faceInfo,
                   const FaceSegRst& segResult, // input, const 
                   Mat& outMask);


// 按轮廓序列的自然顺序遍历轮廓点，先找到上弧线中点，而后继续遍历，看先碰到左角点还是右角点，
// 以此来判断出我们的遍历是顺时针，还是逆时针。
// 真个流程最多需要遍历两次。按照抽象出来的环形数据结构方式来遍历。
void JudgeEyeCtNOSMoveDir(const CONTOUR& eyeCont,
                          const SegEyeFPsNOS& eyeFPsNOS,
                          CLOCK_DIR& scanDir);

// 提取眼睛轮廓线的上、下弧线
void SplitEyeCt2UpLowCurves(const CONTOUR& nosEyeCont,
                            const Point2i& nosLCorPt,
                            const Point2i& nosRCorPt,
                            CONTOUR& upEyeCurve,
                            CONTOUR& lowEyeCurve);

#endif /* end of EYEBROW_MASK_V5_HPP */
