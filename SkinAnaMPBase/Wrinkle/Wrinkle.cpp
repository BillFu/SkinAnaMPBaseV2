//
//  Wrinkle.cpp

/*******************************************************************************
本模块负责检测皱纹和相关处理。

Author: Fu Xiaoqiang
Date:   2022/11/1

********************************************************************************/

#include "../Utils.hpp"

#include "Wrinkle.hpp"
#include "cvgabor.h"
#include "frangi.h"
#include "WrinkleFrangi.h"
#include "WrinkleGabor.h"
#include "../ImgProc.h"


void PreprocGrImg(const Mat& grSrcImg,
                    Mat& outImg)
{
    Size srcImgS = grSrcImg.size();
    
    // 这个公式仅对前额区域有效；若imgW表示图像全域或其他子区域，这个公式需要调整。
    // 也许这个公式以后需要调整为普遍适用的公式。
    int blurKerS = srcImgS.width / 272;
    if(blurKerS % 2 == 0)
        blurKerS += 1;  // make it be a odd number
    
    Mat blurGrImg;
    blur(grSrcImg, blurGrImg, Size(blurKerS, blurKerS));
    
    // 这个公式仅对前额区域有效；若imgW表示图像全域或其他子区域，这个公式需要调整。
    // 也许这个公式以后需要调整为普遍适用的公式。
    int gridSize = srcImgS.width / 100;
    ApplyCLAHE(blurGrImg, gridSize, outImg);
    blurGrImg.release();
}

//-------------------------------------------------------------------------------------------
// Frangi + Gabor Filtering

// wrkGaborRespMap: 输出，记录Gabor滤波的结果，大小和位置由Face_Rect来限定
void DetectWrinkle(const Mat& inImg, const Rect& faceRect,
                   const Mat& wrkFrgiMask,
                   WrkRegGroup& wrkRegGroup,
                   CONTOURS& deepWrkConts,
                   Mat& wrkGaborRespMap)
{
    cv::Mat imgGray;
    cvtColor(inImg, imgGray, COLOR_BGR2GRAY);

    //------------------ 第一次是使用Frangi2d滤波，针对粗皱纹 --------------------
    // 计算Frangi滤波响应，并提取深皱纹和长皱纹
    Mat ehGrImg; // eh: enhanced
    PreprocGrImg(imgGray, ehGrImg);
    
    int scaleRatio = 2;
        
    Mat frgiMapSSInFR; // SS: source scale
    CcFrgiRespInFR(ehGrImg, faceRect,
                    scaleRatio, frgiMapSSInFR);

    float avgFrgiRespValue;
    CONTOURS longWrkConts;
    int longWrkThresh = 0.16 * faceRect.width;
    int minWrkSize = longWrkThresh / 2; // 皱纹（包括长、短皱纹）的最短下限
    PickWrkFromFrgiMap(wrkFrgiMask(faceRect),
                        minWrkSize, longWrkThresh,
                        frgiMapSSInFR, deepWrkConts, longWrkConts, avgFrgiRespValue);
    //----------------- 第二次使用Gabor滤波，针对细皱纹---------------------
    //WrinkRespMap是由Face_Rect来限定的
    CalcGaborResp(ehGrImg, wrkRegGroup, wrkGaborRespMap);

#ifdef TEST_RUN
    /*
    string gaborMapFile =  wrk_out_dir + "/gaborResp.png";
    imwrite(gaborMapFile.c_str(), wrkGaborRespMap);
    */
#endif
    
}
