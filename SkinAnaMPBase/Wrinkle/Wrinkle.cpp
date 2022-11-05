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


//-------------------------------------------------------------------------------------------
// Frangi + Gabor Filtering

// wrkGaborRespMap: 输出，记录Gabor滤波的结果，大小和位置由Face_Rect来限定
void DetectWrinkle(const Mat& inImg, const Rect& faceRect,
                   const Mat& wrkMask,
                   const SPLINE& wrkSpline,
                   CONTOURS& deepWrkConts,
                   Mat& wrkGaborRespMap)
{
    cv::Mat imgGray;
    cvtColor(inImg, imgGray, COLOR_BGR2GRAY);

    // crop input gray image by Face Bounding Box, i.e., faceRect
    Mat grFrImg; // the copy of input image cropped by FaceRect
    imgGray(faceRect).copyTo(grFrImg);
    imgGray.release();
    
    Mat wrkMaskInFR; // wrinkle mask cropped by face rect
    wrkMask(faceRect).copyTo(wrkMaskInFR);

    int longWrkThresh = 0.16 * grFrImg.cols;;
    int minWrkSize = longWrkThresh / 2; // 皱纹（包括长、短皱纹）的最短下限

    //------------------ 第一次是使用Frangi2d滤波，针对粗皱纹 --------------------
    
    // 计算Frangi滤波响应，并提取深皱纹和长皱纹
    float avgFrgiRespValue;
    int scaleRatio = 5;
    Mat wrkRespRz;
    CONTOURS longWrkConts;
    CalcFrgiRespAndPickWrk(grFrImg, wrkMaskInFR, scaleRatio, minWrkSize,
                        longWrkThresh,
                        wrkRespRz, deepWrkConts, longWrkConts, avgFrgiRespValue);
       
    //----------------- 第二次使用Gabor滤波，针对细皱纹---------------------
    
#ifdef TEST_RUN
    /*
    // view the polygon of forehead used to do gabor filtering
    Mat annoImage = inImg.clone();
    int ptIDsInFh[] = {0, 3, 4, 5, 26, 27};
    AnnoPointsOnImg(annoImage, wrkSpline,
                        ptIDsInFh,  6);
    string fhSpPtsFile =  outDir + "/fhSpPts.png";
    imwrite(fhSpPtsFile.c_str(), annoImage);
    */
#endif
    
    GaussianBlur(grFrImg, grFrImg, cv::Size(11, 11), 0, 0); // ???
    
    //WrinkRespMap是由Face_Rect来限定的
    CalcGaborResp(grFrImg, faceRect, wrkSpline, wrkGaborRespMap);

#ifdef TEST_RUN
    /*
    string gaborMapFile =  wrk_out_dir + "/gaborResp.png";
    imwrite(gaborMapFile.c_str(), wrkGaborRespMap);
    */
#endif
    
}
