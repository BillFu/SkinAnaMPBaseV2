//
//  Wrinkle.cpp

/*******************************************************************************
本模块负责检测皱纹和相关处理。

Author: Fu Xiaoqiang
Date:   2022/11/1

********************************************************************************/

#include "../Utils.hpp"

#include "Wrinkle.hpp"
#include "frangi.h"
#include "WrinkleFrangi.h"
#include "WrinkleGabor.h"
#include "../ImgProc.h"




//-------------------------------------------------------------------------------------------
// Frangi + Gabor Filtering

// wrkGaborRespMap: 输出，记录Gabor滤波的结果，大小和位置由Face_Rect来限定
void DetectWrinkle(const Mat& inImg, const Rect& faceRect,
                   const Mat& wrkFrgiMask,
                   WrkRegGroup& wrkRegGroup,
                   CONTOURS& deepWrkConts,
                   CONTOURS& lightWrkConts,
                   int& numLongWrk, int& numShortWrk,
                   int& numDeepWrk, int& numLightWrk,
                   Mat& wrkGaborMap)
{
    cv::Mat imgGray;
    cvtColor(inImg, imgGray, COLOR_BGR2GRAY);

    //------------------ 第一次是使用Frangi2d滤波，针对粗皱纹 --------------------
    // 计算Frangi滤波响应，并提取深皱纹和长皱纹
    
    int scaleRatio = 4;
    Mat frgiMapSSInFR; // SS: source scale
    CcFrgiMapInFR(imgGray, faceRect,
                    scaleRatio, frgiMapSSInFR);

    float avgFrgiMapV;
    CONTOURS longWrkConts;
    int longWrkTh = 0.16 * faceRect.width;  // 大致为2cm
    int minWrkTh = longWrkTh / 2; // 皱纹（包括长、短皱纹）的最短下限，大致为1cm
    cout << "minWrkTh: " << minWrkTh << endl;
    PickWrkInFrgiMap(wrkFrgiMask(faceRect),
                     minWrkTh, longWrkTh,
                     frgiMapSSInFR,
                     deepWrkConts, longWrkConts, avgFrgiMapV);
    
#ifdef TEST_RUN2
    Mat canvas = inImg(faceRect).clone();
    drawContours(canvas, deepWrkConts, -1, cv::Scalar(255, 0, 0), 2);
    drawContours(canvas, longWrkConts, -1, cv::Scalar(0, 0, 255), 2);

    string frgiDLWrkFile = wrkOutDir + "/DLWrkFrgi.png";
    imwrite(frgiDLWrkFile.c_str(), canvas);
#endif
    
    //----------------- 第二次使用Gabor滤波，针对细皱纹---------------------
    //WrinkRespMap是由Face_Rect来限定的
    CalcGaborMap(imgGray, wrkRegGroup, wrkGaborMap);
    wrkGaborMap = wrkGaborMap & wrkFrgiMask; // !!!
    
#ifdef TEST_RUN2
    string gaborMapFile = wrkOutDir + "/gaborMap.png";
    imwrite(gaborMapFile.c_str(), wrkGaborMap);
#endif
    
    int totalWrkLen = 0;
    //Mat wrkFrgiMaskInFR = wrkFrgiMask(faceRect);
    ExtLightWrk(wrkGaborMap, minWrkTh, longWrkTh, lightWrkConts, longWrkConts, totalWrkLen);
    ExtDeepWrk(wrkGaborMap, minWrkTh, longWrkTh, deepWrkConts, longWrkConts);

    numLongWrk = (int)(longWrkConts.size());
    numLightWrk = (int)(lightWrkConts.size());
    numDeepWrk = (int)(deepWrkConts.size());
    numShortWrk = numDeepWrk + numLightWrk - numLongWrk;
    
}
