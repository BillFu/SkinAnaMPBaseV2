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
    Mat fhFrgiMap8U = CcFrgiMapInRect(imgGray,
            wrkRegGroup.fhReg.bbox, scaleRatio);

    Mat fhMaskGS = TransMaskFromLS2GS(inImg.size(), wrkRegGroup.fhReg);
    fhFrgiMap8U = fhFrgiMap8U & fhMaskGS;
    fhMaskGS.release();
    
    CONTOURS longWrkConts;
    int longWrkTh = 0.12 * faceRect.width;  // 大致为2cm
    int minWrkTh = longWrkTh / 2; // 皱纹（包括长、短皱纹）的最短下限，大致为1cm
    cout << "minWrkTh: " << minWrkTh << endl;
    PickDLWrkInFrgiMapV2(minWrkTh, longWrkTh,
                         fhFrgiMap8U,
                     deepWrkConts, longWrkConts);
    
    //cv::Scalar sumResp = cv::sum(frgiMap8U);
    //int nonZero2 = cv::countNonZero(DLWrkMaskGS); // Mask中有效面积，即非零元素的数目
    //float avgFrgiMapV = sumResp[0] / (nonZero2+1);  // 平均响应值，不知道为啥要除以12.8
    
#ifdef TEST_RUN2
    Mat canvas = inImg.clone();
    drawContours(canvas, deepWrkConts, -1, cv::Scalar(255, 0, 0), 2); 
    drawContours(canvas, longWrkConts, -1, cv::Scalar(0, 0, 255), 2);

    string frgiDLWrkFile = wrkOutDir + "/DLWrkFhFrgi.png";
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
    
    CONTOURS deepWrkConts1, lightWrkConts1;
    ExtLightWrk(wrkGaborMap, minWrkTh, longWrkTh, lightWrkConts1, longWrkConts, totalWrkLen);
    ExtDeepWrk(wrkGaborMap, minWrkTh, longWrkTh, deepWrkConts1, longWrkConts);

    numLongWrk = (int)(longWrkConts.size());
    numLightWrk = (int)(lightWrkConts.size());
    numDeepWrk = (int)(deepWrkConts.size());
    numShortWrk = numDeepWrk + numLightWrk - numLongWrk;
    
    // 只显示浅皱纹和深皱纹
    Mat outDLWrkImg = forgeWrkAnno(inImg.size(), lightWrkConts1, deepWrkConts1);
        
#ifdef TEST_RUN2
    string outDLWrkImgFile =  wrkOutDir + "/DLWrkImg.png";
    imwrite(outDLWrkImgFile.c_str(), outDLWrkImg);
    
    Mat wrkAnnoImg = SpWrkOnSrcImg(inImg, lightWrkConts, deepWrkConts);
    string wrkAnnoImgFile =  wrkOutDir + "/WrkAnnoImg.png";
    imwrite(wrkAnnoImgFile.c_str(), wrkAnnoImg);
#endif
        
    //return outDLWrkImg;
}

// 把检测出的浅皱纹和深皱纹在背景图像上画出来
// 返回的标注图像是4通道，即RGBA。
Mat forgeWrkAnno(const Size& mapSize,
                   const CONTOURS& LightWrkConts,
                   const CONTOURS& DeepWrkConts)
{
    Mat rgbMat(mapSize, CV_8UC3, cv::Scalar(0, 0, 0));
    //浅皱纹用黄色显示
    drawContours(rgbMat, LightWrkConts, -1, cv::Scalar(53, 255, 148), 2);
    //深皱纹用绿色显示
    drawContours(rgbMat, DeepWrkConts, -1, cv::Scalar(0, 178, 0), 2);
    
    Mat alpha_channel(mapSize, CV_8UC1, Scalar(0, 0, 0));
    drawContours(alpha_channel, LightWrkConts, -1, cv::Scalar(255, 255, 255), 2);
    drawContours(alpha_channel, DeepWrkConts, -1, cv::Scalar(255, 255, 255), 2);
    
    vector<cv::Mat> rgb_channels;
    split(rgbMat, rgb_channels);
    
    Mat fourChans[] = {rgb_channels[0], rgb_channels[1], rgb_channels[2], alpha_channel};
    
    cv::Mat resImg(mapSize, CV_8UC4, cv::Scalar(0, 0, 0, 0));
    merge(fourChans, 4, resImg);
    
    return resImg;
}

// Sp: superposition
Mat SpWrkOnSrcImg(const Mat srcImg,
                  const CONTOURS& LightWrkConts,
                  const CONTOURS& DeepWrkConts)
{
    Mat annoImg = srcImg.clone();

    //浅皱纹用黄色显示
    drawContours(annoImg, LightWrkConts, -1, cv::Scalar(53, 255, 148), 2);
    //深皱纹用绿色显示
    drawContours(annoImg, DeepWrkConts, -1, cv::Scalar(0, 178, 0), 2);
    
    return annoImg;
}
