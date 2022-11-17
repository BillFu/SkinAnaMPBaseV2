//
//  WrinkleV2.cpp

/*******************************************************************************
本模块负责检测皱纹和相关处理。

Author: Fu Xiaoqiang
Date:   2022/11/15

********************************************************************************/

#include "../Utils.hpp"

#include "WrinkleV2.hpp"
#include "frangi.h"
#include "WrinkleFrangi.h"
#include "WrinkleGaborV2.h"
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
// Only Gabor Filtering

// wrkGaborRespMap: 输出，记录Gabor滤波的结果，大小和位置由Face_Rect来限定
void DetectWrinkle(const Mat& inImg, const Rect& faceRect,
                   WrkRegGroup& wrkRegGroup,
                   CONTOURS& deepWrkConts,
                   CONTOURS& lightWrkConts,
                   int& numLongWrk, int& numShortWrk,
                   int& numDeepWrk, int& numLightWrk)
{
    cv::Mat imgGray;
    cvtColor(inImg, imgGray, COLOR_BGR2GRAY);

    //WrinkRespMap是由Face_Rect来限定的
    Mat fhGabMap8U, glabGabMap8U, lEbGabMap8U, rEbGabMap8U, lNagvGabMap8U, rNagvGabMap8U;
    CalcGaborMap(imgGray, wrkRegGroup,
                 fhGabMap8U, glabGabMap8U,
                 lEbGabMap8U,rEbGabMap8U,
                 lNagvGabMap8U, rNagvGabMap8U);
    
    int totalWrkLen = 0;
    
    int longWrkTh = 0.20 * faceRect.width;  // 大致为2cm
    int minWrkTh = longWrkTh / 2; // 皱纹（包括长、短皱纹）的最短下限，大致为1cm
    cout << "minWrkTh: " << minWrkTh << endl;
    
    CONTOURS longWrkConts;
    ExtWrkFromFhGabMap(wrkRegGroup.fhReg.bbox, fhGabMap8U,
        minWrkTh,longWrkTh, deepWrkConts, longWrkConts);

    ExtWrkFromGlabGabMap(wrkRegGroup.glabReg.bbox,
                         glabGabMap8U, minWrkTh,longWrkTh,
                         deepWrkConts, longWrkConts);

    ExtWrkFromEgGabMap(wrkRegGroup.lEyeBagReg, lEbGabMap8U,
                       minWrkTh/2, longWrkTh/2,
                       lightWrkConts, longWrkConts);
    
    ExtWrkFromEgGabMap(wrkRegGroup.rEyeBagReg, rEbGabMap8U,
                       minWrkTh/2, longWrkTh/2,
                       lightWrkConts, longWrkConts);

    /*
    numLongWrk = (int)(longWrkConts.size());
    numLightWrk = (int)(lightWrkConts.size());
    numDeepWrk = (int)(deepWrkConts.size());
    numShortWrk = numDeepWrk + numLightWrk - numLongWrk;
    */
    // 只显示浅皱纹和深皱纹
    Mat outDLWrkImg = forgeWrkAnno(inImg.size(), longWrkConts, deepWrkConts);
        
#ifdef TEST_RUN2
    string outDLWrkImgFile =  wrkOutDir + "/DLWrkImg.png";
    imwrite(outDLWrkImgFile.c_str(), outDLWrkImg);
    
    Mat wrkAnnoImg = SpWrkOnSrcImg(inImg, lightWrkConts, deepWrkConts);
    string wrkAnnoImgFile =  wrkOutDir + "/NewWrkAnno.png";
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
