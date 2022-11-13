#include <algorithm>

#include "WrinkleFrangi.h"
#include "../ImgProc.h"
#include "../Utils.hpp"
#include "frangi.h"

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

// -------------------------------------------------------------------------
// 从Frangi滤波响应中提取深皱纹和长皱纹
void PickWrkInFrgiMap(const Mat& wrkMask,
                      int minWrkTh, int longWrkTh,
                      Mat& frgiResp8U,
                      CONTOURS& deepWrkConts,
                      CONTOURS& longWrkConts,
                      float& avgFrgiMapValue)
{
    // DL: deep and long
    PickDLWrkInFrgiMap(frgiResp8U,
                       wrkMask, // 原始尺度，经过了Face_Rect裁切
                       minWrkTh, longWrkTh,
                       longWrkConts, deepWrkConts);
    
    Mat frgiMapInMask = frgiResp8U & wrkMask;
    cv::Scalar sumResp = cv::sum(frgiMapInMask);
    int nonZero2 = cv::countNonZero(wrkMask); // Mask中有效面积，即非零元素的数目
    avgFrgiMapValue = sumResp[0] / (nonZero2+1) / 12.8;  // 平均响应值，不知道为啥要除以12.8
}

////////////////////////////////////////////////////////////////////////////////////////
/// 从frangi滤波的结果（经过了二值化、细化、反模糊化等处理）中，提取深皱纹、长皱纹
void PickDLWrkInFrgiMap(const Mat& frgiMap8U, //Original Scale
                        const Mat& wrkMask, // 原始尺度，经过了Face_Rect裁切
                        int minsWrkSize,
                        int longWrkThresh,
                        CONTOURS& longWrkConts,
                        CONTOURS& deepWrkConts)
{
    Mat tmp_m, tmp_sd;
    double m = 0, sd = 0;
    //Mat frgiMapInMask = frgiMap8U & wrkMask;
    //meanStdDev(frgiMapInMask, tmp_m, tmp_sd);
    meanStdDev(frgiMap8U, tmp_m, tmp_sd);

    m = tmp_m.at<double>(0,0);
    sd = tmp_sd.at<double>(0,0);
    
    int thickBiTh = (int)(m + 4*sd);
    cout << "thickTh: " << thickBiTh << endl;
    
    cv::Mat thickBi; // thick: 浓的，厚的，粗的
    cv::threshold(frgiMap8U, thickBi, thickBiTh, 255, cv::THRESH_BINARY);
    
#ifdef TEST_RUN2
    string fraThRespFile =  wrkOutDir + "/FrgiBiResp.png";
    imwrite(fraThRespFile.c_str(), thickBi);
#endif
    
    thickBi = 255 - thickBi;
    int wdFR = wrkMask.cols;
    int htFR = wrkMask.rows;
    // 给二值图像中的粗黑线“瘦身”
    BlackLineThinInBiImg(thickBi.data, wdFR, htFR);
    
#ifdef TEST_RUN2
    string frgiThinRespFile = wrkOutDir + "/FrgiThinResp.png";
    imwrite(frgiThinRespFile.c_str(), thickBi);
#endif
    
    thickBi = 255 - thickBi;
    removeBurrs(thickBi, thickBi);
    thickBi = thickBi & wrkMask;
    
#ifdef TEST_RUN2
    string frgiFinalRespFile = wrkOutDir + "/FrgFinalResp.png";
    imwrite(frgiFinalRespFile.c_str(), thickBi);
#endif
    
    CONTOURS thickCts;
    findContours(thickBi, thickCts, RETR_EXTERNAL, CHAIN_APPROX_NONE);
    
    CONTOURS::const_iterator it_ct = thickCts.begin();
    unsigned long ct_size = thickCts.size();
    for (unsigned int i = 0; i < ct_size; ++i)
    {
        if (it_ct->size() >= minsWrkSize)
        {
            deepWrkConts.push_back(thickCts[i]);
        }
        if (it_ct->size() >= longWrkThresh /*&& it_c->size() <= sizeMax*/)
        {
            longWrkConts.push_back(thickCts[i]);
        }
        it_ct++;
    }
}

////////////////////////////////////////////////////////////////////////////////////////

void CcFrgiMap(const Mat& imgGray, int scaleRatio, Mat& frgiMap8U)
{
    Mat ehGrImg; // eh: enhanced
    PreprocGrImg(imgGray, ehGrImg);
        
    Mat frgiMapRz8U;
    ApplyFrgiFilter(ehGrImg, scaleRatio, frgiMapRz8U);
    
#ifdef TEST_RUN2
    string frgiMapFile =  wrkOutDir + "/frgiMap.png";
    imwrite(frgiMapFile, frgiMapRz8U);
#endif
    
    //把响应强度图又扩大到原始影像的尺度上来，但限定在Face Rect内。
    resize(frgiMapRz8U, frgiMap8U, imgGray.size());
}

void CcFrgiMapInFR(const Mat& imgGray,
                    const Rect& faceRect,
                    int scaleRatio,
                    Mat& frgiMap8U)
{
    Mat ehGrImg; // eh: enhanced
    PreprocGrImg(imgGray, ehGrImg);
    
    Mat ehGrImgFR = ehGrImg(faceRect);
    
    Mat frgiMapFRRz8U;
    ApplyFrgiFilter(ehGrImgFR, scaleRatio, frgiMapFRRz8U);
    
#ifdef TEST_RUN2
    string frgiRespImgFile =  wrkOutDir + "/frgiFR.png";
    imwrite(frgiRespImgFile, frgiMapFRRz8U);
#endif
    
    //把响应强度图又扩大到原始影像的尺度上来，但限定在Face Rect内。
    Mat frgiMapSSInFR8U;
    resize(frgiMapFRRz8U, frgiMapSSInFR8U, faceRect.size());
    Mat frgiMapGS(imgGray.size(), CV_8UC1, Scalar(0));
    
    frgiMapSSInFR8U.copyTo(frgiMapGS(faceRect));
    frgiMap8U = frgiMapGS;
}

void ApplyFrgiFilter(const Mat& inGrImg,
                     int scaleRatio,
                     Mat& frgiRespRzU8)
{
    Size rzSize = inGrImg.size() / scaleRatio;
    Mat rzImg;
    resize(inGrImg, rzImg, rzSize);
    
    Mat rzFtImg; // Ft: float
    rzImg.convertTo(rzFtImg, CV_32FC1);
    rzImg.release();
    
    cv::Mat respScaleRz, respAngRz;
    frangi2d_opts opts;
    opts.sigma_start = 1;
    opts.sigma_end = 5;
    opts.sigma_step = 2;
    opts.BetaOne = 0.5;  // BetaOne: suppression of blob-like structures.
    opts.BetaTwo = 12.0; // background suppression. (See Frangi1998...)
    opts.BlackWhite = true;
    
    // !!! 计算fangi2d时，使用的是缩小版的衍生影像
    Mat frgiRespRz;
    frangi2d(rzFtImg, frgiRespRz, respScaleRz, respAngRz, opts);
    rzFtImg.release();
    
    //返回的scaleRz, anglesRz没有派上实际的用场
    respScaleRz.release();
    respAngRz.release();
        
    frgiRespRzU8 = CvtFtImgTo8U_MinMax(frgiRespRz);
}
