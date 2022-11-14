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
void PickDLWrkInFrgiMapV2(int minWrkTh, int longWrkTh,
                      Mat& frgiMap8U,
                      CONTOURS& deepWrkConts,
                      CONTOURS& longWrkConts)
{
    // DL: deep and long
    cv::Mat WrkBi(frgiMap8U.size(), CV_8UC1, cv::Scalar(0));
    
    threshold(frgiMap8U, WrkBi, 40, 255, THRESH_BINARY);

    Mat thinBi = 255 - WrkBi;
    WrkBi.release();
    
    // 给二值图像中的粗黑线“瘦身”
    BlackLineThinInBiImg(thinBi.data,
                         frgiMap8U.cols, frgiMap8U.rows);
    
#ifdef TEST_RUN2
    string frgiThinFile = wrkOutDir + "/FrgiThin.png";
    imwrite(frgiThinFile.c_str(), thinBi);
#endif
    
    thinBi = 255 - thinBi;
    //removeBurrs(thinBi, thinBi);
    
#ifdef TEST_RUN2
    //string remBurFile = wrkOutDir + "/remBur.png";
    //imwrite(remBurFile.c_str(), thinBi);
#endif
    
    Mat elmt = getStructuringElement(MORPH_ELLIPSE, Size(15, 1));
    Mat dilBi, EroBi;
    dilate(thinBi, dilBi, elmt, Point2i(-1,-1), 1);
    
    //Mat elmt2 = getStructuringElement(MORPH_ELLIPSE, Size(1, 5));
    //erode(dilBi, EroBi, elmt2, Point2i(-1,-1), 1);
    //dilBi.release();
    
#ifdef TEST_RUN2
    //string frgiFinalRespFile = wrkOutDir + "/EroBi.png";
    //imwrite(frgiFinalRespFile.c_str(), EroBi);
#endif
    
    CONTOURS thickCts;
    findContours(dilBi, thickCts, RETR_EXTERNAL, CHAIN_APPROX_NONE);
    
    CONTOURS::const_iterator it_ct = thickCts.begin();
    unsigned long ct_size = thickCts.size();
    for (unsigned int i = 0; i < ct_size; ++i)
    {
        if (it_ct->size() >= minWrkTh)
        {
            deepWrkConts.push_back(thickCts[i]);
        }
        if (it_ct->size() >= longWrkTh)
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

/*
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
*/

Mat CcFrgiMapInRect(const Mat& imgGray,
                    const Rect& rect,
                    int scaleRatio)
{
    Mat ehGrImg; // eh: enhanced
    PreprocGrImg(imgGray, ehGrImg);
    
    Mat ehGrImgRt = ehGrImg(rect);
    
    Mat mapRtRz8U;
    ApplyFrgiFilter(ehGrImgRt, scaleRatio, mapRtRz8U);
    
    //把响应强度图又扩大到原始影像的尺度上来，但限定在Face Rect内。
    Mat mapSSInRt8U;
    resize(mapRtRz8U, mapSSInRt8U, rect.size());
    Mat frgiMapGS(imgGray.size(), CV_8UC1, Scalar(0));
    
    mapSSInRt8U.copyTo(frgiMapGS(rect));
    return frgiMapGS;
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
