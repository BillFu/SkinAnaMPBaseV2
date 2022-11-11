#include <algorithm>

#include "WrinkleFrangi.h"
#include "../ImgProc.h"
#include "../Utils.hpp"
#include "frangi.h"


// 从Frangi滤波响应中提取深皱纹和长皱纹
void PickWrkFromFrgiMap( const Mat& wrkMaskInFR,
                         int minWrkSize,
                         int longWrkThresh,
                         Mat& frgiRespSSInFR,  // Rz: resized, i.e., scale down
                         CONTOURS& deepWrkConts,
                         CONTOURS& longWrkConts,
                         float& avgFrgiRespValue)
{
    // DL: deep and long
    PickDLWrkFromFrgiResp(frgiRespSSInFR,
                          wrkMaskInFR, // 原始尺度，经过了Face_Rect裁切
                          longWrkThresh, minWrkSize,
                          longWrkConts, deepWrkConts);

    cv::Scalar sumResp = cv::sum(frgiRespSSInFR);
    int nonZero2 = cv::countNonZero(wrkMaskInFR); // Mask中有效面积，即非零元素的数目
    avgFrgiRespValue = sumResp[0] / (nonZero2+1) / 12.8;  // 平均响应值，不知道为啥要除以12.8
}

////////////////////////////////////////////////////////////////////////////////////////
/// 从frangi滤波的结果（经过了二值化、细化、反模糊化等处理）中，提取深皱纹、长皱纹
void PickDLWrkFromFrgiResp(const Mat& frgiRespOS, //Original Scale
                            const Mat& wrkMaskInFR, // 原始尺度，经过了Face_Rect裁切
                            int longWrkThresh,
                                         unsigned int minsWrkSize,
                                         CONTOURS& longWrkConts,
                                         CONTOURS& deepWrkConts)
{
    cv::Mat thickBi; // thick: 浓的，厚的，粗的
    cv::threshold(frgiRespOS, thickBi, 20, 255, cv::THRESH_BINARY);
    
#ifdef TEST_RUN
    string fraThRespFile =  outDir + "/FrgiBiResp.png";
    imwrite(fraThRespFile.c_str(), thickBi);
#endif
    
    thickBi = 255 - thickBi;
    int wdFR = wrkMaskInFR.cols;
    int htFR = wrkMaskInFR.rows;
    // 给二值图像中的粗黑线“瘦身”
    BlackLineThinInBiImg(thickBi.data, wdFR, htFR);
    
#ifdef TEST_RUN
    string frgiThinRespFile =  outDir + "/FrgiThinResp.png";
    imwrite(frgiThinRespFile.c_str(), thickBi);
#endif
    
    thickBi = 255 - thickBi;
    removeBurrs(thickBi, thickBi);
    thickBi = thickBi & wrkMaskInFR;
    
#ifdef TEST_RUN
    string frgiFinalRespFile =  outDir + "/FrgFinalResp.png";
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

void CalcFrgiRespInFhReg(const Mat& grSrcImg,
                         const Rect& fhRect,
                         int scaleRatio,
                         Mat& frgiRespRz)
{
    Mat imgOfFh = grSrcImg(fhRect);
    Size fhImgS = imgOfFh.size();
    
    // 这个公式仅对前额区域有效；若imgW表示图像全域或其他子区域，这个公式需要调整。
    // 也许这个公式以后需要调整为普遍适用的公式。
    int blurKerS = fhImgS.width / 150; // 1/142 约等于9/1286
    if(blurKerS % 2 == 0)
        blurKerS += 1;  // make it be a odd number
    if(blurKerS < 3)
        blurKerS = 3;
    cout << "blurKerS: " << blurKerS << endl;
    
    blur(imgOfFh, imgOfFh, Size(blurKerS, blurKerS));
    
    Mat clachRst;
    // 这个公式仅对前额区域有效；若imgW表示图像全域或其他子区域，这个公式需要调整。
    // 也许这个公式以后需要调整为普遍适用的公式。
    int gridSize = fhImgS.width / 54; // 1/54与24/1286有关
    cout << "gridSize: " << gridSize << endl;
    ApplyCLAHE(imgOfFh, gridSize, clachRst);
    imgOfFh.release();
    
    Mat frgiRespRz8U;
    ApplyFrgiFilter(clachRst, scaleRatio, frgiRespRz8U);
    clachRst.release();
        
#ifdef TEST_RUN2
    string frgiRespImgFile = BuildOutImgFNV2(wrkOutDir, "fhFrgiResp.png");
    bool isOK2 = imwrite(frgiRespImgFile, frgiRespRz8U);
    assert(isOK2);
#endif

}

void CcFrgiRespInFR(const Mat& ehGrImg,
                    const Rect& faceRect,
                    int scaleRatio,
                    Mat& frgiMapSSInFR)
{
    Mat ehGrImgFR = ehGrImg(faceRect);
    
    Mat frgiMapFRRz8U;
    ApplyFrgiFilter(ehGrImgFR, scaleRatio, frgiMapFRRz8U);
    
#ifdef TEST_RUN2
    string frgiRespImgFile =  wrkOutDir + "/frgiFR.png";
    imwrite(frgiRespImgFile, frgiMapFRRz8U);
#endif
    
    //把响应强度图又扩大到原始影像的尺度上来，但限定在Face Rect内。
    resize(frgiMapFRRz8U, frgiMapSSInFR, faceRect.size());
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
