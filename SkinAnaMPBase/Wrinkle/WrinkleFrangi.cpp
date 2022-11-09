#include <algorithm>

#include "WrinkleFrangi.h"
#include "../ImgProc.h"
#include "../Utils.hpp"
#include "frangi.h"


// 计算Frangi滤波响应，并提取深皱纹和长皱纹
void CalcFrgiRespAndPickWrk(const Mat& grFrImg, // gray and cropped by face rect
                         const Mat& wrkMaskInFR,
                         int scaleRatio,
                         int minWrkSize,
                         int longWrkThresh,
                         Mat& frgiRespRz,  // Rz: resized, i.e., scale down
                         CONTOURS& deepWrkConts,
                         CONTOURS& longWrkConts,
                         float& avgFrgiRespValue)
{
    int wFR = grFrImg.cols;
    int hFR = grFrImg.rows;
    
    Size rzSize = grFrImg.size() / scaleRatio;
    Mat grFrRzImg;
    resize(grFrImg, grFrRzImg, rzSize);
    
    Mat grFrRzFlImg;
    grFrRzImg.convertTo(grFrRzFlImg, CV_32FC1);
    grFrRzImg.release();
    
    cv::Mat respScaleRz, respAngRz;
    frangi2d_opts opts;
    opts.sigma_start = 1;
    opts.sigma_end = 3;
    opts.sigma_step = 1;
    opts.BetaOne = 0.5;  // BetaOne: suppression of blob-like structures.
    opts.BetaTwo = 10.0; // background suppression. (See Frangi1998...)
    opts.BlackWhite = true;
    
    // !!! 计算fangi2d时，使用的是缩小版的衍生影像
    frangi2d(grFrRzFlImg, frgiRespRz, respScaleRz, respAngRz, opts);
    grFrRzFlImg.release();
    
    //返回的scaleRz, anglesRz没有派上实际的用场
    respScaleRz.release();
    respAngRz.release();
    
    frgiRespRz.convertTo(frgiRespRz, CV_8UC1, 255.);
    
#ifdef TEST_RUN
    string frgiRespImgFile = BuildOutImgFNV2(outDir, "frgiFrRz.png");
    bool isOK = imwrite(frgiRespImgFile, frgiRespRz);
    assert(isOK);
#endif

    cv::Mat frgiRespSSInFR; // SS: source scale
    //把响应强度图又扩大到原始影像的尺度上来，但限定在Face Rect内。
    resize(frgiRespRz, frgiRespSSInFR, cv::Size(wFR, hFR));
    frgiRespRz.release();

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
    
    Size rzSize = imgOfFh.size() / scaleRatio;
    Mat fhRzImg;
    resize(imgOfFh, fhRzImg, rzSize);
    
    Mat fhRzFlImg;
    fhRzImg.convertTo(fhRzFlImg, CV_32FC1);
    fhRzImg.release();
    
    float maxV = *max_element(fhRzFlImg.begin<float>(), fhRzFlImg.end<float>());
    float minV = *min_element(fhRzFlImg.begin<float>(), fhRzFlImg.end<float>());
    
    fhRzFlImg = (fhRzFlImg - minV) / (maxV - minV); // 调整到[0.0, 1.0]
    
    cv::Mat respScaleRz, respAngRz;
    frangi2d_opts opts;
    opts.sigma_start = 1;
    opts.sigma_end = 5;
    opts.sigma_step = 2;
    opts.BetaOne = 0.5;  // BetaOne: suppression of blob-like structures.
    opts.BetaTwo = 15.0; // background suppression. (See Frangi1998...)
    opts.BlackWhite = true;
    
    // !!! 计算fangi2d时，使用的是缩小版的衍生影像
    frangi2d(fhRzFlImg, frgiRespRz, respScaleRz, respAngRz, opts);
    fhRzFlImg.release();
    
    //返回的scaleRz, anglesRz没有派上实际的用场
    respScaleRz.release();
    respAngRz.release();
        
    Mat respMap8U = CvtFloatImgTo8UImg(frgiRespRz);
    
#ifdef TEST_RUN2
    string frgiRespImgFile = BuildOutImgFNV2(wrkOutDir, "fhFrgiResp.png");
    bool isOK = imwrite(frgiRespImgFile, respMap8U);
    assert(isOK);
#endif

}

void CalcSobelRespInFhReg(const Mat& grSrcImg,
                         const Rect& fhRect,
                         int scaleRatio,
                         Mat& frgiRespRz)
{
    Mat imgOfFh = grSrcImg(fhRect);
    
    GaussianBlur( imgOfFh, imgOfFh, Size(3,3), 0, 0, BORDER_DEFAULT );
    
    Mat grad_y;
    //Sobel(imgOfFh, grad_y, CV_8U, 0, 1, 3, 1, 0, BORDER_DEFAULT);

    Scharr(imgOfFh, grad_y, CV_16S, 0, 1);
    convertScaleAbs(grad_y, grad_y, 2.0, 20.0);
    
    //grad_y = ~grad_y;
#ifdef TEST_RUN2
    string gradImgFile = BuildOutImgFNV2(outDir, "gradYInFh.png");
    bool isOK = imwrite(gradImgFile, grad_y);
    assert(isOK);
#endif

}
