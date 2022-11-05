#include "WrinkleFrangi.h"
#include "../ImgProc.h"
#include <algorithm>
#include "frangi.h"


// 计算Frangi滤波响应，并提取深皱纹和长皱纹
void CalcFrgRespAndExtWrk(const Mat& imgInFR,
                         const Mat& wrkMaskInFR,
                         int scaleRatio,
                         int minWrkSize,
                         int longWrkThresh,
                         Mat& frangiRespRz,  // Rz: resized, i.e., scale down
                         CONTOURS& deepWrkConts,
                         CONTOURS& longWrkConts,
                         float& avgFrgRespValue)
{
    int wFR = imgInFR.cols;
    int hFR = imgInFR.rows;
    
    cv::Mat inImgInFR_rz; // resized copy of inImgInFR
    resize(imgInFR, inImgInFR_rz,
           cv::Size(wFR/scaleRatio, hFR/scaleRatio)); //scaleRatio now is 5
    
    cv::Mat inImgFR_rz_gray;
    cvtColor(inImgInFR_rz, inImgFR_rz_gray, COLOR_BGR2GRAY);
    inImgInFR_rz.release();
    
    Mat inImgFR_rz_gray_fl;
    inImgFR_rz_gray.convertTo(inImgFR_rz_gray_fl, CV_32FC1);
    inImgFR_rz_gray.release();
    
    cv::Mat scaleRz, anglesRz;
    frangi2d_opts opts;
    opts.sigma_start = 1;
    opts.sigma_end = 3;
    opts.sigma_step = 1;
    opts.BetaOne = 0.5;  // BetaOne: suppression of blob-like structures.
    opts.BetaTwo = 10.0; // background suppression. (See Frangi1998...)
    opts.BlackWhite = true;
    
    // !!! 计算fangi2d时，使用的是缩小版的衍生影像
    frangi2d(inImgFR_rz_gray_fl, frangiRespRz, scaleRz, anglesRz, opts); // !!!
    
    inImgFR_rz_gray_fl.release();
    
    //返回的scaleRz, anglesRz没有派上实际的用场
    scaleRz.release();
    anglesRz.release();
    
    frangiRespRz.convertTo(frangiRespRz, CV_8UC1, 255.);
    
    cv::Mat frgRespSrcScale;
    //把响应强度图又扩大到原始影像的尺度上来，但限定在工作区内。
    resize(frangiRespRz, frgRespSrcScale, cv::Size(wFR, hFR));
    frangiRespRz.release();
    
#ifdef TEST_RUN_WRK
    string frgRespFile =  wrk_out_dir + "/FrangiResp.png";
    imwrite(frgRespFile.c_str(), frgRespSrcScale);
#endif
    
    getDeepLongWrkFromFrangiResp(frgRespSrcScale,
                                wrkMaskInFR, // 原始尺度，经过了Face_Rect裁切
                                longWrkThresh, minWrkSize,
                                longWrkConts, deepWrkConts);

    cv::Scalar sumResp = cv::sum(frgRespSrcScale);
    int nonZero2 = cv::countNonZero(wrkMaskInFR); // Mask中有效面积，即非零元素的数目
    avgFrgRespValue = sumResp[0] / (nonZero2+1) / 12.8;  // 平均响应值，不知道为啥要除以12.8
}

////////////////////////////////////////////////////////////////////////////////////////
/// 从frangi滤波的结果（经过了二值化、细化、反模糊化等处理）中，提取深皱纹、长皱纹
void getDeepLongWrkFromFrangiResp(const Mat& frangiRespOrigScale,
                                         const Mat& wrkMaskInFR, // 原始尺度，经过了Face_Rect裁切
                                         int longWrkThresh,
                                         unsigned int minsWrkSize,
                                         CONTOURS& longWrkConts,
                                         CONTOURS& DeepWrkConts)
{
    cv::Mat thickBinary;
    cv::threshold(frangiRespOrigScale, thickBinary, 40, 255, cv::THRESH_BINARY);
    
#ifdef TEST_RUN_WRK
    string fraThRespFile =  wrk_out_dir + "/FrgThreshResp.png";
    imwrite(fraThRespFile.c_str(), thickBinary);
#endif
    
    thickBinary = 255 - thickBinary;
    int wdFR = wrkMaskInFR.cols;
    int htFR = wrkMaskInFR.rows;
    imageThin(thickBinary.data, wdFR, htFR);
    
#ifdef TEST_RUN_WRK
    string fraThinRespFile =  wrk_out_dir + "/FrgThinResp.png";
    imwrite(fraThinRespFile.c_str(), thickBinary);
#endif
    
    thickBinary = 255 - thickBinary;
    removeBurrs(thickBinary, thickBinary);
    thickBinary = thickBinary & wrkMaskInFR;
    
#ifdef TEST_RUN_WRK
    string frgFinalRespFile =  wrk_out_dir + "/FrgFinalResp.png";
    imwrite(frgFinalRespFile.c_str(), thickBinary);
#endif
    
    CONTOURS contoursThick;
    findContours(thickBinary, contoursThick, RETR_EXTERNAL, CHAIN_APPROX_NONE);
    
    CONTOURS::const_iterator it_ct = contoursThick.begin();
    unsigned long ct_size = contoursThick.size();
    for (unsigned int i = 0; i < ct_size; ++i)
    {
        if (it_ct->size() >= minsWrkSize)
        {
            DeepWrkConts.push_back(contoursThick[i]);
        }
        if (it_ct->size() >= longWrkThresh /*&& it_c->size() <= sizeMax*/)
        {
            longWrkConts.push_back(contoursThick[i]);
        }
        it_ct++;
    }
}

////////////////////////////////////////////////////////////////////////////////////////


