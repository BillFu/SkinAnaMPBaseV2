//
//  Wrinkle.cpp

/*******************************************************************************
本模块负责检测皱纹和相关处理。

Author: Fu Xiaoqiang
Date:   2022/11/1

********************************************************************************/

//#include "Geometry.hpp"
#include "Wrinkle.hpp"
#include "cvgabor.h"
#include "frangi.h"
#include "../Utils.hpp"

//-------------------------------------------------------------------------------------------
// Frangi + Gabor Filtering
void DetectWrinkle(const Mat& inImg, const Rect& faceRect)
{
    // In the preprocessing stage, Color ==> Gray should be performed firstly
    cv::Mat imgGray;
    cvtColor(inImg, imgGray, COLOR_BGR2GRAY);

    // crop input gray image by Face Bounding Box, i.e., faceRect
    Mat grFrImg; // the copy of input image cropped by FaceRect
    imgGray(faceRect).copyTo(grFrImg);
    imgGray.release();
    
    int scaleRatio = 4;
    Mat grFrRzImg; //resized
    Size rzSize = grFrImg.size() / scaleRatio;
    resize(grFrImg, grFrRzImg, rzSize);
    grFrImg.release();
        
    Mat grFrRzFlImg;
    grFrRzImg.convertTo(grFrRzFlImg, CV_32FC1);
    grFrRzImg.release();
    //--------preprocessin for frangi filtering is done--------------------------------------
    
    cv::Mat frgiRespRz, scaleRz, respAngRz;
    frangi2d_opts opts;
    frangi2d_createopts(opts);
    
    // !!! 计算fangi2d时，使用的是缩小版的衍生影像
    frangi2d(grFrRzFlImg, frgiRespRz, scaleRz, respAngRz, opts); // !!!
    grFrRzFlImg.release();
    //返回的scaleRz, anglesRz没有派上实际的用场
    scaleRz.release();
    respAngRz.release();
    
    frgiRespRz.convertTo(frgiRespRz, CV_8UC1, 255.);
    
    cv::Mat frgiRespOSFR; // orignal sacle, cropped by Face Rect
    //把响应强度图又扩大到原始影像的尺度上来，但限定在工作区内。
    resize(frgiRespRz, frgiRespOSFR, faceRect.size());
    
#ifdef TEST_RUN
    string frgiRespImgFile = BuildOutImgFNV2(outDir, "frgiFrRz.png");
    bool isOK = imwrite(frgiRespImgFile, frgiRespRz);
    assert(isOK);
#endif
    
    frgiRespRz.release();
    
    int temp0 = 0.042*faceRect.width; // ???
    int longWrkTh = max(temp0, 38);
    
#ifdef TEST_RUN
    cout << "temp0: " << temp0 << endl;
    cout << "longWrkTh: " << longWrkTh << endl;
#endif
    
    CONTOURS longContours;
    unsigned int sizeMin = 38;
    //unsigned int sizeMax = 400;
    
    /*
    get_deep_long_wrink_from_frangiResp(wrinkRespOrigScale,
                                       _wrinkleMask_ROI, // 原始尺度，经过了Face_Rect裁切
                                       longWrinkThresh,
                                       sizeMin,
                                       longContours,
                                       DeepWrinkleContours);
    */
#ifdef TEST_RUN
    
#endif
}
