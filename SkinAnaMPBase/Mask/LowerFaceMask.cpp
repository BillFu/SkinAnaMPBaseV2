//
//  LowerFaceMask.cpp

/*******************************************************************************
Author: Fu Xiaoqiang
Date:   2022/10/3

********************************************************************************/

#include "LowerFaceMask.hpp"
#include "../FaceBgSeg/FaceBgSeg.hpp"
#include "../Utils.hpp"


/**********************************************************************************************
利用分割的结果来计算眼睛水平线以下的脸部轮廓和Mask，
嘴部、胡子、部分眼睛在集成时再去剔除。
***********************************************************************************************/
void ForgeLowerFaceMask(const FaceSegResult& segResult,
                        const Mat& fbBiLab, Mat& outMask)
{
    // NOTE: be careful with there are two coordinate system!
    /*
    // 1. binary the seg labels image into two classes:
    // background and face(including its sub-component)
    
    cv::Mat labelsBi;
    cv::threshold(segResult.segLabels, labelsBi, SEG_BG_LABEL, 255, cv::THRESH_BINARY);
    
    // 2. set the upper portion of binaried segLables to zeroes
    int cpY = segResult.faceCP.y;  // in source image space
    int srcImgH = segResult.srcImgS.height;
    int cpY_net = convSrcY2SegNetY(srcImgH, cpY); // now in seg net space
    
    Rect upperBox(0, 0, SEG_NET_OUTPUT_SIZE, cpY_net);
    
    //imwrite("labelsBi.png", labelsBi);

    resize(labelsBi, outMask, segResult.srcImgS, INTER_NEAREST);
    */
    int cpY = segResult.faceCP.y;  // in source image space
    //int srcImgH = segResult.srcImgS.height;
    
    Rect upperBox(0, 0, segResult.srcImgS.width, cpY);
    outMask = fbBiLab.clone();
    outMask(upperBox) = Scalar(SEG_BG_LABEL);

    // finally, outMask should be shrinked a bit
    int eroSize = segResult.faceBBox.width * 0.02;
    int eroDm = 2*eroSize + 1; // Dm : diameter
    Mat element = getStructuringElement(MORPH_ELLIPSE,
                           Size(eroDm, eroDm),
                           Point(eroSize, eroSize) );
    erode(outMask, outMask, element);
    
    /*
    int ksize = segResult.faceBBox.width * 0.01; //1
    if(ksize % 2 == 0)
        ksize += 1;
    medianBlur(outMask, outMask, ksize);
     */
}

//-------------------------------------------------------------------------------------------
