//
//  Utils.cpp

/*******************************************************************************
Author: Fu Xiaoqiang
Date:   2022/9/11

********************************************************************************/

#include "Utils.hpp"
#include <array>
#include <limits>


// return a string that present a float with 2 decimal digits.
// for example, return "3.14" for 3.1415927
string convertFloatToStr2DeciDigits(float value)
{
    std::stringstream stream;
    stream << std::fixed << std::setprecision(2) << value;
    std::string out_str = stream.str();
    return out_str;
}

string openCVType2str(int type)
{
    string r;

    uchar depth = type & CV_MAT_DEPTH_MASK;
    uchar chans = 1 + (type >> CV_CN_SHIFT);

    switch ( depth )
    {
        case CV_8U:  r = "8U"; break;
        case CV_8S:  r = "8S"; break;
        case CV_16U: r = "16U"; break;
        case CV_16S: r = "16S"; break;
        case CV_32S: r = "32S"; break;
        case CV_32F: r = "32F"; break;
        case CV_64F: r = "64F"; break;
        default:     r = "User"; break;
    }

    r += "C";
    r += (chans+'0');

    return r;
}

/*
Mat BlendImages(const Mat& src1, const Mat& src2, float alpha)
{
    Mat dst = Mat::zeros(src1.size(), CV_8UC3);
    float beta = 1.0 - alpha;
    //float gama = 0.0;
    for (int i=0; i<src1.rows; i++)
    {
        const uchar* src1_ptr = src1.ptr<uchar>(i);
        const uchar* src2_ptr = src2.ptr<uchar>(i);
        uchar* dst_ptr  = dst.ptr<uchar>(i);
        for (int j=0; j<src1.cols*3; j++)
        {
            dst_ptr[j] = saturate_cast<uchar>(src1_ptr[j]*alpha + src2_ptr[j]*beta);
        }
    }
    
    return dst;
}
*/

/**********************************************************************************************
RC: rows and cols
***********************************************************************************************/
void PadImgWithRC4Div(Mat& srcImg) //, Mat& outImg)
{
    int padR = srcImg.cols % 4;
    int padB = srcImg.rows % 4;
    
    if(padR == 0 && padB == 0)
        return;
    
    Scalar blackColor(0, 0, 0);
    
    copyMakeBorder( srcImg, srcImg,
                    0, padB, //
                    0, padR,
                   BORDER_CONSTANT, blackColor);
}

//-------------------------------------------------------------------------------------------

/******************************************************************************************
将192*192的缩小版的影像“喂”给TF Lite网络的输入端，图像采用BGR通道次序。
这个函数为Face Detect和Face Mesh推理时所共享。
https://github.com/bferrarini/FloppyNet_TRO/blob/master/FloppyNet_TRO/TRO_pretrained/RPI4/src/lce_cnn.cc
see the function: ProcessInputWithFloatModel()
对像素值Normalization，使之变为Float，取值范围为[0.0 1.0]。
imgDataPtr已经是缩小版的输入影像了。
 *******************************************************************************************/
void FeedInputWithNormalizedImage(uint8_t* imgDataPtr, float* netInputBuffer, int H, int W, int C)
{
    for (int y = 0; y < H; ++y)
    {
        for (int x = 0; x < W; ++x)
        {
            if (C == 3)
            {
                // here is TRICK: BGR ==> RGB, be careful with the channel indices coupling
                // happened on the two sides of assignment operator.
                // another thing to be noted: make the value of each channel of the pixels
                // in [0.0 1.0], dividing by 255.0
                int yWC = y * W * C;
                int xC = x * C;
                int yWC_xC = yWC + xC;
                *(netInputBuffer + yWC_xC + 0) = *(imgDataPtr + yWC_xC + 2) / 255.0f;
                *(netInputBuffer + yWC_xC + 1) = *(imgDataPtr + yWC_xC + 1) / 255.0f;
                *(netInputBuffer + yWC_xC + 2) = *(imgDataPtr + yWC_xC + 0) / 255.0f;
            }
            else
            {
                // only one channel, gray scale image
                {
                    *(netInputBuffer + y * W + x ) = *(imgDataPtr + y * W + x) / 255.0f;
                }
            }
        }
    }
}

//-------------------------------------------------------------------------------------------
// convert a uint8 to a float in the range [-1.0 1.0]
float Quantize(uint8_t gray_value)
{
    float result = (gray_value - 128.0f) / 128.0f;
    return result;
}

/**********************************************************************************************
将缩小版(大小为H*W)的影像“喂”给TF Lite网络的输入端，图像采用BGR通道次序。
同时，在“喂”之前，对像素值Quantization，使之变为Float，取值范围为[-1.0 1.0]。
imgDataPtr已经是缩小版的输入影像了。
***********************************************************************************************/
void FeedInWithQuanImage(uint8_t* imgDataPtr, float* netInputBuffer, int H, int W, int C)
{
    for (int y = 0; y < H; ++y)
    {
        for (int x = 0; x < W; ++x)
        {
            if (C == 3)
            {
                // here is TRICK: BGR ==> RGB, be careful with the channel indices coupling
                // happened on the two sides of assignment operator.
                // another thing to be noted: make the value of each channel of the pixels
                // in [0.0 1.0], dividing by 255.0
                int yWC = y * W * C;
                int xC = x * C;
                int yWC_xC = yWC + xC;
                *(netInputBuffer + yWC_xC + 0) = Quantize(imgDataPtr[yWC_xC + 2]);
                *(netInputBuffer + yWC_xC + 1) = Quantize(imgDataPtr[yWC_xC + 1]);
                *(netInputBuffer + yWC_xC + 2) = Quantize(imgDataPtr[yWC_xC + 0]);
            }
            else
            {
                // only one channel, gray scale image
                {
                    *(netInputBuffer + y * W + x ) = Quantize(imgDataPtr[y * W + x]);
                }
            }
        }
    }
}

void FeedPadImgToNet(const cv::Mat& resizedPadImg, float* inTensorBuf)
{
    Mat rpImgRGB; //rp: resized and padded
    cv::cvtColor(resizedPadImg, rpImgRGB, cv::COLOR_BGR2RGB);

    /*
        std::vector<int> inputShape = getInputShape(idx);
        int H = inputShape[1];
        int W = inputShape[2];

        cv::Size wantedSize = cv::Size(W, H);
        cv::resize(out, out, wantedSize);
    */
    // Equivalent to (out - mean)/ std
    Mat quanImg;
    rpImgRGB.convertTo(quanImg, CV_32FC3, 1.0 / 127.5, -1.0);
    uint8_t* inImgMem = quanImg.ptr<uint8_t>(0);
    int sqSize = resizedPadImg.rows * resizedPadImg.cols;
    long totalBytes = sqSize*3*sizeof(float32_t);
    memcpy(inTensorBuf, inImgMem, totalBytes);
}

//-------------------------------------------------------------------------------------------

/**********************************************************************************************
采用Padding的方式，将原始图像扩充为一个正方形。
上下左右都扩展。
先扩充上下，newH = srcH * alpha, alpha取[0.2 0.4]
依据newH，再扩展左右，使之成为一个正方形
正方形的边长等于newH，原始像素矩阵在正方形中居中。
填充的像素取黑色。
***********************************************************************************************/
void MakeSquareImageV2(const Mat& srcImg, float deltaHRatio, Mat& squareImg)
{
    int srcW = srcImg.cols;
    int srcH = srcImg.rows;
    
    assert(srcW % 2 == 0); // must be a even number
    assert(srcH % 2 == 0); // must be a even number

    if(srcW >= srcH) // this case will not happened in our task
        squareImg = srcImg.clone();
    else // srcW < srcH
    {
        Scalar blackColor(0, 0, 0);
        
        int newH = (int)(srcH*(1 + deltaHRatio));
        if(newH % 2 == 1) // is odd number, should be converted to a even number
            newH += 1;
        
        int padVertW = (newH - srcH) / 2;  // padding at top and bottom
        int padSideW = (newH - srcW) / 2;
        
        copyMakeBorder( srcImg, squareImg,
                        padVertW, padVertW, //
                        padSideW, padSideW,
                       BORDER_CONSTANT, blackColor);
    }
}
//-------------------------------------------------------------------------------------------

/**********************************************************************************************
采用Shifting & Padding的方式，将原始图像进行几何方面的修正，使之适合于输入Face Mesh With Attention模型提取关键点。
基本的要求：
1. Face CP should be located in the center of the newly forged image.
2. 25% margin should be remained at both sides of the output image. 25% for each side.
3. In the output image, the ratio of Height vs Widht should be 1.4.
4. the raw content of source image should be contained completely in the out image.
5. scale not changed.
FV: front view
***********************************************************************************************/
void GeoFixFVSrcImg(const Mat& srcImg, const Rect& faceBBox,
                    const Point2i& faceCP, float alpha, Mat& outImg,
                    int& TP, int& LP)
{
    //TP, BP, LP, RP stand for the padding for top, bottom, left, and right side.

    int H = srcImg.rows;
    int W = srcImg.cols;
    
    int BW = faceBBox.width;
    int expandHalfW = (int)(BW * (1+alpha) / 2);
    
    int t1 = max(faceCP.x, srcImg.cols - faceCP.x);
    int half_Wp = max(expandHalfW, t1); // p: prime，右上侧的撇号；Wp: width of the out image
    
    int Wp = half_Wp * 2;
    Wp = max(Wp, srcImg.cols);

    int Hp;
    int BP;
    // 根据9月28日晚间的推导，目前将Hp与Wp脱离关系
    if(2*faceCP.y <= H)  // faceCp.y <= 1/2*H
    {
        // padding at top side
        Hp = 2*(H - faceCP.y);
        TP = H - 2*faceCP.y;
        BP = 0;
    }
    else
    {
        // padding at bottom side
        Hp = 2*faceCP.y;
        TP = 0;
        BP = 2*faceCP.y - H;
    }
        
    assert(Hp > srcImg.rows);
    
    if( Wp % 2 != 0)
        Wp += 1;
    
    int RP = Wp / 2 + faceCP.x - W;
    LP = Wp / 2 - faceCP.x;

    Scalar blackColor(0, 0, 0);

    copyMakeBorder( srcImg, outImg,
                    TP, BP, LP, RP,
                   BORDER_CONSTANT, blackColor);
}

//-------------------------------------------------------------------------------------------

string BuildOutImgFileName(const fs::path& outDir,
                         const string& fileNameBone,
                         const string& outPrefix)
{
    string outImgFile = outPrefix + fileNameBone + ".png";
    fs::path outImgFullPath = outDir / outImgFile;
    string outFileName_FP = outImgFullPath.string();
    
    return outFileName_FP;
}


// File Bone Name: no path and no extension
// the bone name of "images/JPN/cross_2.jpg" is "cross_2"
string GetFileBoneName(string fileNameFP)
{
    fs::path fp(fileNameFP);
    string basicFileName = fp.filename().string();
    //cout << "Basic File Name: " << basicFileName << endl;
    
    size_t lastindex = basicFileName.find_last_of(".");
    string fileBoneName = basicFileName.substr(0, lastindex);
    return fileBoneName;
}

//-------------------------------------------------------------------------------------------
