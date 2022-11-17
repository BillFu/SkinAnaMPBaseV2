//
//  Utils.cpp

/*******************************************************************************
Author: Fu Xiaoqiang
Date:   2022/9/11

********************************************************************************/

#include "Utils.hpp"
#include <array>
#include <limits>
#include "Common.hpp"
#include <float.h>
#include <numeric>      // std::accumulate

#include "Geometry.hpp"

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

//-------------------------------------------------------------------------------------------

bool isInImg(const Point2i& pt, int cols, int rows)
{
    if(pt.x < cols && pt.x >=0
       && pt.y >=0 && pt.y < rows)
        return true;
    else
        return false;
}

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

    // Equivalent to (out - mean)/ std
    Mat quanImg;
    rpImgRGB.convertTo(quanImg, CV_32FC3, 1.0 / 127.5, -1.0);
    uint8_t* inImgMem = quanImg.ptr<uint8_t>(0);
    int sqSize = resizedPadImg.rows * resizedPadImg.cols;
    long totalBytes = sqSize*3*4;  // 4 == sizeof(float32_t);
    memcpy(inTensorBuf, inImgMem, totalBytes);
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

string BuildOutImgFNV2(const fs::path& outDir,
                         const string& fileName)
{
    fs::path outImgFullPath = outDir / fileName;
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

int convSegNetY2SrcY(int srcImgH, int segNetY)
{
    int srcY = segNetY * srcImgH / SEG_NET_OUTPUT_SIZE;
    return srcY;
}

int convSegNetX2SrcX(int srcImgW, int segNetX)
{
    int srcX = segNetX * srcImgW / SEG_NET_OUTPUT_SIZE;
    return srcX;
}

int convSrcY2SegNetY(int srcImgH, int srcY)
{
    int segNetY = srcY * SEG_NET_OUTPUT_SIZE / srcImgH;
    return segNetY;
}

int convSrcX2SegNetX(int srcImgW, int srcX)
{
    int segNetX = srcX * SEG_NET_OUTPUT_SIZE / srcImgW;
    return segNetX;
}


Point2i convSegNetPt2SrcPt(const Size& srcImgS, const Point2i& snPt)
{
    int srcY = snPt.y * srcImgS.height / SEG_NET_OUTPUT_SIZE;
    int srcX = snPt.x * srcImgS.width / SEG_NET_OUTPUT_SIZE;
    
    return Point2i(srcX, srcY);
}

Point2i convSrcPt2SegNetPt(const Size& srcImgS, const Point2i& srcPt)
{
    int segNetY = srcPt.y * SEG_NET_OUTPUT_SIZE / srcImgS.height;
    int segNetX = srcPt.x * SEG_NET_OUTPUT_SIZE / srcImgS.width;

    return Point2i(segNetX, segNetY);
}

/*
template<typename T>
T stddev(std::vector<T> const & func)
{
    T mean = std::accumulate(func.begin(), func.end(), 0.0) / func.size();
    T sq_sum = std::inner_product(func.begin(), func.end(), func.begin(), 0.0,
        [](T const & x, T const & y) { return x + y; },
        [mean](T const & x, T const & y) { return (x - mean)*(y - mean); });
    return std::sqrt(sq_sum / func.size());
}
*/

float stddev_float(vector<float> const & func)
{
    float mean = std::accumulate(func.begin(), func.end(), 0.0) / func.size();
    float sq_sum = std::inner_product(func.begin(), func.end(), func.begin(), 0.0,
        [](float const & x, float const & y) { return x + y; },
        [mean](float const & x, float const & y) { return (x - mean)*(y - mean); });
    return std::sqrt(sq_sum / func.size());
}

CONTOUR ComTwoArcs2Cont(const CONTOUR arc1, const CONTOUR arc2)
{
    CONTOUR mergeCt(arc1);
        
    // combine lower curve and upper curve into one complete contour
    for(int i=0; i<arc2.size(); i++)
    {
        mergeCt.push_back(arc2[i]);
    }

    return mergeCt;
}

//-------------------------------------------------------------------------------------------
// 计算某点相对于Face Rect左上角点的相对坐标
Point2i CalcRelCdToFR(const Point2i& pt, const Rect& faceRect)
{
    return Point2i(pt.x - faceRect.x, pt.y - faceRect.y);
}

// 用faceRect(宽高分别用frW、frH表示)去裁剪rect。rect既是输入，也是输出。
void ClipRectByFR(int frW, int frH, int margin, Rect& rect)
{
    rect.x = rect.x > 0 ? rect.x : margin;
    rect.y = rect.y > 0 ? rect.y : margin;
    rect.y = rect.y < frH ? rect.y : frH - margin;
    rect.x = rect.x < frW ? rect.x : frW - margin;
    rect.width = rect.br().x < frW ? rect.width : frW - rect.x;
    rect.height = rect.br().y < frH ? rect.height : frH - rect.y;
}

//-------------------------------------------------------------------------------------------
// !!!调用这个函数前，outMask必须进行过初始化，或者已有内容在里面！！！
void DrawContOnMask(const POLYGON& contours, Mat& outMask)
{
    cv::fillPoly(outMask, contours, cv::Scalar(255));
}

Mat TransMaskLS2GS(const Size& srcSize, const DetectRegion& localMask)
{
    Mat maskInGlobal(srcSize, CV_8UC1, Scalar(0));
    localMask.mask.copyTo(maskInGlobal(localMask.bbox));
    return maskInGlobal;
}

void TransMaskGS2LS(const Mat& maskGS, DetectRegion& localMask)
{
    Rect maskRect = boundingRect(maskGS);
    Mat cropMask;
    maskGS(maskRect).copyTo(cropMask);
    localMask = DetectRegion(maskRect, cropMask);
}

void TransPgGS2LSMask(const POLYGON& gsPg, DetectRegion& localMask)
{
    Rect pgRect = boundingRect(gsPg);
    
    CONTOUR lssPg;
    transCt_GS2LSS(gsPg, pgRect.tl(), lssPg);
    
    Mat lssMask(pgRect.size(), CV_8UC1, Scalar(0));
    DrawContOnMask(lssPg, lssMask);
    
    localMask = DetectRegion(pgRect, lssMask);
}


void SubstractDetReg(const Size& gsSize, const DetectRegion& reg1,
                     const DetectRegion& reg2, DetectRegion& outReg)
{
    Mat gsMask1 = TransMaskLS2GS(gsSize, reg1);
    Mat gsMask2 = TransMaskLS2GS(gsSize, reg2);

    Mat outMask = gsMask1 & (~gsMask2);
    
    TransMaskGS2LS(outMask, outReg);
}

void SumDetReg2GSMask(const Size& gsSize, const DetectRegion& reg1,
                     const DetectRegion& reg2, Mat& outMask)
{
    Mat gsMask1 = TransMaskLS2GS(gsSize, reg1);
    Mat gsMask2 = TransMaskLS2GS(gsSize, reg2);
    
    outMask = gsMask1 | gsMask2;
}

//-------------------------------------------------------------------------------
bool SaveTestOutImgInDir(const Mat& out_img,
                         const string& outDir,
                         const char* outFileName)
{
    string outImgFullPath = outDir + "//" + outFileName;
    bool isSucceeded = imwrite(outImgFullPath, out_img);
    return isSucceeded;
}
