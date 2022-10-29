//
//  GaussField.cpp

/*******************************************************************************
Author: Fu Xiaoqiang
Date:   2022/10/28

********************************************************************************/

#include "GaussField.hpp"
#include <algorithm>
#include "../Utils.hpp"

using namespace std;
/*
// srcImg is a gray image
void DetectCorner_Harris(const Mat& srcImg)
{
    int blockSize = 4; //2;
    int apertureSize = 7; //3;
    double k = 0.04;
    
    Mat respMap = Mat::zeros(srcImg.size(), CV_32FC1);
    cornerHarris(srcImg, respMap, blockSize, apertureSize, k );
    
    Mat respNorm, respNormScaled;
    normalize( respMap, respNorm, 0, 255, NORM_MINMAX, CV_32FC1, Mat() );
    convertScaleAbs(respNorm, respNormScaled);

    int thresh = 200;
    for( int i = 0; i < respNorm.rows ; i++ )
    {
        for( int j = 0; j < respNorm.cols; j++ )
        {
            if((int) respNorm.at<float>(i,j) > thresh)
            {
                circle(respNormScaled, Point(j,i), 5,  Scalar(0), 2, 8, 0 );
            }
        }
    }
}
*/

/*
     image：输入灰度图像，float32类型
     maxCorners：返回角点的最大数目，值为0表表示没有设置最大值限制，返回所有检测到的角点。
     qualityLevel：质量系数（小于1.0的正数，一般在0.01-0.1之间），
        表示可接受角点的最低质量水平。该系数乘以最好的角点分数（也就是上面较小的那个特征值），
        作为可接受的最小分数；例如，如果最好的角点分数值为1500且质量系数为0.01，那么所有质量分数小于15的角都将被忽略。
     minDistance：角之间最小欧式距离，忽略小于此距离的点。
     corners：输出角点坐标
     mask：可选的感兴趣区域，指定想要检测角点的区域。
     blockSize：默认为3，角点检测的邻域大小（窗口尺寸）
     useHarrisDetector：用于指定角点检测的方法，如果是true则使用Harris角点检测，false则使用Shi Tomasi算法。默认为False。
     k：默认为0.04，Harris角点检测时使用。
     */
/*
void DetectCorner_ST(const Mat& srcImg) // shi-tomasi
{
    int maxCorners = 7;
    vector<Point2f> corners;
    
    double qualityLevel = 0.05;
    double minDistance = 30;
    int blockSize = 11, gradientSize = 3;
    bool useHarris = false;
    double k = 0.04;
    
    
    goodFeaturesToTrack( srcImg,
                         corners,
                         maxCorners,
                         qualityLevel,
                         minDistance,
                         Mat(),
                         blockSize,
                         gradientSize,
                         useHarris,
                         k );
    cout << "** Number of corners detected: " << corners.size() << endl;
    int radius = 4;
    Mat copy = srcImg.clone();
    for( size_t i = 0; i < corners.size(); i++ )
    {
        circle(copy, corners[i], radius,
               Scalar(150), FILLED );
    }
    
    imwrite("cornerEyeMask.png", copy);
}
*/

cv::Mat getGaussKernel(int rows, int cols, double sigmax, double sigmay)
{
    const auto y_mid = (rows-1) / 2.0;
    const auto x_mid = (cols-1) / 2.0;

    const auto x_spread = 1. / (sigmax*sigmax*2);
    const auto y_spread = 1. / (sigmay*sigmay*2);

    const auto denominator = 8 * std::atan(1) * sigmax * sigmay;

    std::vector<double> gauss_x, gauss_y;

    gauss_x.reserve(cols);
    for (auto i = 0;  i < cols;  ++i)
    {
        auto x = i - x_mid;
        gauss_x.push_back(std::exp(-x*x * x_spread));
    }

    gauss_y.reserve(rows);
    for (auto i = 0;  i < rows;  ++i)
    {
        auto y = i - y_mid;
        gauss_y.push_back(std::exp(-y*y * y_spread));
    }

    cv::Mat kernel = cv::Mat::zeros(rows, cols, CV_32FC1);
    for (auto j = 0;  j < rows;  ++j)
        for (auto i = 0;  i < cols;  ++i)
        {
            kernel.at<float>(j,i) = gauss_x[i] * gauss_y[j] / denominator;
        }

    return kernel;
}

Mat BuildGaussField(int fieldW, int fieldH, int sigma,
                     const vector<Point2i>& peaks)
{
    Rect fieldRect(0, 0, fieldW, fieldH);
    
    // 注意：构造一个Mat时，排在前面的是rows，后面才是cols，换言之，先高后宽！！！
    Mat field(fieldH, fieldW, CV_32FC1, Scalar(0.0));
    
    int kerRadius = 3*sigma;
    int kerDiameter = kerRadius*2 + 1; // must be a odd number
    
    cv::Mat kernel = getGaussKernel(kerDiameter, kerDiameter, sigma, sigma);
    string kernel_DataType = openCVType2str(kernel.type());
    cout << "kernel_DataType: " << kernel_DataType << endl;

    for(Point2i pt: peaks)
    {
        Point2i tl(pt.x-kerRadius, pt.y-kerRadius);
        Point2i br(pt.x+kerRadius, pt.y+kerRadius);
        Rect kerRect(tl.x, tl.y, kerDiameter, kerDiameter);
        Rect validKerRect = kerRect & fieldRect;
        Mat clippedKer = kernel(Rect(0, 0, validKerRect.width, validKerRect.height));
        clippedKer.copyTo(field(validKerRect));
    }
    
    float maxV = *max_element(field.begin<float>(), field.end<float>());
    //float minV = *min_element(field.begin<float>(), field.end<float>());

    Mat scaleImg = 1.0 - field / maxV;
    //imwrite("field.png", scaleImg);
    //Mat ucharField;
    //scaleImg.assignTo(ucharField, CV_8UC1);
    
    //ucharField = ~ucharField; // inversed, more smaller (positive) when near the peaks
    return scaleImg;
}
