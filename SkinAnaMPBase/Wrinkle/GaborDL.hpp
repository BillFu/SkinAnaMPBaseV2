//
//  GaborDL.hpp
//
//
/*******************************************************************************

本模块是Gabor滤波的实现。
 https://github.com/dominiklessel/opencv-gabor-filter
 https://medium.com/@anuj_shah/through-the-eyes-of-gabor-filter-17d1fdb3ac97
 
Author: Fu Xiaoqiang
Date:   2022/11/11
 
********************************************************************************/


#ifndef GABOR_DL_HPP
#define GABOR_DL_HPP

#include "opencv2/opencv.hpp"

using namespace std;
using namespace cv;

#include "../Common.hpp"

// 表征参数，外观参数，实际计算时要进行转换为实际可用的数值
struct GaborOpt
{
    int kerSize; 
    int sigma;
    int lambda;
    int thetaDeg;
    int psiDeg;
    
    GaborOpt(int kerSize0, int sigma0, int lambda0,
             int thetaDeg0, int psiDeg0):
        kerSize(kerSize0), sigma(sigma0), lambda(lambda0),
        thetaDeg(thetaDeg0), psiDeg(psiDeg0)
    {
        
    }
};

typedef vector<GaborOpt> GaborOptBank;

// Deg: in degrees
Mat BuildGaborKernel(int kerSize, double sig, double thetaDeg, double lm, double psiDeg);

void doGaborFilter(const Mat& inGrFtImg, const GaborOpt& opt, Mat& gaborMap);


#endif /* end of GABOR_DL_HPP */
