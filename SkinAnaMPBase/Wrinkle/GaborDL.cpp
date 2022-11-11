//
//  GaborDL.cpp

/*******************************************************************************
本模块是Gabor滤波的实现。
 https://github.com/dominiklessel/opencv-gabor-filter
 https://medium.com/@anuj_shah/through-the-eyes-of-gabor-filter-17d1fdb3ac97

Author: Fu Xiaoqiang
Date:   2022/11/11

********************************************************************************/

#include "GaborDL.hpp"

/*
Mat BuildGaborKernel(int kerSize, double sig,
                     double thetaDeg, double lm, double psiDeg)
{
    int hks = (kerSize-1)/2;
    double theta = thetaDeg*CV_PI/180;
    double psi = psiDeg*CV_PI/180;
    double del = 2.0/(kerSize-1);
    double lmbd = lm;
    double sigma = sig/kerSize;
    double x_theta;
    double y_theta;
    
    cv::Mat kernel(kerSize,kerSize, CV_32F);
    
    for (int y=-hks; y<=hks; y++)
    {
        for (int x=-hks; x<=hks; x++)
        {
            x_theta = x*del*cos(theta)+y*del*sin(theta);
            y_theta = -x*del*sin(theta)+y*del*cos(theta);
            kernel.at<float>(hks+y,hks+x) = (float)exp(-0.5*(pow(x_theta,2)+pow(y_theta,2))/pow(sigma,2))* cos(2*CV_PI*x_theta/lmbd + psi);
        }
    }
    return kernel;
}
*/

// AR: aspect ratio
Mat BuildGabKerAR(int kerSize, double gamma, double sigma,
                  double thetaDeg, double lambda, double psiDeg)
{
    if(kerSize % 2 == 0)
        kerSize += 1;
    
    int hks = (kerSize - 1)/2; // h: half

    double theta = thetaDeg * CV_PI/180;
    double psi = psiDeg * CV_PI/180;
    double del = 2.0/(kerSize-1);
    double x_theta;
    double y_theta;
    
    cv::Mat kernel(kerSize, kerSize, CV_32F);
    
    for (int y=-hks; y<=hks; y++)
    {
        for (int x=-hks; x<=hks; x++)
        {
            x_theta = x*del*cos(theta)+y*del*sin(theta);
            y_theta = -x*del*sin(theta)+y*del*cos(theta);
            x_theta *= gamma;
            
            kernel.at<float>(hks+y,hks+x) = (float)exp(-0.5*(pow(x_theta,2)+pow(y_theta,2))/pow(sigma,2))* cos(2*CV_PI*x_theta/lambda + psi);
        }
    }
    return kernel;
}

void doGaborFilter(const Mat& inGrFtImg, const GaborOpt& opt, Mat& gaborMap)
{
    double gam = opt.gamma / 100.0;
    double sig = opt.sigma / (double)opt.kerSize;
    double lm = 0.5 + opt.lambda/100.0;
    double th = opt.thetaDeg;
    double ps = opt.psiDeg;
    
    Mat kernel = BuildGabKerAR(opt.kerSize, gam, sig, th, lm, ps);
    filter2D(inGrFtImg, gaborMap, CV_32F, kernel);
}


