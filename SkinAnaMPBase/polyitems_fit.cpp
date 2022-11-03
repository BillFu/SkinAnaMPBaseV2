
// OpenCV（C++）的曲线拟合polyfit
// https://blog.csdn.net/jpc20144055069/article/details/103232641

#include <iostream>
#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;

#include "polyitems_fit.hpp"

Mat polyfit(const vector<Point2f>& chain, int n)
{
	Mat y(chain.size(), 1, CV_32F, Scalar::all(0));
	/* ********【预声明phy超定矩阵】************************/
	/* 多项式拟合的函数为多项幂函数
	* f(x)=a0+a1*x+a2*x^2+a3*x^3+......+an*x^n
	*a0、a1、a2......an是幂系数，也是拟合所求的未知量。设有m个抽样点，则：
	* 超定矩阵phy=1 x1 x1^2 ... ...  x1^n
	*           1 x2 x2^2 ... ...  x2^n
	*           1 x3 x3^2 ... ...  x3^n
	*              ... ... ... ...
	*              ... ... ... ...
	*           1 xm xm^2 ... ...  xm^n
	*
	* *************************************************/
	cv::Mat phy(chain.size(), n, CV_32F, Scalar::all(0));

	for (int i = 0; i<phy.rows; i++)
	{
		float* pr = phy.ptr<float>(i);
		for (int j = 0; j<phy.cols; j++)
		{
			pr[j] = pow(chain[i].x, j);
		}
		y.at<float>(i) = chain[i].y;
	}

	Mat phy_t = phy.t();
	Mat phyMULphy_t = phy.t()*phy;
	Mat phyMphyInv = phyMULphy_t.inv();
	Mat a = phyMphyInv*phy_t;
	a = a*y;

	return a;
}


Mat polyfit_int(const vector<Point2i>& chain, int n)
{
    Mat y(chain.size(), 1, CV_32F, Scalar::all(0));
    /* ********【预声明phy超定矩阵】************************/
    /* 多项式拟合的函数为多项幂函数
    * f(x)=a0+a1*x+a2*x^2+a3*x^3+......+an*x^n
    *a0、a1、a2......an是幂系数，也是拟合所求的未知量。设有m个抽样点，则：
    * 超定矩阵phy=1 x1 x1^2 ... ...  x1^n
    *           1 x2 x2^2 ... ...  x2^n
    *           1 x3 x3^2 ... ...  x3^n
    *              ... ... ... ...
    *              ... ... ... ...
    *           1 xm xm^2 ... ...  xm^n
    *
    * *************************************************/
    cv::Mat phy(chain.size(), n, CV_32F, Scalar::all(0));

    for (int i = 0; i<phy.rows; i++)
    {
        float* pr = phy.ptr<float>(i);
        for (int j = 0; j<phy.cols; j++)
        {
            pr[j] = pow(chain[i].x, j);
        }
        y.at<float>(i) = chain[i].y;
    }

    Mat phy_t = phy.t();
    Mat phyMULphy_t = phy.t()*phy;
    Mat phyMphyInv = phyMULphy_t.inv();
    Mat a = phyMphyInv*phy_t;
    a = a*y;

    return a;
}

int calFittedYOrder2(int x, float a0, float a1, float a2)
{
    int y = (int)(a2*x*x + a1*x + a0);
    return y;
}

int calFittedYOrder3(int x, float a0, float a1, float a2, float a3)
{
    int y = (int)(a3*x*x*x + a2*x*x + a1*x + a0);
    return y;
}

void SmoothCtByPIFitOrder2(const CONTOUR inCt, CONTOUR& outCt)
{
    Mat a = polyfit_int(inCt, 3);
    float a0 = a.at<float>(0);
    float a1 = a.at<float>(1);
    float a2 = a.at<float>(2);
    
    /*
    int addedNumPt = 15;
    int dx = 2;
    for(int i=-addedNumPt; i<0; i++)
    {
        int x = inCt[0].x + i*dx;
        int y = calFittedYOrder2(x,  a0, a1, a2);
        outCt.push_back(Point2i(x, y));
    }
    */
    
    for(int i=0; i<inCt.size(); i++)
    {
        int x = inCt[i].x;
        int y = calFittedYOrder2(x,  a0, a1, a2);
        outCt.push_back(Point2i(x, y));
    }
    
    /*
    int oriNumPt = static_cast<int>(inCt.size());
    for(int i=1; i<=addedNumPt; i++)
    {
        int x = inCt[oriNumPt-1].x + i*dx;
        int y = calFittedYOrder2(x,  a0, a1, a2);
        outCt.push_back(Point2i(x, y));
    }
    */
    
}

void SmoothCtByPIFitOrder2V2(const CONTOUR inCt, CONTOUR& outCt)
{
    CONTOUR xSeq, ySeq;
    for(int i=0; i<inCt.size(); i++)
    {
        xSeq.push_back(Point2i(i, inCt[i].x));
        ySeq.push_back(Point2i(i, inCt[i].y));
    }
    
    Mat a = polyfit_int(xSeq, 3);
    float a0 = a.at<float>(0);
    float a1 = a.at<float>(1);
    float a2 = a.at<float>(2);
    
    
    Mat b = polyfit_int(ySeq, 3);
    float b0 = b.at<float>(0);
    float b1 = b.at<float>(1);
    float b2 = b.at<float>(2);
    
    for(int i=0; i<inCt.size(); i++)
    {
        //int x = inCt[i].x;
        int x = calFittedYOrder2(i,  a0, a1, a2);
        int y = calFittedYOrder2(i,  b0, b1, b2);
        outCt.push_back(Point2i(x, y));
    }
}

void SmoothCtByPIFitOrder3(const CONTOUR inCt, CONTOUR& outCt)
{
    Mat a = polyfit_int(inCt, 4);
    float a0 = a.at<float>(0);
    float a1 = a.at<float>(1);
    float a2 = a.at<float>(2);
    float a3 = a.at<float>(3);

    for(int i=0; i<inCt.size(); i++)
    {
        int x = inCt[i].x;
        int y = calFittedYOrder3(x,  a0, a1, a2, a3);
        outCt.push_back(Point2i(x, y));
    }
}

/*
// 传入的是NOS的Local坐标，传出的是SS尺度下的全局坐标
void SmoothCtByPIFitV2(const CONTOUR inCtLocNOS, CONTOUR& outCt)
{
    Mat a = polyfit_int(inCtLocNOS, 3); // 传入3，实际计算的是2次多项式函数
    float a0 = a.at<float>(0);
    float a1 = a.at<float>(1);
    float a2 = a.at<float>(2);
    

    for(int i=0; i<inCt.size(); i++)
    {
        int x = inCt[i].x;
        int y = calFittedY(x,  a0, a1, a2);
        outCt.push_back(Point2i(x, y));
    }
    
}
*/
