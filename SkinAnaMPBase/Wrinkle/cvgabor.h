#ifndef CVGABOR_H
#define CVGABOR_H
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

class CvGabor
{
public:
	CvGabor();
	~CvGabor();

	CvGabor(int iMu, int iNu);
	CvGabor(int iMu, int iNu, double dSigma);
	CvGabor(int iMu, int iNu, double dSigma, double dF);
	CvGabor(double dPhi, int iNu);
	CvGabor(double dPhi, int iNu, double dSigma);
	CvGabor(double dPhi, int iNu, double dSigma, double dF);

	void Init(int iMu, int iNu, double dSigma, double dF);
	void Init(double dPhi, int iNu, double dSigma, double dF);

	bool IsInit();
	bool IsKernelCreate();
	int mask_width();
	int get_mask_width();

	void get_image(int Type, cv::Mat& image);
	void get_matrix(int Type, cv::Mat& matrix);

	void conv_img(cv::Mat& src, cv::Mat& dst, int Type);

protected:
	double Sigma;
	double F;
	double Kmax;
	double K;
	double Phi;
	bool bInitialised;
	bool bKernel;
	int Width;
	cv::Mat Imag;
	cv::Mat Real;

private:
	void creat_kernel();
};

#endif
