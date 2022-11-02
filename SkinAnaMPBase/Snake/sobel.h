#ifndef SOBEL_H
#define SOBEL_H

#include <iostream>

#include <math.h>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

// Faster if this isn't defined, nice to look at
#define SOBEL_FILL_IN_NEG_SPACE 1

// Faster if this isn't defined, but could be useful
#define SOBEL_CALCULATE_ANGLE 1

#define SOBEL_THRESH 120

using namespace cv;


struct sobelPack
{
    // Image returned
    Mat frame;

    // Points of interedt
    std::vector<Point> contours;
};

sobelPack FullSobel(const Mat& gray, int sobelThresh);

#endif // SOBEL_H
