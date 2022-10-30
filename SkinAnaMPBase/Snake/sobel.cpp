#include "sobel.h"

/*
 GX             GY
    1, 0,-1,        1, 2, 1,
    2, 0,-2,        0, 0, 0,
    1, 0,-1        -1,-2,-1
*/

sobelPack FullSobel(Mat gray, int thresh, bool calcAngle, bool deadSpace)
{
    int reservedSpace = (gray.cols*gray.rows)/2;

    // Get the data to return ready
    sobelPack data;
    //gray.copyTo(data.frame);
    data.frame = Mat(gray.size(), CV_8UC1, Scalar(0));
    data.contours.reserve(reservedSpace);

    if(calcAngle)
    {
        data.angleAvailable = true;
        data.angles.reserve(reservedSpace);
    }
    else
    {
        data.angleAvailable = false;
    }

    for(int x = 1; x < gray.rows-2; x++)
    {
        for(int y = 1; y < gray.cols-2; y++)
        {
            // Bottom
            short b0 = (short)gray.at<uchar>(x-1,y-1);
            short b1 = (short)gray.at<uchar>(x,y-1);
            short b2 = (short)gray.at<uchar>(x+1,y-1);

            // Middle
            short m0 = (short)gray.at<uchar>(x-1,y);
            // Middle value (x,y) always multd by 0
            short m2 = (short)gray.at<uchar>(x+1,y);

            // Top
            short t0 = (short)gray.at<uchar>(x-1,y+1);
            short t1 = (short)gray.at<uchar>(x,y+1);
            short t2 = (short)gray.at<uchar>(x+1,y+1);

            // Reduced from matrix operations
            int px = b0 + (-b2) + (2 * m0) + (-2 * m2) + t0 + (-t2);
            int py = b0 + (2 * b1) + b2 + (-t0) + (-2 * t1) + (-t2);

            int value = static_cast<int>(std::ceil(std::sqrt((px*px) + (py*py))));
            if(value > 255)
                value = 255;

            if(value >= thresh)
            {
                data.contours.push_back(Point(x,y));
                data.frame.at<uchar>(x,y) = value;
            }
            
        } // End Y
    }// End X
    return data;
}
