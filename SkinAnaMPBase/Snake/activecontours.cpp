#include "activecontours.h"
#include "../Geometry.hpp"

#ifdef ACTIVE_CONTOUR_ALG

namespace cvalg {

ActiveContours::ActiveContours()
{
    _w = 0; _h = 0;
    _points.clear();
    _beta = CONTOUR_BETA;
    _alpha = CONTOUR_ALPHA;
    _gamma = CONTOUR_GAMMA;
    previously_reset = false;

#ifdef USE_ANGLE_INSERTION
    _lastAngle = clock();
#endif

#ifdef USE_AVERAGE_LENGTH_BISECTION
    _lastBisect = clock();
#endif
}

vector<Point> ActiveContours::getOptimizedCont()
{
    return _points;
}

void ActiveContours::setParams(AlgoParams* params)
{
    _params = params;
}

void ActiveContours::init(int iw, int ih)
{
    _w = iw;
    _h = ih;
    _center = cv::Point(_w/2, _h/2);
    _points.reserve(_w*_h);
    previously_reset = false;  
}

void ActiveContours::insertPoint(Point p)
{
    _points.push_back(p);
}

void cvalg::ActiveContours::optimize(const Mat& inImg, const Mat& cornerField,
                                     int viewRadius,
                                     int iterTimes)
{
    // Take the current frame, and do sobel edge detection, threshold = 120
    // any contour with an intensity < 120 won't come back
    //  - Inside the pack is frame (Mat), contours (vector<Point>)
    //      angleAvailable (bool) to indicate if angle is calcd
    //      angles (vector<float>) giving the angles of the contours
    sobelPack sobelEdges = FullSobel(inImg,
                                     _params->getSobelThreash(),
                                     _params->getSobelAngle(),
                                     _params->getSobelDeadSpace());
    
    // For each snake point
    for(int k=0; k<iterTimes; k++)
    {
        for(int i = 0; i < static_cast<int>(_points.size()); i++)
        {
            // Define the neighborhood
            // Neighborhood size is square of viewRadius*2+1 at most
            int startx, endx, starty, endy;
            (_points[i].x - viewRadius < 0) ? startx = 0 : startx = _points[i].x - viewRadius;
            (_points[i].x + viewRadius > _w - 1) ? endx = _w - 1 : endx = _points[i].x + viewRadius;
            (_points[i].y - viewRadius < 0) ? starty = 0 : starty = _points[i].y - viewRadius;
            (_points[i].y + viewRadius > _h - 1) ? endy = _h - 1 : endy = _points[i].y + viewRadius;
            
            Point s(startx, starty);
            Point e(endx, endy);
            
            _points[i] = updatePos(i, s, e, sobelEdges.frame, cornerField);
        }
    }
}

bool ActiveContours::minimumRunReqSet()
{
    if(_points.size() >= MINIMUM_POINTS)
        return true;
    return false;
}

// pointIndex: 当前被扫描点在轮廓序列中的索引
Point ActiveContours::updatePos(int ptIndex, Point start, Point end,
                                const Mat& edgeImage, const Mat& cornerField)
{
    int cols = end.x - start.x;
    int rows = end.y - start.y;

    int numPt = static_cast<int>(_points.size());
    // Location of point in center of neighborhood
    Point location = _points[ptIndex];

    float localMin = 999999;

    // Update the average dist
    AvgPointDist();

    Rect ROI(start, end);
    Mat edgeSubImg = edgeImage(ROI);
    int ngmax = (int)(*max_element(edgeSubImg.begin<uchar>(), edgeSubImg.end<uchar>()));
    int ngmin = (int)(*min_element(edgeSubImg.begin<uchar>(), edgeSubImg.end<uchar>()));

    bool flag = true;
    // 逐个遍历当前点的邻域
    for(int y = 0; y < rows-1; y ++)
    {
        for(int x = 0; x < cols-1; x++)
        {
            /* E = ∫(α(s)Econt + β(s)Ecurv + γ(s)Eimage)ds */
            // X,Y Location in image
            Point2i parentPt = start + Point2i(x, y);

            /*  Econt
                (δ- (x[i] - x[i-1]) + (y[i] - y[i-1]))^2
                δ = avg dist between snake points */
            Point prevPt;  // 在轮廓线上当前点的前一个被扫描点
            if(ptIndex == 0)
                prevPt = _points[_points.size()-1];  // 环形
            else
                prevPt = _points[ptIndex-1];
            
            // Calculate Econt
            float dis = DisBetw2Pts(parentPt, prevPt);
            float econt = std::pow(_avgDist - dis, 2);

            // Multiply by alpha
            econt *= _params->getAlpha();

            // Ecurv = (x[i-1] - 2x[i] + x[i+1])^2 + (y[i-1] - 2y[i] + y[i+1])^2
            Point nextPt;
            if(ptIndex == numPt-1)
                nextPt = _points[0];
            else
                nextPt = _points[ptIndex+1];

            //float ecurv = std::pow( (prevPt.x - (parentPt.x*2) + nextPt.x), 2);
            //ecurv += std::pow( (prevPt.y - (parentPt.y*2) + nextPt.y), 2);
            Point2i delta = prevPt + nextPt - parentPt*2;
            float ecurv = LenOfVector(delta);
            ecurv *= _params->getBeta();

            /*  Eimage
                -||∇||
                Gradient magnitude encoded in pixel information
                    - May want to change this 'feature' */
            float eimage = -(int)edgeImage.at<uchar>(parentPt.y, parentPt.x);
            eimage *= _params->getGama();

            // Normalize
            econt /= ngmax;
            ecurv /= ngmax;

            int divisor = ngmax-ngmin;
            if (divisor <= 0)
                divisor = 1;
            eimage = (eimage-ngmin) / divisor;

            float energy = econt + ecurv + eimage;

            if(flag)
            {
                flag = false;
                localMin = energy;
                location = parentPt;
            }
            else if (energy < localMin)
            {
                localMin = energy;
                location = parentPt;
            }
        }
    }

    // Return The (new) location
    return location;
}

void ActiveContours::AvgPointDist()
{
    float sum = 0.0;
    int numPt = static_cast<int>(_points.size());
    
    for(int i = 0; i <numPt-1; i++)
    {
        float dis = DisBetw2Pts(_points[i], _points[i+1]);
        sum += dis;
    }
    
    float dis1 = DisBetw2Pts(_points[numPt-1], _points[0]);
    sum += dis1;
    _avgDist = sum / _points.size();
}

}
#endif // ACTIVE_CONTOUR_ALG

