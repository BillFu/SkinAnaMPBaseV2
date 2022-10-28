#include "activecontours.h"

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
    /*
    if(static_cast<int>(_points.size()) >= _params->getNumberPoints())
        return;
    
    std::vector<Point>::iterator it = std::find (_points.begin(), _points.end(), p);
    
    if(it != _points.end())
        return;
    */
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

Point ActiveContours::updatePos(int pointIndex, Point start, Point end,
                                const Mat& edgeImage, const Mat& cornerField)
{
    int cols = end.x - start.x;
    int rows = end.y - start.y;

    // Location of point in center of neighborhood
    Point location = _points[pointIndex];

    bool flag = true;
    float localMin;

    // Update the average dist
    averagePointDistance();

    int ngmax = -99999;
    int ngmin = 99999;
    for(int y = 0; y < rows; y ++)
        for(int x = 0; x < cols; x++)
        {
            int cval = (int)edgeImage.at<uchar>(y + start.y,x + start.x);
            if(flag)
            {
                flag = false;
                 ngmax = cval;
                 ngmin = cval;
            }
            else if (cval> ngmax)
            {
                ngmax = cval;
            }
            else if (cval < ngmin)
            {
                ngmin = cval;
            }
        }

    if(ngmax == 0)
        ngmax = 1;
    if(ngmin == 0)
         ngmin = 1;

    flag = true;
    for(int y = 0; y < rows-1; y ++)
    {
        for(int x = 0; x < cols-1; x++)
        {

            /*
                E = ∫(α(s)Econt + β(s)Ecurv + γ(s)Eimage)ds
            */
            // X,Y Location in image
            int parentX = x + start.x;
            int parentY = y + start.y;

            /*  Econt
                (δ- (x[i] - x[i-1]) + (y[i] - y[i-1]))^2
                δ = avg dist between snake points
            */
            Point prevPoint;
            if(pointIndex == 0)
                prevPoint = _points[_points.size()-1];
            else
                prevPoint = _points[pointIndex-1];
            // Calculate Econt
            float econt = (parentX - prevPoint.x) + (parentY - prevPoint.y);
            econt = std::pow(econt, 2);

            econt = _avgDist - econt;

            // Multiply by alpha
            econt *= _params->getAlpha();

            /*  Ecurv
                (x[i-1] - 2x[i] + x[i+1])^2 + (y[i-1] - 2y[i] + y[i+1])^2
            */
            Point nextPoint;
            if(pointIndex == static_cast<int>(_points.size()-1))
                nextPoint = _points[0];
            else
                nextPoint = _points[pointIndex+1];

            float ecurv = std::pow( (prevPoint.x - (parentX*2) + nextPoint.x), 2);
            ecurv += std::pow( (prevPoint.y - (parentY*2) + nextPoint.y), 2);
            ecurv *= _params->getBeta();

            /*  Eimage
                -||∇||

                Gradient magnitude encoded in pixel information
                    - May want to change this 'feature'
            */
            float eimage = -(int)edgeImage.at<uchar>(parentY,parentX);
            //float eimage = (int)image.at<uchar>(parentY,parentX);
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
                location = Point(parentX, parentY);
            }
            else if (energy < localMin)
            {
                localMin = energy;
                location = Point(parentX, parentY);
            }
        }
    }

    // Return The (new) location
    return location;
}

void ActiveContours::averagePointDistance()
{
    float sum = 0.0;
    for(int i = 0; i < static_cast<int>(_points.size()-1); i++)
    {
        sum += std::sqrt(
        ((_points[i].x - _points[i+1].x)*(_points[i].x - _points[i+1].x))+
        ((_points[i].y - _points[i+1].y)*(_points[i].y - _points[i+1].y)));
    }
    sum += std::sqrt(
    ((_points[_points.size()-1].x - _points[0].x)*
            (_points[_points.size()-1].x - _points[0].x))+
    ((_points[_points.size()-1].y - _points[0].y)*
            (_points[_points.size()-1].y - _points[0].y)));
    _avgDist = sum / _points.size();
}

}
#endif // ACTIVE_CONTOUR_ALG

