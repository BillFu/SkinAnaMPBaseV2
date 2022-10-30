#include "activecontours.h"
#include "../Geometry.hpp"
#include "../Utils.hpp"
#include <numeric>

using namespace std;

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

void ActiveContours::setInitCont(const CONTOUR& initCont)
{
    copy(initCont.begin(), initCont.end(), back_inserter(_points));
}

void ShowMaxMinInRec(const vector<float>& rec,
    const string& title)
{
    float MaxV = *max_element(rec.begin(), rec.end());
    float MinV = *min_element(rec.begin(), rec.end());
    cout << "-----------" << title << "------------" << endl;
    cout << "max: " << MaxV << "," <<  "min: " << MinV << endl;
    
    float sumV = accumulate(rec.begin(), rec.end(), 0.0);
    float avgV = sumV / rec.size();
    cout << "avg: " << avgV << endl;
    cout << "-------------------------------------" << endl;
}

float AvgOfEComRec(const vector<float>& rec)
{
    float sumV = accumulate(rec.begin(), rec.end(), 0.0);
    float avgV = sumV / rec.size();
    
    return avgV;
}

void ActiveContours::ShowECompData(vector<float>& EcontRec,
                   vector<float>& EcurvRec,
                   vector<float>& EedgeRec,
                   vector<float>& EcorRec)
{
    float avgEcont = AvgOfEComRec(EcontRec);
    float avgEcurv = AvgOfEComRec(EcurvRec);
    float avgEedge = AvgOfEComRec(EedgeRec);
    float avgEcor = AvgOfEComRec(EcorRec);

    cout << "avgEcont: " << avgEcont << ","
         << "avgEcurv: " << avgEcurv << ","
         << "avgEedge: " << avgEedge << ","
         << "avgEcor: " << avgEcor << endl;
}


Point2i ActiveContours::GetPrevPt(int ptIndex)
{
    int numPt = static_cast<int>(_points.size());
    int prevIndex = (ptIndex-1 + numPt) % numPt;
    return _points[prevIndex];
}

Point2i ActiveContours::GetNextPt(int ptIndex)
{
    int nextIndex = (ptIndex+1) % _points.size();
    return _points[nextIndex];
}

/*  Econt: (δ- (x[i] - x[i-1]) + (y[i] - y[i-1]))^2
    δ = avg dist between snake points */
float ActiveContours::CalcEcont(const Point2i& prevPt, const Point2i& neibPt, float avgDist)
{
    // Calculate Econt
    float dis = DisBetw2Pts(neibPt, prevPt);
    float econt = std::pow(avgDist - dis, 2);
    
    return econt;
}

// Ecurv = (x[i-1] - 2x[i] + x[i+1])^2 + (y[i-1] - 2y[i] + y[i+1])^2
float ActiveContours::CalcEcurv(const Point2i& prevPt,
                const Point2i& neibPt, const Point2i& nextPt)
{
    Point2i delta = prevPt + nextPt - neibPt*2;
    float ecurv = LenOfVector(delta);
    return ecurv;
}

// 估算E的各分量的大致取值（是各点取最大值，还是取平均值？）

void ActiveContours::estiEnergeComponents(float& MaxEcont, float& MaxEcurv)
{
    AvgPointDist();

    vector<float> EcontRec, EcurvRec, EtotalRec;
    for(int i = 0; i < static_cast<int>(_points.size()); i++)
    {
        Point2i prevPt = GetPrevPt(i);
        Point2i nextPt = GetNextPt(i);
        Point2i curPt = _points[i];
        
        float Econt = CalcEcont(prevPt, curPt, _avgDist);
        float Ecurv = CalcEcurv(prevPt, curPt, nextPt);
        EcontRec.push_back(Econt);
        EcurvRec.push_back(Ecurv);
    }
    
    MaxEcont = *max_element(EcontRec.begin(), EcontRec.end());
    float sumEcont = accumulate(EcontRec.begin(), EcontRec.end(), 0.0);
    float avgEcont = sumEcont / EcontRec.size();
    cout << "maxEcont: " << MaxEcont << endl;
    cout << "avgEcont: " << avgEcont << endl;

    MaxEcurv = *max_element(EcurvRec.begin(), EcurvRec.end());
    float sumEcurve = accumulate(EcurvRec.begin(), EcurvRec.end(), 0.0);
    float avgEcurve = sumEcurve / EcurvRec.size();
    cout << "maxEcurv: " << MaxEcurv << endl;
    cout << "avgEcurve: " << avgEcurve << endl;
    
}

void ActiveContours::BuildNeigArea(int ptIndex, int viewRadius,
                                   Point2i& startPt, Point2i& endPt)
{
    // Define the neighborhood
    // Neighborhood size is square of viewRadius*2+1 at most
    int startx, endx, starty, endy;
    (_points[ptIndex].x - viewRadius < 0) ? startx = 0 : startx = _points[ptIndex].x - viewRadius;
    (_points[ptIndex].x + viewRadius > _w - 1) ? endx = _w - 1 : endx = _points[ptIndex].x + viewRadius;
    (_points[ptIndex].y - viewRadius < 0) ? starty = 0 : starty = _points[ptIndex].y - viewRadius;
    (_points[ptIndex].y + viewRadius > _h - 1) ? endy = _h - 1 : endy = _points[ptIndex].y + viewRadius;
    
    startPt = Point2i(startx, starty);
    endPt   = Point2i(endx, endy);
}

void ActiveContours::optimize(const Mat& inImg, const Mat& cornerField,
                                     int viewRadius,
                                     int iterTimes)
{
    string typeStr = openCVType2str(cornerField.type());
    cout << "type of cornerField: " << typeStr << endl;

    float MaxEcont, MaxEcurv;
    estiEnergeComponents(MaxEcont, MaxEcurv);

    // Take the current frame, and do sobel edge detection, threshold = 120
    // any contour with an intensity < 120 won't come back
    //  - Inside the pack is frame (Mat), contours (vector<Point>)
    //      angleAvailable (bool) to indicate if angle is calcd
    //      angles (vector<float>) giving the angles of the contours
    sobelPack sobelEdges = FullSobel(inImg,
                                     _params->getSobelThreash(),
                                     _params->getSobelAngle(),
                                     _params->getSobelDeadSpace());
    
    imwrite("sobel.png", sobelEdges.frame);
    
    // For each snake point
    for(int k=0; k<iterTimes; k++)
    {
        vector<float> EcontRec, EcurvRec, EedgeRec, EcorRec,EtotalRec;
        for(int i = 0; i < static_cast<int>(_points.size()); i++)
        {
            Point2i startPt, endPt;
            BuildNeigArea(i, viewRadius, startPt, endPt);
            _points[i] = updatePos(i, startPt, endPt, sobelEdges.frame, cornerField,
                                   MaxEcont, MaxEcurv,
                                   EcontRec, EcurvRec, EedgeRec, EcorRec,
                                   EtotalRec);
        }
        
        cout << "--------------------------------" << k << "-----------------------------------" << endl;
        // output the average value of all energy components in current epoch
        ShowECompData(EcontRec, EcurvRec, EedgeRec, EcorRec);

        Mat canvas = inImg.clone();
        CONTOURS curCts;
        curCts.push_back(_points);
        drawContours(canvas, curCts, 0, Scalar(150), 2);
        string outFile = "rstAC_" + to_string(k) + ".png";
        imwrite(outFile, canvas);
    }
}

bool ActiveContours::minimumRunReqSet()
{
    if(_points.size() >= MINIMUM_POINTS)
        return true;
    return false;
}

// pointIndex: 当前被扫描点在轮廓序列中的索引
/*
 Econt为正值，越小表示轮廓线越短，形状越紧致，越成团状。
 Ecurv为正值，越小表示曲率越小，越光滑；越大表示形状越复杂，振荡越严重。
 Eedge被人为设置为负值，越小表示此处为边缘的概率大，越接近0，表示越为内部的灰度均匀点。
 Eedge也变化为正值，越小表示此处为边缘的概率大；越大，表示越为内部的灰度均匀点。
 Ecor正值，但被反相，越大表示越不可能为角点；越小表示越接近角点。
 */
Point ActiveContours::updatePos(int ptIndex, Point start, Point end,
                                const Mat& edgeImage, const Mat& cornerField,
                                float MaxEcont, float MaxEcurv,
                                vector<float>& EcontRec,
                                vector<float>& EcurvRec,
                                vector<float>& EedgeRec,
                                vector<float>& EcorRec,
                                vector<float>& EtotalRec)
{
    int cols = end.x - start.x;
    int rows = end.y - start.y;

    //int numPt = static_cast<int>(_points.size());
   
    // Update the average dist
    AvgPointDist();

    Rect ROI(start, end);
    /*
    Mat edgeSubImg = edgeImage(ROI);
    int edgeMax = (int)(*max_element(edgeSubImg.begin<uchar>(), edgeSubImg.end<uchar>()));
    int edgeMin = (int)(*min_element(edgeSubImg.begin<uchar>(), edgeSubImg.end<uchar>()));
    
    if(edgeMax == 0)
        edgeMax = 1; // edgeMax will act as denominator later
    if(edgeMax == edgeMin)
        edgeMax++;  // avoid that: edgeMax - edgeMin == 0
    */

    // 在轮廓线上当前点的前一个被扫描点
    Point prevPt = GetPrevPt(ptIndex);
    Point nextPt = GetNextPt(ptIndex);
    
    // Location of point in center of neighborhood
    Point minLoc = _points[ptIndex]; // minLoc also be the new destionation
    float minEnerge = 999999;
    // 逐个遍历当前点的邻域
    for(int y = 0; y < rows-1; y ++)
    {
        for(int x = 0; x < cols-1; x++)
        {
            /* E = ∫(α(s)Econt + β(s)Ecurv + γ(s)Eimage)ds */
            // X,Y Location in image
            Point2i curNeibPt = start + Point2i(x, y);
            float Econt = CalcEcont(prevPt, curNeibPt, _avgDist);
            Econt /= MaxEcont;
            EcontRec.push_back(Econt);

            // Multiply by alpha
            Econt *= _params->getAlpha();
            
            float Ecurv = CalcEcurv(prevPt, curNeibPt, nextPt);
            Ecurv /= MaxEcurv;
            EcurvRec.push_back(Ecurv);
            Ecurv *= _params->getBeta();

            /*  Eimage: -||∇||
                Gradient magnitude encoded in pixel information
                    - May want to change this 'feature' */
            float Eedge = (float)edgeImage.at<uchar>(curNeibPt);
            // divisor never be zero for specical processing has been taken ahead.
            //int divisor = edgeMax - edgeMin;
            //Eedge = (Eedge - edgeMin) / divisor; // make it in [0, 1] after normalization
            //then add minus sign before it
            Eedge = 1.0 - Eedge/255.0; // make it in the interval: [0, 1]
            EedgeRec.push_back(Eedge);
            Eedge *= _params->getGama();
            
            float Ecor = cornerField.at<float>(curNeibPt); // already lies in [0.0, 1.0]
            EcorRec.push_back(Ecor);
            Ecor *= _params->_lambda;

            float Energy = Econt + Ecurv + Eedge + Ecor;
            EtotalRec.push_back(Energy);
            if (Energy < minEnerge)
            {
                minEnerge = Energy;
                minLoc = curNeibPt;
            }
        }
    }

    // Return The (new) location
    return minLoc;
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

