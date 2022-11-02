#include "activecontours.h"
#include "../Geometry.hpp"
#include "../Tie.hpp"
#include "../Utils.hpp"
#include <numeric>

using namespace std;

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
    _avgDist = AvgPointDist(_points);

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

void ActiveContours::UpdateAmpCoeff(const Energy& energy, AmpCoeff& ampCoeff)
{
    ampCoeff.alpha = 1.5*energy.Eunwtol / (energy.Econt + 0.0001);
    ampCoeff.beta = energy.Eunwtol / (energy.Ecurv + 0.0001);
    ampCoeff.gamma = 0.3* energy.Eunwtol / (energy.Eedge + 0.0001);
    ampCoeff.lambda = energy.Eunwtol / (energy.Ecor + 0.0001);
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
    int sobelThresh = 120;
    sobelPack sobelEdges = FullSobel(inImg, sobelThresh);
    
    //imwrite("sobel.png", sobelEdges.frame);
    
    // For each snake point
    // every point has its own AmpCoeff and will be changed during the iterations
    int numPts = static_cast<int>(_points.size());
    vector<AmpCoeff> ampCoeffList;
    ampCoeffList.reserve(numPts);
    for(int i=0; i<numPts; i++)
    {
        AmpCoeff coeff;
        ampCoeffList.push_back(coeff);
    }

    for(int k=0; k<iterTimes; k++)
    {
        //----------start to debug--------------------------------------------
        Mat canvas = inImg.clone();
        //CONTOURS curCts;
        //curCts.push_back(_points);
        //drawContours(canvas, curCts, 0, Scalar(150), 2);
        for(int i = 0; i < _points.size(); i++)
        {
            cv::circle(canvas, _points[i], 1, Scalar(150), cv::FILLED);
        }
        string outFile = "rstAC_" + to_string(k) + ".png";
        imwrite(outFile, canvas);
        //---------end of debugging-------------------------------------------
        
        vector<Energy> EnergeRec;
        Energy destEnergy;
        for(int i = 0; i < static_cast<int>(_points.size()); i++)
        {
            Point2i startPt, endPt;
            BuildNeigArea(i, viewRadius, startPt, endPt);
            
            _points[i] = updatePosV2(i, startPt, endPt, sobelEdges.frame, cornerField,
                                     MaxEcont, MaxEcurv, ampCoeffList[i], destEnergy);
            EnergeRec.push_back(destEnergy);
            
            // update amplifying coefficient
            UpdateAmpCoeff(destEnergy, ampCoeffList[i]);
        }
        
        //cout << "--------------------------------" << k << "-----------------------------------" << endl;
        // output the average value of all energy components in current epoch
        //ShowECompData(EcontRec, EcurvRec, EedgeRec, EcorRec);
        
        /*
        TieGroup tieGroup;
        CheckTieOnContour(_points, 5, tieGroup);
        
        if(tieGroup.hasTie())
        {
            cout << "Num of Tie found: " << tieGroup.getTieNum() << endl;
            for(Tie tie: tieGroup.ties)
                cout << tie << endl;
        }
        */

        CONTOUR evenCont;  // resampled Points that locate evenly along with S
        MakePtsEvenWithS(_points, _points.size(), evenCont);
        //CONTOUR cleanCont;
        //DelTiesOnContV2(_points, tieGroup, cleanCont);

        //CONTOUR sparsedCont;
        //SparsePtsOnContour(cleanCont, 0.7, sparsedCont);
        _points = evenCont;
    }
}

bool ActiveContours::minimumRunReqSet()
{
    if(_points.size() >= MINIMUM_POINTS)
        return true;
    return false;
}

/*
 destUnwTolE: 未加权获得的总能量。
 destWTolE:  加权之后计算出的总能量。
 */
Point ActiveContours:: updatePosV2(int ptIndex, Point start, Point end,
                const Mat& edgeImage, const Mat& cornerField,
                float MaxEcont, float MaxEcurv,
                const AmpCoeff& ampCoeff, Energy& destEnerge)
{
    int cols = end.x - start.x;
    int rows = end.y - start.y;
   
    // Update the average dist
    _avgDist = AvgPointDist(_points);

    Rect ROI(start, end);
    
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
            
            float Ecurv = CalcEcurv(prevPt, curNeibPt, nextPt);
            Ecurv /= MaxEcurv;

            /*  Eimage: -||∇||
                Gradient magnitude encoded in pixel information
                    - May want to change this 'feature' */
            float Eedge = (float)edgeImage.at<uchar>(curNeibPt);
            //then add minus sign before it
            Eedge = 1.0 - Eedge/255.0; // make it in the interval: [0, 1]
            
            float Ecor = cornerField.at<float>(curNeibPt); // already lies in [0.0, 1.0]

            float UnwEnergy = Econt + Ecurv + Eedge + Ecor;
            float Energy = Econt* ampCoeff.alpha + Ecurv*ampCoeff.beta +
                Eedge*ampCoeff.gamma + Ecor*ampCoeff.lambda;
            if (Energy < minEnerge)
            {
                minEnerge = Energy;
                minLoc = curNeibPt;
                
                destEnerge.SetValues(Econt, Ecurv, Eedge, Ecor, UnwEnergy, Energy);
            }
        }
    }

    // Return The (new) location
    return minLoc;
}


}

