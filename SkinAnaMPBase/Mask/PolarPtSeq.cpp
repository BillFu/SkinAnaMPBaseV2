//
//  PolarPtSeq.cpp

/*******************************************************************************
Author: Fu Xiaoqiang
Date:   2022/10/12

********************************************************************************/

#include "PolarPtSeq.hpp"
//#include "../BSpline/ParametricBSpline.hpp"
//#include "FundamentalMask.hpp"
//#include "Geometry.hpp"
#include "Common.hpp"


/**********************************************************************************************
only cover the two eyes
***********************************************************************************************/
// extract the primary and coarse polar sequence on the contour
void CalcPolarSeqOnCt(const CONTOUR& spCt, const Point2i& eyeCP,
                      PolarContour& polarCont)
{
    polarCont.oriPt = eyeCP;
    for(Point pt: spCt)
    {
        Point diff = pt - eyeCP;
        double r = sqrt(diff.x * diff.x + diff.y * diff.y);
        double theta = atan2(diff.y, diff.x);
        PtInPolarCd ptPolar(r, theta);
        polarCont.ptSeq.push_back(ptPolar);
    }
}

// extract the primary and coarse polar sequence on the contour
// polePt: the origin of Pole Coordinate System, also called Pole Point.
// scanDir: CW or CCW when to scan in the natural order of curve
// the output values of theta will be in range: [0, 2*PI)
void CalcPolarSeqOnCurve(const CONTOUR& curve, const Point2i& polePt,
                         CLOCK_DIR scanDir,
                      PolarContour& polarSeq)
{
    if(scanDir == CLOCK_WISE)
    {
        CalcPolarSeqOnCurveCW(curve, polePt, polarSeq);
        
    } //CCLOCK_WISE
    else
    {
        CalcPolarSeqOnCurveCCW(curve, polePt, polarSeq);
    }
}

void CalcPolarSeqOnCurveCCW(const CONTOUR& curve, const Point2i& polePt,
                      PolarContour& polarSeq)
{
    polarSeq.oriPt = polePt;
    double preTheta = - M_PI;
    
    for(Point pt: curve)
    {
        Point diff = pt - polePt;
        double r = sqrt(diff.x * diff.x + diff.y * diff.y);
        double theta = atan2(diff.y, diff.x);
        if(theta < 0.0)
            theta += M_PI*2.0;
        
        if(theta > preTheta)
        {
            PtInPolarCd ptPolar(r, theta);
            polarSeq.ptSeq.push_back(ptPolar);
            
            preTheta = theta;
        }
    }
}

void CalcPolarSeqOnCurveCW(const CONTOUR& curve, const Point2i& polePt,
                      PolarContour& polarSeq)
{
    polarSeq.oriPt = polePt;
    double preTheta = 3.0 * M_PI;

    for(Point pt: curve)
    {
        Point diff = pt - polePt;
        double r = sqrt(diff.x * diff.x + diff.y * diff.y);
        double theta = atan2(diff.y, diff.x);
        if(theta < 0.0)
            theta += M_PI*2.0;
        
        if(theta < preTheta)
        {
            PtInPolarCd ptPolar(r, theta);
            polarSeq.ptSeq.push_back(ptPolar);
            
            preTheta = theta;
        }
    }
}

///////////////////////////////////////////////////////////////////////////////////////////////

PtInPolarSeq MakePolarSeqInCCWise(const PtInPolarSeq& oriSeq)
{
    int numPts = (int)(oriSeq.size());

    PtInPolarCd prevPt = oriSeq[numPts - 1];
    PtInPolarCd curPt = oriSeq[0];
    int curID = 0;
    int times_gt = 0; // times that current theta is greater than previous theta
    while(curID < oriSeq.size() - 1)
    {
        if(curPt.theta > prevPt.theta)
            times_gt++;
            
        prevPt = curPt;
        curID++;
        curPt = oriSeq[curID];
    }
    
    if(times_gt > numPts/2) // has been in CCW
        return oriSeq;
    else  // in CW, need turn the order
    {
        PtInPolarSeq ccwSeq;
        for(int i = numPts-1; i>=0; i--)
        {
            ccwSeq.push_back(oriSeq[i]);
        }
        
        return ccwSeq;
    }
}

// make the pt with angle near zero placed at the header of seq
void ReorderPolarPtSeq(const PtInPolarSeq& ccSeq,
                     PtInPolarSeq& zeroThetaFirstSeq)
{
    int numPts = (int)(ccSeq.size());
    
    int curID = 0;
    double Th_2PI = 1.8 * M_PI;
    while(curID <= ccSeq.size() - 1)
    {
        PtInPolarCd prevPt;
        PtInPolarCd curPt;
        if(curID == 0)
        {
            prevPt = ccSeq[numPts - 1];
        }
        else
            prevPt = ccSeq[curID-1];
        
        curPt = ccSeq[curID];
        
        if(abs(prevPt.theta - curPt.theta) > Th_2PI) // means abs() near 2*PI
            break;
        
        curID++;
    }
    
    if(curID > ccSeq.size() - 1) // Not found sign shifting
    {
        cout << "Point with theta sign turning NOT found!" << endl;
        return;
    }
    
    int zeroThetaID = curID;
    // fourthly reorder the seq make the pt with angle near 0 (and positive)
    // locates at the header of sequence
    // the Pt associated with curID should be placed firstly
    while(curID <= ccSeq.size() - 1)
    {
        zeroThetaFirstSeq.push_back(ccSeq[curID]);
        curID++;
    }
    
    curID = 0;
    while(curID < zeroThetaID)
    {
        zeroThetaFirstSeq.push_back(ccSeq[curID]);
        curID++;
    }
}

// find the initial appropriate values for uID and lID
void InitIpSetting(const PtInPolarSeq& seq, double angle, int& uID, int& lID)
{
    int numPts = (int)(seq.size());

    lID = -1; //numPts - 1;
    uID = 0;
    
    double uAngle = seq[uID].theta;
    //double lAngle = seq[lID].theta - 2*M_PI;
    
    while(uAngle < angle && uID < numPts -1)
    {
        lID = uID;
        uID++;
        uAngle = seq[uID].theta;
    }
}

// Ip: abbr. of Interpolate
void IpPolarCd(const PtInPolarCd& uPt, const PtInPolarCd& lPt,
                        double angle, PtInPolarCd& outPt)
{
    double uPt_theta = uPt.theta;
    double lPt_theta = lPt.theta;
    
    if(lPt_theta > uPt_theta)
        lPt_theta -= 2*M_PI;
        
    double t = (angle - lPt_theta) / (uPt_theta - lPt_theta);
    double r = t * lPt.r + (1.0 - t)* uPt.r;
    
    outPt.theta = angle;
    outPt.r = r;
}

void IpPolarPtSeqEvenly(const PtInPolarSeq& ascThetaSeq, int numInterval,
                        const Point& oriPt,
                        PolarContour& evenPolarSeq)
{
    int uID, lID;
    double angle = 0.0;
    double dAngle = 2.0*M_PI / numInterval;
    InitIpSetting(ascThetaSeq, angle, uID, lID); //Ip: interpolate
    
    evenPolarSeq.oriPt = oriPt;
    while(angle < 2*M_PI)
    {
        PtInPolarCd uPt = ascThetaSeq[uID];
        while(angle < uPt.theta)
        {
            angle += dAngle;
            PtInPolarCd outPt;
            IpPolarCd(ascThetaSeq[uID], ascThetaSeq[lID],
                                angle, outPt);
            evenPolarSeq.ptSeq.push_back(outPt);
        }
        
        lID = uID;
        uID++;
    }
}

// build a new version of polar point sequence that with even intervals of theta.
void BuildEvenPolarSeq(const PolarContour& rawPolarSeq,
                       int numInterval, // how many intervals from 0 to 2*Pi
                       PolarContour& evenPolarSeq)
{
    assert(rawPolarSeq.ptSeq.size() > 2);
    
    int numRawPts = (int)(rawPolarSeq.ptSeq.size());
    
    //firstly, add 2*Pi to those nagative thetas, make all the angles fall into [0 2*Pi]
    PtInPolarSeq posiThetaPtSeq;
    for(PtInPolarCd ptPC: rawPolarSeq.ptSeq)
    {
        if(ptPC.theta < 0)
            posiThetaPtSeq.push_back(PtInPolarCd(ptPC.r, ptPC.theta + 2*M_PI));
        else
            posiThetaPtSeq.push_back(ptPC);
    }
    
    // secondly rebuild the seq to make points allocated in counter clock-wise
    PtInPolarSeq ccwSeq = MakePolarSeqInCCWise(posiThetaPtSeq);
    
    // thridly we need to look for the point where 2*PI turns into zero
    // Here Monotonicity is assumed to be hold, i.e., when in the scaning, theta
    // would increase continuously, or decrease continuously.
    PtInPolarSeq ascThetaSeq; // the seq with ascending thetas
    ReorderPolarPtSeq(ccwSeq, ascThetaSeq);

    // finally, the very important step: interpolate with even angular interval
    // and a step full of challenge
    IpPolarPtSeqEvenly(ascThetaSeq, numInterval,
                       rawPolarSeq.oriPt, evenPolarSeq);
}

// 针对的是封闭的弧线
void SmClosedPolarSeq(const PolarContour& evenPolarSeq,
                      int mwLen, //length of moving window
                      PolarContour& smoothPolarSeq)
{
    smoothPolarSeq.oriPt = evenPolarSeq.oriPt;
    
    int numPts = (int)(evenPolarSeq.ptSeq.size());
    int mwLenHalf = mwLen / 2;
    
    for(int i = 0; i <= numPts-1; i++)
    {
        //vector<int> ptIDsInWin;
        //vector<PtInPolarCd&> ptsInWin;
        double sum_r = 0.0;
        for(int l = i-mwLenHalf; l<= i+mwLenHalf; l++)
        {
            //ptIDsInWin.push_back(l);
            int id = l % numPts;
            //ptsInWin.push_back(evenPolarSeq.ptSeq[id]);
            sum_r += evenPolarSeq.ptSeq[id].r;
        }
        
        double avg_r = sum_r / mwLen;
        double theta = evenPolarSeq.ptSeq[i].theta;
        PtInPolarCd ipPt(avg_r, theta);
        smoothPolarSeq.ptSeq.push_back(ipPt);
    }
}

// 针对的是不封闭的弧线
void SmOpenPolarSeqV2(const PolarContour& evenPolarSeq,
                      int mwLen, //length of moving window
                      PolarContour& smPolarSeq)
{
    smPolarSeq.oriPt = evenPolarSeq.oriPt;
    PtInPolarSeq& smPtSeq = smPolarSeq.ptSeq;
    
    int numPts = (int)(evenPolarSeq.ptSeq.size());
    int mwLenHalf = mwLen / 2;
    
    for(int i = 0; i <= numPts-1; i++)
    {
        // 在两端，若凑不够窗口中的点数目，则保留原坐标，跳过平均
        if(i-mwLenHalf < 0)
        {
            smPtSeq.push_back(evenPolarSeq.ptSeq[i]);
            continue;
        }
        if(i+mwLenHalf >= numPts)
        {
            smPtSeq.push_back(evenPolarSeq.ptSeq[i]);
            continue;
        }
        
        double sum_r = 0.0;
        for(int l = i-mwLenHalf; l<= i+mwLenHalf; l++)
        {
            sum_r += evenPolarSeq.ptSeq[l].r;
        }
        
        double avg_r = sum_r / mwLen;
        double theta = evenPolarSeq.ptSeq[i].theta;
        PtInPolarCd ipPt(avg_r, theta);
        smPtSeq.push_back(ipPt);
    }
}

// 在两个端点处，以复制的方式来凑足窗口所需点数
void SmOpenPolarSeqV3(const PolarContour& evenPolarSeq,
                      int mwLen,
                      PolarContour& smPolarSeq)
{
    smPolarSeq.oriPt = evenPolarSeq.oriPt;
    PtInPolarSeq& smPtSeq = smPolarSeq.ptSeq;
    
    int numPts = (int)(evenPolarSeq.ptSeq.size());
    int mwLenHalf = mwLen / 2;
    
    // padding with the first copy and last copy to meet the moving windows
    PtInPolarSeq padSeq;
    for(int i = 0; i <= mwLenHalf-1; i++)
        padSeq.push_back(evenPolarSeq.ptSeq[0]);
    for(int i = 0; i <= numPts-1; i++)
        padSeq.push_back(evenPolarSeq.ptSeq[i]);
    for(int i = 0; i <= mwLenHalf-1; i++)
        padSeq.push_back(evenPolarSeq.ptSeq[numPts-1]);
    
    for(int i = 0; i <= numPts-1; i++)
    {
        double sum_r = 0.0;
        for(int l = i-mwLenHalf; l<= i+mwLenHalf; l++)
        {
            sum_r += padSeq[l].r;
        }
        
        double avg_r = sum_r / mwLen;
        double theta = padSeq[i].theta;
        PtInPolarCd ipPt(avg_r, theta);
        smPtSeq.push_back(ipPt);
    }
}

Point PolarPt2CartPt(const PtInPolarCd& polarPt, const Point& oriPt) //, Point& cartPt)
{
    int x = oriPt.x + (int)(polarPt.r * cos(polarPt.theta));
    int y = oriPt.y + (int)(polarPt.r * sin(polarPt.theta));
    
    return Point(x, y);
}

void PolarPtSeq2CartPtSeq(const PolarContour& polarCont,
                          CONTOUR& cartCont)
{
    for(PtInPolarCd polarPt: polarCont.ptSeq)
    {
        Point cartPt = PolarPt2CartPt(polarPt, polarCont.oriPt);
        cartCont.push_back(cartPt);
    }
}

//////////////////////////////////////////////////////////////////////////////////////////
void IpPolarSeqEvenlyCCW(const PtInPolarSeq& ascThetaSeq, int numInterval,
                        const Point& oriPt,
                        PolarContour& evenPolarSeq)
{
    int numRawPt = ascThetaSeq.size();
    PtInPolarCd stPt = ascThetaSeq[0];
    PtInPolarCd edPt = ascThetaSeq[numRawPt-1];
    
    int uID = 1;
    int lID = 0;
    
    double dAngle = (edPt.theta - stPt.theta) / numInterval;
    double angle = stPt.theta;
    
    evenPolarSeq.oriPt = oriPt;
    while(angle < edPt.theta)
    {
        PtInPolarCd uPt = ascThetaSeq[uID];
        while(angle < uPt.theta)
        {
            angle += dAngle;
            PtInPolarCd outPt;
            IpPolarCd(ascThetaSeq[uID], ascThetaSeq[lID],
                                angle, outPt);
            evenPolarSeq.ptSeq.push_back(outPt);
        }
        
        lID = uID;
        uID++;
    }
}

void IpPolarSeqEvenlyCW(const PtInPolarSeq& desThetaSeq, int numInterval,
                        const Point& oriPt,
                        PolarContour& evenPolarSeq)
{
    int numRawPt = desThetaSeq.size();
    PtInPolarCd stPt = desThetaSeq[0];
    PtInPolarCd edPt = desThetaSeq[numRawPt-1];
    
    int uID = 1;
    int lID = 0;
    
    double dAngle = (edPt.theta - stPt.theta) / numInterval;
    double angle = stPt.theta;
    
    evenPolarSeq.oriPt = oriPt;
    while(angle > edPt.theta)
    {
        PtInPolarCd uPt = desThetaSeq[uID];
        while(angle > uPt.theta)
        {
            angle += dAngle; // dAngle is a negative value
            PtInPolarCd outPt;
            IpPolarCd(desThetaSeq[uID], desThetaSeq[lID],
                                angle, outPt);
            evenPolarSeq.ptSeq.push_back(outPt);
        }
        
        lID = uID;
        uID++;
    }
}
