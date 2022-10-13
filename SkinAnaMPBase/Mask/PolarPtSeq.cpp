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
        for(int i = numPts; i>=0; i--)
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

    lID = numPts - 1;
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

void SmoothPolarPtSeq(const PolarContour& evenPolarSeq,
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


//----Polar coordinate system not used in the following functions method,
//----Cartesian coordinate system used instead--------------------------

void FindPtWithMaxR(const CONTOUR& contSect, const Point& eyeCP, Point& outPt)
{
    double max_r = 0.0;
    Point candiOutPt(-1, -1); // candidate
    for(Point pt: contSect)
    {
        Point relaCd = pt - eyeCP;
        double r = sqrt(relaCd.x*relaCd.x + relaCd.y*relaCd.y);
        if(r > max_r)
        {
            max_r = r;
            candiOutPt = pt;
        }
    }
    
    outPt = candiOutPt;
}

// P1: the top left corner on the eye contour,
// P2: the top right corner on the eye contour.
void CalcEyeCtP1P2(const CONTOUR& eyeCont, const Point& eyeCP, Point& P1, Point& P2)
{
    // divide the coordinate space into 4 sections, I, II, III, IV
    // by eyeCP and the horizontal and vertical axises.
    
    CONTOUR contSect1, contSect2;
    for(Point pt: eyeCont)
    {
        int dx = pt.x - eyeCP.x;
        int dy = pt.y - eyeCP.y;
        
        if(dx > 0 && dy < 0)
            contSect1.push_back(pt);
        else if(dx < 0 && dy < 0)
            contSect2.push_back(pt);
    }
    
    // seek for P1 in the section II,
    FindPtWithMaxR(contSect2, eyeCP, P1);

    // Seek for P2 in the section I
    FindPtWithMaxR(contSect1, eyeCP, P2);
}

// P4: the top middle point on the eye contour,
// P3: the bottom middle point on the eye contour.
void CalcEyeCtP3P4(const CONTOUR& eyeCont, const Point& eyeCP, Point& P3, Point& P4)
{
    // divide the coordinate space into 4 rotated sections, I, II, III, IV
    // by rotating the ordinary 4-sections 45 degree CCW.
    // eyeCP as orignal point of coordinate system
    
    const double M_PI_Quarter = M_PI / 4.0;
    const double M_PI_3Quarters = M_PI * 0.75;
    const double M_PI_5Quarters = M_PI * 1.25;
    const double M_PI_7Quarters = M_PI * 1.75;

    CONTOUR contSect1, contSect3;
    for(Point pt: eyeCont)
    {
        int dx = pt.x - eyeCP.x;
        int dy = pt.y - eyeCP.y;
        double theta = atan2(dy, dx); // theta in [-M_PI M_PI] at this time point
        if(theta < 0.0)
            theta += M_PI;
        
        if(theta > M_PI_Quarter && theta < M_PI_3Quarters)
            contSect1.push_back(pt);
        else if(theta > M_PI_5Quarters && theta < M_PI_7Quarters)
            contSect3.push_back(pt);
    }
}
