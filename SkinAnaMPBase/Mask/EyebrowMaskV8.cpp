//
//  EyebrowMaskV8.cpp

/*******************************************************************************
Author: Fu Xiaoqiang
Date:   2022/11/3

********************************************************************************/

#include "../BSpline/ParametricBSpline.hpp"
#include "FundamentalMask.hpp"
#include "../Geometry.hpp"
#include "../Chaikin.hpp"
#include "../Common.hpp"
#include "../Utils.hpp"

#include "../FaceBgSeg/FaceBgSegV2.hpp"
#include "EyebrowMaskV8.hpp"
#include "SmEyePg.hpp"
#include "polyitems_fit.hpp"

//-------------------------------------------------------------------------------------------
// Eb is abbreviation for Eyebrow
// Pg is abbreviation for Polygon
Point2i getPtOnEb(const FaceInfo& faceInfo, EyeID eyeID, int ptIndex)
{
    if(eyeID == LEFT_EYE)
    {
        return faceInfo.lEyeRefinePts[ptIndex];
    }
    else  // RIGHT_EYE
    {
        return faceInfo.rEyeRefinePts[ptIndex];
    }
}

Point2i IpPtInERG(const Point2i eyeRefPts[NUM_PT_EYE_REFINE_GROUP],
                  int pIndex1, int pIndex2, float t)
{
    Point2i P1 = eyeRefPts[pIndex1];
    Point2i P2 = eyeRefPts[pIndex2];
    
    // be careful with the Point order and the value of t when to invoke InnerInterpolate()
    Point2i P3 = Interpolate(P1, P2, t);
    return P3;
}

// (dx, dy) = kpCP - segCP
Point2i TransEyeRefPt2SegSpace(const Point2i& eyeRefPt, int dx, int dy)
{
    return Point2i(eyeRefPt.x - dx, eyeRefPt.y - dy);
}

//ERPs: eye refined points
// 用分割出的眉毛中心点来修正ERPs的坐标
// transform the eye refine points in face mesh space into the segment space
void FixERPsBySegBrowCP(const Point2i eyeRefPts[NUM_PT_EYE_REFINE_GROUP], // input
                       const Point2i& segCP, // eye center point in segment space
                       Point2i fixedEyeRefPts[NUM_PT_EYE_REFINE_GROUP]   /* output */)
{
    Point2i kpCP = (eyeRefPts[53] + eyeRefPts[52]) / 2; // brow center point in key points space
    int dx = kpCP.x - segCP.x;  //
    int dy = kpCP.y - segCP.y;  //
    
    for(int i=0; i<NUM_PT_EYE_REFINE_GROUP; i++)
    {
        fixedEyeRefPts[i] = TransEyeRefPt2SegSpace(eyeRefPts[i], dx, dy);
    }
}

/******************************************************************************************
*******************************************************************************************/

void ForgeBrowPg(const Point2i eyeRefPts[NUM_PT_EYE_REFINE_GROUP],
                 const Point2i& segBrowCP,
                 POLYGON& browPg)
{
    
    // 采用Eye Refine Region的点，左眉毛、右眉毛的坐标索引是相同的！
    // 对右侧眉毛而言，69点是内侧左下角点。点序列按顺时针方向排列。
    //int browERPIDs[] = {69, 68, 67, 66, 65, 64, 50, 43, 44, 45};
    Point2i fixedERPts[NUM_PT_EYE_REFINE_GROUP]; // ER: eye refine
    FixERPsBySegBrowCP(eyeRefPts, segBrowCP, fixedERPts);

    POLYGON coarsePg;
    
    Point2i pt69a = InterpolateX(fixedERPts[69], fixedERPts[70], 0.5);
    coarsePg.push_back(pt69a);
    
    Point2i pt68a = InterpolateY(fixedERPts[68], fixedERPts[69], 0.10);
    coarsePg.push_back(pt68a);
    
    Point2i pt67a = Interpolate(fixedERPts[67], fixedERPts[53], -0.25);
    coarsePg.push_back(pt67a);

    coarsePg.push_back(fixedERPts[66]);
    coarsePg.push_back(fixedERPts[65]);
    
    Point2i iPt64 = IpPtInERG(fixedERPts, 64, 63, 0.5);
    coarsePg.push_back(iPt64);

    coarsePg.push_back(fixedERPts[50]);

    Point2i iPt43 = IpPtInERG(fixedERPts, 50, 44, 0.5);
    coarsePg.push_back(iPt43);
    
    coarsePg.push_back(fixedERPts[44]);
    coarsePg.push_back(fixedERPts[45]);

    // then convert the corse to the refined
    int csNumPoint = 50;
    DenseSmoothPolygon(coarsePg, csNumPoint, browPg);
}
/******************************************************************************************
*******************************************************************************************/
void ForgeBrowsMask(const FaceInfo& faceInfo,
                    const FaceSegRst& segRst,  // Rst: result
                    Mat& outMask)
{
    POLYGON leftBrowPg, rightBrowPg;
    ForgeBrowPg(faceInfo.lEyeRefinePts, segRst.leftBrowCP, leftBrowPg);
    ForgeBrowPg(faceInfo.rEyeRefinePts, segRst.rightBrowCP, rightBrowPg);

    POLYGON_GROUP polygonGroup;
    polygonGroup.push_back(leftBrowPg);
    polygonGroup.push_back(rightBrowPg);
    
    //Mat outOrigMask = ContourGroup2Mask(faceInfo.imgWidth, faceInfo.imgHeight, polygonGroup);
    
    /*
    int dila_size = 10;
    Mat element = getStructuringElement(MORPH_ELLIPSE,
                           Size(2*dila_size + 1, 2*dila_size+1),
                           Point(dila_size, dila_size));
    
    cv::Mat outExpandedMask(faceInfo.imgHeight, faceInfo.imgWidth, CV_8UC1, cv::Scalar(0));
    dilate(outOrigMask, outMask, element);
    */
    
    outMask = ContourGroup2Mask(faceInfo.srcImgS, polygonGroup);
}

//-------------------------------------------------------------------------------------------


//RefPts: refined points
// transform the eye refine points in face mesh space into the segment space
void FixERPsBySegEyeCP(const Point2i eyeRefPts[NUM_PT_EYE_REFINE_GROUP], // input
                       const Point2i& segCP, // eye center point in segment space
                       Point2i fixedEyeRefPts[NUM_PT_EYE_REFINE_GROUP] /* utput */)
{
    Point2i kpCP = (eyeRefPts[19] + eyeRefPts[20]) / 2; // eye center point in key points space
    int dx = kpCP.x - segCP.x;  //
    int dy = kpCP.y - segCP.y;  //
    
    for(int i=0; i<NUM_PT_EYE_REFINE_GROUP; i++)
    {
        fixedEyeRefPts[i] = TransEyeRefPt2SegSpace(eyeRefPts[i], dx, dy);
    }
}


/**********************************************************************************************/

void ForgeInitEyePg(const Point2i eyeRefinePts[NUM_PT_EYE_REFINE_GROUP],
                    const Point2i& eyeSegCP, POLYGON& initEyePg)
{
    POLYGON refinedPolygon;
        
    // 采用Eye Refine Region的点！
    // 以右眼为基准，从内侧上角点开始，顺时针绕一周
    int ptIDsOnCurve[] = {40, 23, 22, 21, 20, 19, 18, 17, 32,
        55, 56, 57, 58, 59, 60, 61}; // 顺时针计数, total 9 points
    
    POLYGON initEyePg0;
    int numCoarsePts = sizeof(ptIDsOnCurve) / sizeof(int);
    for(int i = 0; i<numCoarsePts; i++)
    {
        int index = ptIDsOnCurve[i];
        initEyePg0.push_back(eyeRefinePts[index]);
    }
    
    // 根据两个区域（分割出的栅格版，和关键点构造出的矢量版）的质心位置差异，
    // 对矢量版的点位置进行移动，使得二者的重叠面积尽可能地大。
    Moments m = moments(initEyePg0, false);
    Point2f mc_f(m.m10/m.m00, m.m01/m.m00);
    
    Point2i eyePgCP((int)mc_f.x, (int)mc_f.y);
    
    Point2i vecDiff = eyePgCP - eyeSegCP;
    for(Point2i pt: initEyePg0)
    {
        initEyePg.push_back(pt - vecDiff);
    }
}

// 检查点序列X坐标的单调性
void CheckMonoX(const CONTOUR& smEyeCt)
{
    int gtTimes = 0;
    int ltTimes = 0;
    
    int numPt = static_cast<int>(smEyeCt.size());

    for(int i=1; i<=numPt-1; i++)
    {
        if(smEyeCt[i-1].x < smEyeCt[i].x)
            gtTimes++;
        else
            ltTimes++;
    }
    
    cout << "gtTimes: " << gtTimes << ", ltTimes: " << ltTimes << endl;
}

// forge the polygong of one eye, only use the result of face/bg segment
void ForgeEyePg(Size srcImgS, const SegMask& eyeSegMask,
                const EyeSegFPs& eyeFPs,
                const Point2i& browCP,
                CONTOUR& smEyeCt)
{
    CONTOURS contours;
    findContours(eyeSegMask.mask, contours,
                     cv::noArray(), RETR_EXTERNAL, CHAIN_APPROX_NONE);

    CONTOUR ssEyeCt; // in Source Space
    transCt_LocalSegNOS2SS(contours[0], eyeSegMask.bbox.tl(),
                    srcImgS, ssEyeCt);
        
    CONTOUR upEyeCurve, lowEyeCurve;
    SplitEyeCt(ssEyeCt,eyeFPs.lCorPt, eyeFPs.rCorPt,
                upEyeCurve, lowEyeCurve);
    
    CONTOUR smUpEyeCurve;
    SmCtByPIFitOrd3V2(upEyeCurve, smUpEyeCurve);

    CONTOUR smLowEyeCurve;
    SmCtByPIFitOrd3V2(lowEyeCurve, smLowEyeCurve);
    
    SmCorSecOnEyePgV2(srcImgS, smUpEyeCurve, smLowEyeCurve, browCP, smEyeCt);
}

void ForgeEyesMask(const Mat& srcImage, // add this variable just for debugging
                   const FaceInfo& faceInfo,
                   const FaceSegRst& segRst, //Rst: result,
                   Mat& outMask)
{
    POLYGON leftEyePg, rightEyePg;

    Size srcImgS = faceInfo.srcImgS;
    
    ForgeEyePg(srcImgS, segRst.lEyeMaskNOS, segRst.lEyeFPs,
               segRst.leftBrowCP, leftEyePg);
    ForgeEyePg(srcImgS, segRst.rEyeMaskNOS, segRst.rEyeFPs,
               segRst.rightBrowCP, rightEyePg);
    
    Rect lEyeRect = boundingRect(leftEyePg);
    Rect rEyeRect = boundingRect(rightEyePg);

    CONTOUR lssLEyeCt;
    transCt_GS2LSS(leftEyePg, lEyeRect.tl(), lssLEyeCt);
    Mat lssLEyeMask(lEyeRect.size(), CV_8UC1, Scalar(0));
    DrawContOnMask(lssLEyeCt, lssLEyeMask);
    
    POLYGON_GROUP polygonGroup;
    polygonGroup.push_back(rightEyePg);
    polygonGroup.push_back(leftEyePg);
    
    //Mat outOrigMask = ContourGroup2Mask(srcImgS.width, srcImgS.height, polygonGroup);
    outMask = ContourGroup2Mask(srcImgS.width, srcImgS.height, polygonGroup);
}

void DeterCorSecPtIdxs(const vector<PtInfo>& ptInfoSec,
                       int& idx1, int& idx2 //  idx1 on upper curve, idx2 on lower curve
                       )
{
    //int idx1_lsec = lIdxSec[0];
    //int idx2_lsec = lIdxSec[lIdxSec.size() - 1];
    int numPt = static_cast<int>(ptInfoSec.size());
    if(ptInfoSec[0].polCd.r > ptInfoSec[numPt-1].polCd.r)
    {
        idx1 = ptInfoSec[numPt-1].idx;
        idx2 = ptInfoSec[0].idx;
    }
    else
    {
        idx1 = ptInfoSec[0].idx;
        idx2 = ptInfoSec[numPt-1].idx;
    }
}

void CombOriSec(const CONTOUR& eyeCont, int idx_r1, int idx_l1, CONTOUR& finCt)
{
    int N = static_cast<int>(eyeCont.size());

    if(idx_r1 < idx_l1)
    {
        for(int i = idx_r1; i<= idx_l1; i++)
        {
            finCt.push_back(eyeCont[i]);
        }
    }
    else //idx_r1 > idx_l1
    {
        for(int i = idx_r1; i<= idx_l1 + N; i++)
        {
            int j = i % N;
            finCt.push_back(eyeCont[j]);
        }
    }
}

void CombineSec(const CONTOUR& smCorSec, CONTOUR& finCt)
{
    for(Point pt: smCorSec)
        finCt.push_back(pt);
}

void CalcPtInfoSeq(const CONTOUR& curve, const Point2i& browCP,
                   float& minA, float& maxA, vector<PtInfoV2>& PtInfoSeq)
{
    double DoublePI = 2*M_PI;

    for(Point pt: curve)
    {
        Point diff = pt - browCP;
        double r = sqrt(diff.x * diff.x + diff.y * diff.y);
        /*
         注意图像坐标系和数学坐标系的差异：y轴坐标方向不同，由此导致一系列的麻烦和出错机会。
         我们的设定是，还是按逆时针方向来扫描，四个象限还是按照数学坐标系的设定
         （右上为I，左上为II，左下为III，右下为IV）。但是，给atan2()的返回结果前加个负号！！！
         */
        double theta = -atan2(diff.y, diff.x);
        if(theta < 0.0)
            theta += DoublePI;
        
        if(maxA < theta)
            maxA = theta;
        if(minA > theta)
            minA = theta;
        
        PtInfoV2 ptInfo(pt, r, theta);
        PtInfoSeq.push_back(ptInfo);
    }
}

void GetLCorSec(const vector<PtInfoV2>& upPtInfoSeq,
                const vector<PtInfoV2>& lowPtInfoSeq,
                float lAngTh, CONTOUR& lCorSecCt,
                int& l1IdxOnUpCur, int& l2IdxOnLowCur)
{
    // 求上弧线上属于左角点的点序列
    l1IdxOnUpCur = -1;
    int N1 = static_cast<int>(upPtInfoSeq.size());
    int numGotPtUpCurve = 0;
    CONTOUR lCorSecUpCt;
    for(int i=0; i<=N1-1; i++)
    {
        if(upPtInfoSeq[i].polCd.theta < lAngTh)
        {
            numGotPtUpCurve++;
            lCorSecUpCt.push_back(upPtInfoSeq[i].carCd);
            
            if(l1IdxOnUpCur == -1) // 第一个小于lAngTH的点
                l1IdxOnUpCur = i;
        }
    }
    
    // 求下弧线上属于左角点的点序列
    CONTOUR lCorSecLowCt;
    int N2 = static_cast<int>(lowPtInfoSeq.size());
    for(int i=0; i<=N2-1; i++)
    {
        if(lowPtInfoSeq[i].polCd.theta < lAngTh)
        {
            lCorSecLowCt.push_back(lowPtInfoSeq[i].carCd);
        }
    }
    
    // 中间通过插值的方式，多补几个点
    int numAddedPt = 4;
    float interval = 1.0 / (numAddedPt + 1);
    
    Point2i lastPt = lCorSecUpCt[lCorSecUpCt.size()-1];
    Point2i nextPt = lCorSecLowCt[0];
    CONTOUR middleCt;
    for(int i=1; i<=numAddedPt; i++)
    {
        Point2i pt = Interpolate(lastPt, nextPt, i*interval);
        middleCt.push_back(pt);
    }
    
    CombineSec(lCorSecUpCt, lCorSecCt);
    CombineSec(middleCt, lCorSecCt);
    CombineSec(lCorSecLowCt, lCorSecCt);

    l2IdxOnLowCur = lCorSecCt.size() - numGotPtUpCurve - numAddedPt;
}

void GetRCorSec(const vector<PtInfoV2>& upPtInfoSeq,
                const vector<PtInfoV2>& lowPtInfoSeq,
                float rAngTh, CONTOUR& rCorSecCt,
                int& r1IdxOnUpCur, int& r2IdxOnLowCur)
{
    // 先求下弧线上属于右角点的点序列
    r2IdxOnLowCur = -1;
    int N1 = static_cast<int>(lowPtInfoSeq.size());
    int numGotPtLowCurve = 0;
    CONTOUR rCorSecLowCt;
    for(int i=0; i<=N1-1; i++)
    {
        if(lowPtInfoSeq[i].polCd.theta > rAngTh)
        {
            numGotPtLowCurve++;
            rCorSecLowCt.push_back(lowPtInfoSeq[i].carCd);
            
            if(r2IdxOnLowCur == -1) // 第一个小于lAngTH的点
                r2IdxOnLowCur = i;
        }
    }
    
    // 再求上弧线上属于右角点的点序列
    CONTOUR rCorSecUpCt;
    int N2 = static_cast<int>(upPtInfoSeq.size());
    for(int i=0; i<=N2-1; i++)
    {
        if(upPtInfoSeq[i].polCd.theta > rAngTh)
        {
            rCorSecUpCt.push_back(upPtInfoSeq[i].carCd);
        }
    }
    
    // 中间通过插值的方式，多补几个点
    int numAddedPt = 4;
    float interval = 1.0 / (numAddedPt + 1);
    
    Point2i lastPt = rCorSecLowCt[rCorSecLowCt.size()-1];
    Point2i nextPt = rCorSecUpCt[0];
    CONTOUR middleCt;
    for(int i=1; i<=numAddedPt; i++)
    {
        Point2i pt = Interpolate(lastPt, nextPt, i*interval);
        middleCt.push_back(pt);
    }
    
    CombineSec(rCorSecLowCt, rCorSecCt);
    CombineSec(middleCt, rCorSecCt);
    CombineSec(rCorSecUpCt, rCorSecCt);
    
    r1IdxOnUpCur = rCorSecCt.size() - numGotPtLowCurve - numAddedPt;
}

void getR1L1SecOnUpCurv(const CONTOUR& smUpEyeCurve,
                        int R1IdxUpCur, int L1IdxUpCur,
                        CONTOUR& R1L1Sec)
{
    for(int i=R1IdxUpCur; i<=L1IdxUpCur; i++)
        R1L1Sec.push_back(smUpEyeCurve[i]);
}

void getL2R2SecOnLowCurv(const CONTOUR& smLowEyeCurve,
                         int L2IdxLowCur, int R2IdxLowCur,
                         CONTOUR& L2R2Sec)
{
    for(int i=L2IdxLowCur; i<=R2IdxLowCur; i++)
        L2R2Sec.push_back(smLowEyeCurve[i]);
}

void SmCorSecOnEyePgV2(const Size& srcImgS,
                       const CONTOUR& upEyeCurve, const CONTOUR& lowEyeCurve,
                       const Point2i& browCP, CONTOUR& finCt)
{
    float maxA = -999.9;
    float minA = 999.9;
    
    vector<PtInfoV2> upPtInfoSeq;
    CalcPtInfoSeq(upEyeCurve, browCP,
                  minA, maxA, upPtInfoSeq);
    
    vector<PtInfoV2> lowPtInfoSeq;
    CalcPtInfoSeq(lowEyeCurve, browCP,
                  minA, maxA, lowPtInfoSeq);
        
    float devA = (maxA - minA) / 8.0;
    float lAngTh = minA + devA;
    float rAngTh = maxA - devA;
    
    //vector<PtInfo> lCorPtInfoSec, rCorPtInfoSec; // Sec: Section
    CONTOUR lCorSecCt, rCorSecCt;
    
    //int idx_r1, idx_r2, idx_l1, idx_l2; // in original eye contour
    int l1IdxOnUpCur, l2IdxOnLowCur;
    int r1IdxOnUpCur, r2IdxOnLowCur;

    GetLCorSec(upPtInfoSeq, lowPtInfoSeq, lAngTh, lCorSecCt,
               l1IdxOnUpCur, l2IdxOnLowCur);
    
    GetRCorSec(upPtInfoSeq, lowPtInfoSeq, rAngTh, rCorSecCt,
               r1IdxOnUpCur, r2IdxOnLowCur);
    
    int smIterTimes = 5;
    CONTOUR lSmCorSec0, lSmCorSec;
    DelMaxCurTwoPtsOnCt(lCorSecCt, lSmCorSec0);
    SmUnclosedContCK(lSmCorSec0, smIterTimes, lSmCorSec);
    
    /*
    Mat canvas2(srcImgS, CV_8UC1, Scalar(0));
    for(int i = 0; i < lCorSecCt.size(); i++)
    {
        cv::circle(canvas2, lCorSecCt[i], 1, Scalar(150), cv::FILLED);
    }
    cv::putText(canvas2, to_string(0), lCorSecCt[0],
                FONT_HERSHEY_SIMPLEX, 1, Scalar(150), 1);
    cv::putText(canvas2, to_string(lCorSecCt.size()-1), lCorSecCt[lCorSecCt.size()-1],
                FONT_HERSHEY_SIMPLEX, 1, Scalar(150), 1);
    for(int i = 0; i < rCorSecCt.size(); i++)
    {
        cv::circle(canvas2, rCorSecCt[i], 1, Scalar(150), cv::FILLED);
    }
    cv::putText(canvas2, to_string(0), rCorSecCt[0],
                FONT_HERSHEY_SIMPLEX, 1, Scalar(150), 1);
    cv::putText(canvas2, to_string(rCorSecCt.size()-1), rCorSecCt[rCorSecCt.size()-1],
                FONT_HERSHEY_SIMPLEX, 1, Scalar(150), 1);
    imwrite("CorSecCt.png", canvas2);
    */
    
    CONTOUR rSmCorSec0, rSmCorSec;
    DelMaxCurTwoPtsOnCt(rCorSecCt, rSmCorSec0);
    SmUnclosedContCK(rSmCorSec0, smIterTimes, rSmCorSec);

    /*
    Mat canvas3(srcImgS, CV_8UC1, Scalar(0));
    for(int i = 0; i < lSmCorSec.size(); i++)
    {
        cv::circle(canvas3, lSmCorSec[i], 1, Scalar(150), cv::FILLED);
    }
    for(int i = 0; i < rSmCorSec.size(); i++)
    {
        cv::circle(canvas3, rSmCorSec[i], 1, Scalar(150), cv::FILLED);
    }
    imwrite("SmCorSec.png", canvas3);
    */
    
    //合并的顺序： r1-l1, l1-l2, l2-r2, r2-r1
    CONTOUR R1L1Sec;
    getR1L1SecOnUpCurv(upEyeCurve, r1IdxOnUpCur, l1IdxOnUpCur, R1L1Sec);
    CombineSec(R1L1Sec, finCt);
    CombineSec(lSmCorSec, finCt);

    CONTOUR L2R2Sec;
    getL2R2SecOnLowCurv(lowEyeCurve, l2IdxOnLowCur, r2IdxOnLowCur, L2R2Sec);
    CombineSec(L2R2Sec, finCt);
    CombineSec(rSmCorSec, finCt);
}
