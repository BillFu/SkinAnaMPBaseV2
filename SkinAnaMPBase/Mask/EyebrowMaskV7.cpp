//
//  EyebrowMaskV7.cpp

/*******************************************************************************
Author: Fu Xiaoqiang
Date:   2022/10/27

********************************************************************************/

#include "EyebrowMaskV7.hpp"
#include "../BSpline/ParametricBSpline.hpp"
#include "FundamentalMask.hpp"
#include "Geometry.hpp"
#include "Common.hpp"

#include "../FaceBgSeg/FaceBgSegV2.hpp"
#include "../Snake/activecontours.h"

#include "../Snake/GaussField.hpp"

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

// 用分割的结果构造出一只眼睛的轮廓多边形
// eyeCP: given by face/bg segment and in source space
void ForgeEyePgBySnakeAlg(Size srcImgS,
                          const Point2i eyeRefinePts[NUM_PT_EYE_REFINE_GROUP],
                          const SegMask& eyeSegMask, // in NOS
                          const EyeSegFPs& eyeSegFPs,
                          const Point2i& eyeSegCP,
                          POLYGON& eyePg)
{
    POLYGON initEyePg;
    ForgeInitEyePg(eyeRefinePts, eyeSegCP, initEyePg);
    
    Rect initEyePgBBox = boundingRect(initEyePg); // maybe need padding and enlarge
    Rect eyeMaskBBox = RectNOS2RectSS(srcImgS, eyeSegMask.bbox);
    
    Mat segEyeMaskSS;
    //DetectCorner_ST(eyeSegMask.mask); // shi-tomasi
    resize(eyeSegMask.mask, segEyeMaskSS, eyeMaskBBox.size(), INTER_NEAREST);
    
    // enlarge the initEyePgBBox by use of union operation
    initEyePgBBox = initEyePgBBox | eyeMaskBBox;

    // make initEyePgBBox enlarged
    int padSize = 20;
    InflateRect(padSize, initEyePgBBox);

    Mat workImg(initEyePgBBox.size(), CV_8UC1, Scalar(0));
    Rect relativeRect = CalcRelativeRect(initEyePgBBox, eyeMaskBBox);
    segEyeMaskSS.copyTo(workImg(relativeRect)); // 将segEyeMask先复写到workImg
    // imwrite("InitSegMask.png", workImg);

    // 坐标转换，由source space转入initEyePgBBox框定的局部坐标系，也称工作坐标系(WC)
    POLYGON initEyePgWC;
    for(Point2i pt: initEyePg)
    {
        Point relaPt = pt - initEyePgBBox.tl();
        initEyePgWC.push_back(relaPt);
    }
    
    // 将分割结果中提取到的两个角点，转换到initEyePgBBox框定的工作坐标系(WC)
    Point2i lEyeCornerWC, rEyeCornerWC;
    lEyeCornerWC = eyeSegFPs.lCorPt - initEyePgBBox.tl();
    rEyeCornerWC = eyeSegFPs.rCorPt - initEyePgBBox.tl();

    // --- 在栅格域合并两个Mask
    CONTOURS contours;
    contours.push_back(initEyePgWC);
    // convert init eye polygon to raster and calculate the union of it and eye seg mask.
    // 再把initEyePgWC转化为栅格形式，并与segEyeMask会师，合并
    drawContours(workImg, contours, 0, Scalar(255), FILLED);

    int diaRadius = 3;
    Mat element = getStructuringElement( MORPH_ELLIPSE,
                           Size(2*diaRadius + 1, 2*diaRadius + 1),
                           Point(diaRadius, diaRadius));
    dilate(workImg, workImg, element);
    imwrite("CombInitAC.png", workImg);

    // --- 把合并后的结果转为矢量域
    CONTOURS acCts; // acCts[0] will be input snake algorithm
    findContours(workImg, acCts,
            cv::noArray(), RETR_EXTERNAL, CHAIN_APPROX_NONE);
    
    workImg.release();
    
    int numPts = acCts[0].size();

    CONTOUR sampPtCt;  // 稀疏化的轮廓点
    int sampGap = 20;
    int numSampPt = numPts / sampGap;
    
    for(int i = 0; i < numSampPt; i++)
    {
        int id = i * sampGap;
        sampPtCt.push_back(acCts[0][id]);
    }
    
    //--------------------------------------------------------------
    /*
    // ---- just for debugging
    Mat canvas(initEyePgBBox.size(), CV_8UC1, Scalar(0));
    segEyeMaskSS.copyTo(canvas(relativeRect));
    drawContours(canvas, acCts, 0, Scalar(150), 2);
    circle(canvas, lEyeCornerWC, 5, Scalar(150), cv::FILLED);
    circle(canvas, rEyeCornerWC, 5, Scalar(150), cv::FILLED);
    imwrite("InitStateAC.png", canvas);
    canvas.release();

    double arcLen = arcLength(acCts[0], true);
    cout << "arcLen: " << arcLen << endl;
    */
    //--------------------------------------------------------------
    cvalg::ActiveContours acAlg;
    AlgoParams ap;
    ap._nPoints = static_cast<int>(sampPtCt.size());
    acAlg.setParams(&ap);

    Mat acImg(initEyePgBBox.size(), CV_8UC1, Scalar(0));
    segEyeMaskSS.copyTo(acImg(relativeRect)); // 将segEyeMask复写到acImg
    
    Size acImgS = acImg.size();
    acAlg.init(acImgS.width, acImgS.height);

    acAlg.setInitCont(sampPtCt);
    
    //acImg = ~acImg;
    vector<Point2i> peaks;
    peaks.push_back(lEyeCornerWC);
    peaks.push_back(rEyeCornerWC);

    int fieldW = initEyePgBBox.size().width;
    int fieldH = initEyePgBBox.size().height;
    
    // cornerField经过了反相，越接近peaks，值越小
    Mat cornerField = BuildGaussField(fieldW, fieldH, 7, peaks);
    
    acAlg.optimize(acImg, cornerField, 21, 10);

    CONTOUR finCt = acAlg.getOptimizedCont();
    //CONTOURS finCts;
    //finCts.push_back(finCt);
    //drawContours(acImg, finCts, 0, Scalar(150), 1);
    for(int i = 0; i < finCt.size(); i++)
    {
        cv::circle(acImg, finCt[i], 1, Scalar(150), cv::FILLED);
    }
    imwrite("finRstAC.png", acImg);
    
    eyePg = finCt;
}

void ForgeEyesMask(const Mat& srcImage, // add this variable just for debugging
                   const FaceInfo& faceInfo,
                   const FaceSegRst& segRst, //Rst: result,
                   Mat& outMask)
{
    POLYGON leftEyePg, rightEyePg;

    Size srcImgS = faceInfo.srcImgS;
    
    ForgeEyePgBySnakeAlg(faceInfo.srcImgS, faceInfo.lEyeRefinePts,
                         segRst.lEyeMaskNOS, segRst.lEyeFPs,
                         segRst.lEyeSegCP, leftEyePg);
    
    /*
    ForgeEyePgBySnakeAlg(faceInfo.srcImgS, faceInfo.rEyeRefinePts,
                         segRst.rEyeMaskNOS, segRst.rEyeFPs,
                         segRst.rEyeSegCP, rightEyePg);
    */
    
    POLYGON_GROUP polygonGroup;
    polygonGroup.push_back(leftEyePg);
    polygonGroup.push_back(rightEyePg);
    
    Mat outOrigMask = ContourGroup2Mask(srcImgS.width, srcImgS.height, polygonGroup);
    
    int dila_size = segRst.faceBBox.width * 0.005;
    Mat element = getStructuringElement(MORPH_ELLIPSE,
                           Size(2*dila_size + 1, 2*dila_size+1),
                           Point(dila_size, dila_size));
    
    cv::Mat outExpandedMask(srcImgS.height, srcImgS.width, CV_8UC1, cv::Scalar(0));
    dilate(outOrigMask, outMask, element);
    
    //outMask = ContourGroup2Mask(faceInfo.srcImgS, polygonGroup);
}
