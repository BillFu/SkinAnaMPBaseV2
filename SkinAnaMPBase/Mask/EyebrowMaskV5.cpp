//
//  EyebrowMaskV5.cpp

/*******************************************************************************
Author: Fu Xiaoqiang
Date:   2022/10/24

********************************************************************************/

#include "EyebrowMaskV5.hpp"
#include "../BSpline/ParametricBSpline.hpp"
#include "FundamentalMask.hpp"
#include "Geometry.hpp"
#include "Common.hpp"
#include "PolarPtSeq.hpp"


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
    CloseSmoothPolygon(coarsePg, csNumPoint, browPg);
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
// (dx, dy) = kpCP - segCP
Point2i TransEyeRefPt2SegSpace(const Point2i& eyeRefPt, int dx, int dy)
{
    return Point2i(eyeRefPt.x - dx, eyeRefPt.y - dy);
}

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
/**********************************************************************************************
only cover the two eyes
***********************************************************************************************/
/*
void ForgeEyePgBySegRst(Size srcImgS, const SegMask& eyeSegMask,
                        const Point2i& eyeCP,
                        POLYGON& eyePg)
{
    CONTOURS contours;
    findContours(eyeSegMask.mask, contours,
                 cv::noArray(), RETR_EXTERNAL, CHAIN_APPROX_NONE);  //CHAIN_APPROX_SIMPLE);
    
    cout << "The Number of Points on the contour of eye: "
        << contours[0].size() << endl;
    
    float scaleUpX = (float)srcImgS.width / SEG_NET_OUTPUT_SIZE;
    float scaleUpY = (float)srcImgS.height / SEG_NET_OUTPUT_SIZE;
    CONTOUR spCt;
    transCt_NOS2SS(contours[0], eyeSegMask.bbox.tl(),
                    scaleUpX, scaleUpY, spCt);
    
    PolarContour rawPolarSeq;
    CalcPolarSeqOnCt(spCt, eyeCP, rawPolarSeq);
    
    PolarContour evenPolarSeq;
    BuildEvenPolarSeq(rawPolarSeq,
                      144, // how many intervals from 0 to 2*Pi
                      evenPolarSeq);
    
    // then smoothing the evenly interpolated polar pt seq.
    PolarContour smoothPolarSeq;
    SmoothPolarPtSeq(evenPolarSeq, 7, smoothPolarSeq);
    
    PolarPtSeq2CartPtSeq(smoothPolarSeq, eyePg);
}
*/

// 用分割的结果构造出一只眼睛的轮廓多边形
void ForgeEyePgBySegRstV2(Size srcImgS,
                          const SegMask& eyeSegMask,
                          const SegEyeFPsNOS& eyeFPsNOS,
                          POLYGON& eyePg)
{
    CONTOURS contours;
    findContours(eyeSegMask.mask, contours,
                 cv::noArray(), RETR_EXTERNAL, CHAIN_APPROX_NONE);  //CHAIN_APPROX_SIMPLE);
    
    cout << "The Number of Points on the contour of eye: "
        << contours[0].size() << endl;
    
    CONTOUR nosEyeCt;
    transCt_SMS2NOS(contours[0], eyeSegMask.bbox.tl(),
                    nosEyeCt);

    // 先判断出轮廓点的自然顺序是顺时针还是逆时针
    CLOCK_DIR scanDir;
    JudgeEyeCtNOSMoveDir(nosEyeCt, eyeFPsNOS, scanDir);
    
    // 经过测试，左右眼的轮廓序列都是顺时针，但愿这个认知具有普遍性
    // 切分为上弧线、下弧线
    CONTOUR upEyeCurve, lowEyeCurve;
    SplitEyeCt2UpLowCurves(nosEyeCt,
                           eyeFPsNOS.lCorPtNOS,
                           eyeFPsNOS.rCorPtNOS,
                           upEyeCurve, lowEyeCurve);
}

// 按轮廓序列的自然顺序遍历轮廓点，先找到上弧线中点，而后继续遍历，看先碰到左角点还是右角点，
// 以此来判断出我们的遍历是顺时针，还是逆时针。
// 真个流程最多需要遍历两次。按照抽象出来的环形数据结构方式来遍历。
void JudgeEyeCtNOSMoveDir(const CONTOUR& nosEyeCont,
                          const SegEyeFPsNOS& eyeFPsNOS,
                          CLOCK_DIR& scanDir)
{
    int mTopPtIdx = -1;
    int numPts = (int)(nosEyeCont.size());
    // 存在一个问题，mTopPt一定会存在于轮廓序列中吗？
    for(int i=0; i<numPts; i++)
    {
        if(nosEyeCont[i] == eyeFPsNOS.mTopPtNOS)
        {
            mTopPtIdx = i;
            break;
        }
    }
    
    if(mTopPtIdx == -1)
    {
        cout << "Top middle point on the upper curve of eye Not FOUND!" << endl;
        return;
    }
    
    bool rCorFound = false;
    bool lCorFound = false;
    // scan again, starting from the following point behind the top middle point on the upper curve
    for(int i=mTopPtIdx+1; i< mTopPtIdx+1+numPts; i++)
    {
        int actIndex = i % numPts; // act: actual
        if(nosEyeCont[actIndex] == eyeFPsNOS.rCorPtNOS)
        {
            rCorFound = true;
            break;
        }
        else if(nosEyeCont[actIndex] == eyeFPsNOS.lCorPtNOS)
        {
            lCorFound = true;
            break;
        }
    }
    
    if(lCorFound)
        scanDir = CCLOCK_WISE;
    else
        scanDir = CLOCK_WISE;
}

void SplitEyeCt2UpLowCurves(const CONTOUR& nosEyeCont,
                            const Point2i& nosLCorPt,
                            const Point2i& nosRCorPt,
                            CONTOUR& upEyeCurve,
                            CONTOUR& lowEyeCurve)
{
    // firstly, we need search for the left corner point
    int leftCorIdx = -1;
    int numPts = (int)(nosEyeCont.size());
    for(int i=0; i<numPts; i++)
    {
        if(nosEyeCont[i] == nosLCorPt)
        {
            leftCorIdx = i;
            break;
        }
    }
    
    // secondly, collect the points on the upper curve
    int rCorIdx = -1;
    for(int i=leftCorIdx; i< leftCorIdx+numPts; i++)
    {
        int actIndex = i % numPts; // act: actual
        upEyeCurve.push_back(nosEyeCont[actIndex]);
        
        if(nosEyeCont[actIndex] == nosRCorPt)
        {
            rCorIdx = actIndex;
            break;
        }
    }
    
    // thirdly, caollect the points on the lower curve
    for(int i=rCorIdx; i< rCorIdx+numPts; i++)
    {
        int actIndex = i % numPts; // act: actual
        lowEyeCurve.push_back(nosEyeCont[actIndex]);
        
        if(nosEyeCont[actIndex] == nosLCorPt)
        {
            break;
        }
    }
    
}

void ForgeEyesMask(const FaceInfo& faceInfo,
                   const FaceSegRst& segRst, //Rst: result,
                   Mat& outMask)
{
    POLYGON leftEyePg, rightEyePg;

    ForgeEyePgBySegRstV2(faceInfo.srcImgS, segRst.lEyeMaskNOS, segRst.lEyeFPsNOS,
                         leftEyePg);
    
    ForgeEyePgBySegRstV2(faceInfo.srcImgS, segRst.rEyeMaskNOS, segRst.rEyeFPsNOS,
                         rightEyePg);
    
    POLYGON_GROUP polygonGroup;
    polygonGroup.push_back(leftEyePg);
    polygonGroup.push_back(rightEyePg);
    
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
