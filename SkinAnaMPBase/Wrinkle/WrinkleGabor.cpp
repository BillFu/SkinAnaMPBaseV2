#include "WrinkleGabor.h"
#include "../Utils.hpp"
#include "../ImgProc.h"
#include <algorithm>


//-----------------------------------------------------------
void AnnoPointsOnImg(Mat& annoImage,
                         const SPLINE& pts,
                         int ptIDs[], int numPt)
{
    cv::Scalar yellow(0, 255, 255); // (B, G, R)
    cv::Scalar blue(255, 0, 0);
    
    for(int i = 0; i < numPt; i++)
    {
        int ptIdx = ptIDs[i];
        
        Point2i pt = pts[ptIdx];
        cv::circle(annoImage, pt, 5, blue, cv::FILLED);
        cv::putText(annoImage, to_string(ptIdx), pt,
                        FONT_HERSHEY_SIMPLEX, 0.5, blue, 1);
    }
}

void AnnoPointsOnImg(Mat& annoImage,
                    const SPLINE& pts)
{
    cv::Scalar yellow(0, 255, 255); // (B, G, R)
    cv::Scalar blue(255, 0, 0);
    
    for(int i = 0; i < pts.size(); i++)
    {
        Point2i pt = pts[i];
        cv::circle(annoImage, pt, 5, blue, cv::FILLED);
        cv::putText(annoImage, to_string(i), pt,
                        FONT_HERSHEY_SIMPLEX, 0.5, blue, 1);
    }
}

/////////////////////////////////////////////////////////////////////////////////
// agg: aggregated
void ApplyGaborBank(const GaborOptBank& gBank, const Mat& inGrFtImg,
                    Mat& aggGabMapFt)
{
    vector<Mat> respMapBank;
    int numBank = static_cast<int>(gBank.size());
#pragma omp parallel sections num_threads(numBank)
    {
#pragma omp for
        for(int i=0; i<numBank; i++)
        {
            Mat respMap;
            doGaborFilter(inGrFtImg, gBank[i], respMap);
            respMapBank.push_back(respMap);
        }
    }
    
    cv::Mat regRespMap(inGrFtImg.size(), CV_32FC1, cv::Scalar(-9999.9));
    for(int i=0; i<numBank; i++)
    {
        cv::max(regRespMap, respMapBank[i], regRespMap);
    }
    
    aggGabMapFt = regRespMap;
}

//-----------------------------------------------------------
void BuildGabOptsForEb(bool isLeftEye, GaborOptBank& gOptBank)
{
    int kerSize = 21;

    // use the left eye as the reference
    float leftThetaSet[] = {73.125, 50.625, 61.875, 84.375, 95.625};
    int numTheta = sizeof(leftThetaSet) / sizeof(float);
    if(isLeftEye)
    {
        for(int i=0; i<numTheta; i++)
        {
            GaborOpt opt(kerSize, 80, 8, 38, leftThetaSet[i], 180);
            gOptBank.push_back(opt);
        }
    }
    else // right eye
    {
        for(int i=0; i<numTheta; i++)
        {
            GaborOpt opt(kerSize, 80, 8, 38, 180.0 - leftThetaSet[i], 180);
            gOptBank.push_back(opt);
        }
    }
}

Mat CcGabMapInOneEyebag(const Mat& grFtSrcImg,
                        bool isLeftEye, const Rect& ebRect)
{
    Mat inGrFtImg = grFtSrcImg(ebRect);
    
    GaborOptBank gOptBank;
    BuildGabOptsForEb(isLeftEye, gOptBank);
    
    Mat aggGabMapFt;
    ApplyGaborBank(gOptBank, inGrFtImg, aggGabMapFt);
    Mat aggGabMap8U = CvtFtImgTo8U_NoNega(aggGabMapFt);
    return aggGabMap8U;
}

//-----------------------------------------------------------
void BuildGabOptsForCF(bool isLeftEye, GaborOptBank& gOptBank)
{
    int kerSize = 21;

    // use the left eye as the reference
    float leftThetaSet[] = {90, 78.75, 67.60, 101.25, 112.50};
    int numTheta = sizeof(leftThetaSet) / sizeof(float);
    if(isLeftEye)
    {
        for(int i=0; i<numTheta; i++)
        {
            GaborOpt opt(kerSize, 50, 8, 41, leftThetaSet[i], 125);
            gOptBank.push_back(opt);
        }
    }
    else // right eye
    {
        for(int i=0; i<numTheta; i++)
        {
            GaborOpt opt(kerSize, 50, 8, 41, 180.0 - leftThetaSet[i], 125);
            gOptBank.push_back(opt);
        }
    }
}

// 鱼尾纹
Mat CcGabMapInOneCrowFeet(const Mat& grFtSrcImg,
                        bool isLeftEye, const Rect& cfRect)
{
    Mat inGrFtImg = grFtSrcImg(cfRect);
    
    GaborOptBank gOptBank;
    BuildGabOptsForCF(isLeftEye, gOptBank);
    
    Mat aggGabMapFt;
    ApplyGaborBank(gOptBank, inGrFtImg, aggGabMapFt);
    Mat aggGabMap8U = CvtFtImgTo8U_NoNega(aggGabMapFt);
    return aggGabMap8U;
}

//----------------------------------------------------------------------
// forehead，前额
Mat CcGaborMapOnFh(const Mat& grFtSrcImg,
                   const Rect& fhRect)
{
    Mat inGrFtImg = grFtSrcImg(fhRect);
    
    GaborOptBank gOptBank;
    int kerSize = 21;
    // use the left eye as the reference
    float fhThetaSet[] = {73.125, 50.625, 61.875, 84.375, 95.625};
    int numTheta = sizeof(fhThetaSet) / sizeof(float);
    for(int i=0; i<numTheta; i++)
    {
        GaborOpt opt(kerSize, 57, 8, 42, fhThetaSet[i], 105);
        gOptBank.push_back(opt);
    }
    
    Mat aggFhMapFt;
    ApplyGaborBank(gOptBank, inGrFtImg, aggFhMapFt);
    Mat aggMap8U = CvtFtImgTo8U_NoNega(aggFhMapFt);
    return aggMap8U;
}

// glabella，眉间，印堂
Mat CcGaborMapOnGlab(const Mat& grFtSrcImg,
                     const Rect& glabRect)
{
    Mat inGrFtImg = grFtSrcImg(glabRect);
        
    GaborOptBank gOptBank;
    int kerSize = 21;
    // use the left eye as the reference
    float glabThetaSet[] = {163.25, 152, 140.75, 174.50, 185.75};
    int numTheta = sizeof(glabThetaSet) / sizeof(float);
    
    for(int i=0; i<numTheta; i++)
    {
        GaborOpt opt(kerSize, 56, 8, 43, glabThetaSet[i], 126);
        gOptBank.push_back(opt);
    }
    
    Mat aggGlabMapFt;
    ApplyGaborBank(gOptBank, inGrFtImg, aggGlabMapFt);
    Mat aggMap8U = CvtFtImgTo8U_NoNega(aggGlabMapFt);
    return aggMap8U;
}

// 返回一个面颊区域的Gabor滤波响应值
/*
Mat CalcGaborRespInOneCheek(const vector<CvGabor*>& gaborBank,
                  const Mat& grSrcImg,
                  const Rect& cheekRect)
{
    Mat imgInCheekRect = grSrcImg(cheekRect);
    Mat lRegResp = ApplyGaborFilter(gaborBank, imgInCheekRect);

    return lRegResp;
}
*/


///////////////////////////////////////////////////////////////////////////////////////////////
/////output variable
void ExtLightWrk(const Mat& wrkGaborRespMap,
                     const Mat& wrkMaskInFR, // wrinkle mask cropped by face rectangle
                     const Rect& faceRect,
                     int minLenOfWrk,
                     int longWrkThresh,
                     CONTOURS& LightWrkConts,
                     CONTOURS& LongWrkConts,
                     int& totalWrkLen)
{
    cv::Mat lightWrkBi(faceRect.size(), CV_8UC1, cv::Scalar(0));

    //************************ light wrinkle *******************
    // light_wr_binary : binary version of the light wrinkle response map
    int ltWrkGaborRespTh = 30;
    threshold(wrkGaborRespMap, lightWrkBi, ltWrkGaborRespTh, 255, THRESH_BINARY);

    lightWrkBi = 255 - lightWrkBi;
    BlackLineThinInBiImg(lightWrkBi.data, lightWrkBi.cols, lightWrkBi.rows);
    lightWrkBi = 255 - lightWrkBi;
    removeBurrs(lightWrkBi, lightWrkBi);
    lightWrkBi = lightWrkBi & wrkMaskInFR;

    CONTOURS contours;
    findContours(lightWrkBi, contours, RETR_EXTERNAL, CHAIN_APPROX_NONE);
    
    std::sort(contours.begin(), contours.end(),
        [](const CONTOUR& a, const CONTOUR& b){return a.size() > b.size();});
    
    cout << "Max Size of contour in ExtLightWrk(): " << contours[0].size() << endl;
    CONTOURS::const_iterator it_c = contours.begin();

    for (unsigned int i = 0; i < contours.size(); ++i)
    {
        if (it_c->size() >= minLenOfWrk ) //&& it_c->size() <= sizeMax)
        {
            LightWrkConts.push_back(contours[i]); // 所有的浅皱纹，包括短的和长的
            totalWrkLen += contours[i].size();
        }
        if (it_c->size() >= longWrkThresh) // && it_c->size() <= sizeMax)
        {
            LongWrkConts.push_back(contours[i]);
        }
        
        it_c++;
    }
    
    // 注意：LongWrkConts中的坐标没有校正到全局坐标系，后面也不会显示出来，仅仅用于数量统计。
    // 将LightWrkConts中的坐标由FaceRect局部坐标系，恢复到输入影像全局坐标系。
    for (unsigned int i = 0; i < LightWrkConts.size(); i++)
    {
        CONTOUR& oneCont = LightWrkConts[i];
        for (unsigned int j = 0; j < LightWrkConts[i].size(); j++)
        {
            oneCont[j].x += faceRect.x;
            oneCont[j].y += faceRect.y;
        }
    }
}

void ExtDeepWrk(const Mat& wrkGaborRespMap,
                    const Mat& wrkMaskInFR,
                    const Rect& faceRect,
                    int minLenOfWrk,
                    int longWrkThresh,
                    CONTOURS& DeepWrkConts,
                    CONTOURS& LongWrkConts)
{
    cv::Mat binaryDeep;
    // 11作为Deep Wrinkle的阈值
    // 4作为Light Wrinkle的阈值
    int dpWrkGaborRespTh = 50;
    cv::threshold(wrkGaborRespMap, binaryDeep, dpWrkGaborRespTh, 255, THRESH_BINARY);
    
    binaryDeep = 255 - binaryDeep;
    BlackLineThinInBiImg(binaryDeep.data, binaryDeep.cols, binaryDeep.rows);
    binaryDeep = 255 - binaryDeep;
    removeBurrs(binaryDeep, binaryDeep);
    binaryDeep = binaryDeep & wrkMaskInFR;
    
    CONTOURS contours;
    cv::findContours(binaryDeep, contours, RETR_EXTERNAL, CHAIN_APPROX_NONE);
    
    std::sort(contours.begin(), contours.end(),
        [](const CONTOUR& a, const CONTOUR& b){return a.size() > b.size();});
    
    cout << "Max Size of contour in ExtDeepWrk(): " << contours[0].size() << endl;
    
    for (CONTOUR cont: contours)
    {
        if (cont.size() >= minLenOfWrk ) // && it_c->size() <= sizeMax )
        {
            DeepWrkConts.push_back(cont);
        }
        if (cont.size() >= longWrkThresh ) // && it_c->size() <= sizeMax )
        {
            LongWrkConts.push_back(cont);
        }
    }
    
    // 将DeepWrkConts中的坐标由FaceRect局部坐标系，恢复到输入影像全局坐标系。
    unsigned long vector_size = DeepWrkConts.size();
    for (unsigned int i = 0; i < vector_size; i++)
    {
        CONTOUR& oneCont = DeepWrkConts[i];
        for (unsigned int j = 0; j < DeepWrkConts[i].size(); j++)
        {
            oneCont[j].x += faceRect.x;
            oneCont[j].y += faceRect.y;
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////
// fine wrinkle： 细皱纹
// WrinkRespMap的大小和在原始影像坐标系中的位置由Face_Rect限定
void CalcGaborMap(const Mat& grSrcImg,
                   WrkRegGroup& wrkRegGroup,
                   Mat& gaborRespMap)
{
    Mat grFtSrcImg;
    grSrcImg.convertTo(grFtSrcImg, CV_32F, 1.0/255, 0);

    //Eb: eyebag
    Mat lEbGabMap8U = CcGabMapInOneEyebag(grFtSrcImg, true, wrkRegGroup.lEyeBagReg.bbox);
    Mat rEbGabMap8U = CcGabMapInOneEyebag(grFtSrcImg, false, wrkRegGroup.rEyeBagReg.bbox);

    // 前额
    Mat fhGabMap8U = CcGaborMapOnFh(grFtSrcImg, wrkRegGroup.fhReg.bbox);
    // glabella，眉间，印堂
    Mat glabMap8U = CcGaborMapOnGlab(grFtSrcImg, wrkRegGroup.glabReg.bbox);

#ifdef TEST_RUN2
    bool isSuccess;
    isSuccess = SaveTestOutImgInDir(lEbGabMap8U,  wrkOutDir,  "lEbGabMap.png");
    isSuccess = SaveTestOutImgInDir(rEbGabMap8U,  wrkOutDir,  "rEbGabMap.png");
    isSuccess = SaveTestOutImgInDir(fhGabMap8U,  wrkOutDir,  "FhGabMap.png");
    isSuccess = SaveTestOutImgInDir(glabMap8U,  wrkOutDir,  "glabMap.png");

    assert(isSuccess);
#endif
    
    Mat lCFGabMap8U = CcGabMapInOneCrowFeet(grFtSrcImg, true, wrkRegGroup.lCrowFeetReg.bbox);
    Mat rCFGabMap8U = CcGabMapInOneCrowFeet(grFtSrcImg, false, wrkRegGroup.rCrowFeetReg.bbox);

    /*
    // 计算左面颊的Gabor滤波响应值
    vector<CvGabor*> lGaborBank;
    InitLCheekGaborBank(lGaborBank);
    Mat lCheekResp = CalcGaborRespInOneCheek(lGaborBank, blurGrImg, wrkRegGroup.lCheekReg.bbox);

    // 计算右面颊的Gabor滤波响应值
    vector<CvGabor*> rGaborBank;
    InitRCheekGaborBank(rGaborBank);
    Mat rCheekResp = CalcGaborRespInOneCheek(rGaborBank, blurGrImg, wrkRegGroup.rCheekReg.bbox);
     */
    
#ifdef TEST_RUN2
    //bool isSuccess;
    
    isSuccess = SaveTestOutImgInDir(lCFGabMap8U,  wrkOutDir,   "lCFGabMap.png");
    isSuccess = SaveTestOutImgInDir(rCFGabMap8U,  wrkOutDir,   "rCFGabMap.png");

    //isSuccess = SaveTestOutImgInDir(lCheekResp, wrkOutDir,  "lCheekGaborResp.png");
    //isSuccess = SaveTestOutImgInDir(rCheekResp, wrkOutDir,  "rCheekGaborResp.png");
    
    /*
    Mat canvas = grFrImg.clone();
    rectangle(canvas, lCheekRect, CV_COLOR_RED, 8, 0);
    rectangle(canvas, rCheekRect, CV_COLOR_GREEN, 8, 0);
    rectangle(canvas, glabeRect, CV_COLOR_BLUE, 8, 0);
    rectangle(canvas, fhRect, CV_COLOR_YELLOW, 8, 0);
    rectangle(canvas, noseRect, CV_COLOR_WHITE, 8, 0);
    
    isSuccess = SaveTestOutImgInDir(canvas, wrk_out_dir, "fiveRects.png");
    canvas.release();
    */

    /*
    CONTOURS LightWrkConts;
    int minLenOfWrk = 200;

    Mat fhGaborMap = Mat(grFrImg.size(), grFrImg.type(), Scalar(0));
    fhRegResp.copyTo(fhGaborMap(fhRect));
    
    ExtWrkInFhGaborResp(fhGaborMap, faceRect, minLenOfWrk, LightWrkConts);
    fhGaborMap.release();
    
    Mat fhWrkImg = drawFhWrk(grFrImg, LightWrkConts);
    isSuccess = SaveTestOutImgInDir(fhWrkImg, wrk_out_dir, "fhWrkImg.png");
    fhWrkImg.release();
    */
#endif
    /*
    gaborRespMap = Mat(grFrImg.size(), grFrImg.type(), Scalar(0));

    // --------把各个小区域的计算结果合并起来，存贮在WrkRespMap------------------------------------
    lCheekResp.copyTo(gaborRespMap(lCheekRect));
    rCheekResp.copyTo(gaborRespMap(rCheekRect));
    fhRegResp.copyTo(gaborRespMap(fhRect));
    
    // nose上部和眉间glabella与其他区域有重叠，故而处理与其他三个相互不重叠区域的处理有所不同。
    cv::Mat noseRegionTemp(grFrImg.size(), CV_8UC1, cv::Scalar(0));
    cv::Mat glabeRegionTemp(grFrImg.size(), CV_8UC1, cv::Scalar(0));
    
    noseRegResp.copyTo(noseRegionTemp(noseRect));
    glabeRegResp.copyTo(glabeRegionTemp(glabeRect));
    
    cv::max(gaborRespMap, glabeRegionTemp, gaborRespMap);
    cv::max(gaborRespMap, noseRegionTemp, gaborRespMap);
    gaborRespMap = gaborRespMap * 10;  //1.2;
    */
    
    /*
    cout << "gaborRespMap data type: " << openCVType2str(gaborRespMap.type()) << endl;
    uchar maxV = *max_element(gaborRespMap.begin<uchar>(), gaborRespMap.end<uchar>());
    uchar minV = *min_element(gaborRespMap.begin<uchar>(), gaborRespMap.end<uchar>());
    
    cout << "maxV in gaborRespMap:"  << (int)maxV << endl;
    cout << "minV in gaborRespMap:"  << (int)minV << endl;
    */
}

void ExtWrkInFhGaborResp(const Mat& fhGaborMap,
                     const Rect& faceRect,
                     int minLenOfWrk,
                     CONTOURS& LightWrkConts)
{
    cv::Mat lightWrkBi(faceRect.size(), CV_8UC1, cv::Scalar(0));

    //************************ light wrinkle *******************
    // light_wr_binary : binary version of the light wrinkle response map
    int ltWrkGaborRespTh = 10;
    threshold(fhGaborMap, lightWrkBi, ltWrkGaborRespTh, 255, THRESH_BINARY);

    Mat se = Mat(3, 80, CV_8UC1);
    dilate(lightWrkBi, lightWrkBi, se);
    //e_im = cv2.erode(d_im, kernel, iterations=1)
    
    lightWrkBi = 255 - lightWrkBi;
    BlackLineThinInBiImg(lightWrkBi.data, lightWrkBi.cols, lightWrkBi.rows);
    lightWrkBi = 255 - lightWrkBi;
    removeBurrs(lightWrkBi, lightWrkBi);
    //lightWrkBi = lightWrkBi & wrkMaskInFR;

    CONTOURS contours;
    findContours(lightWrkBi, contours, RETR_EXTERNAL, CHAIN_APPROX_NONE);
    
    CONTOURS::const_iterator it_c = contours.begin();

    for (unsigned int i = 0; i < contours.size(); ++i)
    {
        if (it_c->size() >= minLenOfWrk ) //&& it_c->size() <= sizeMax)
        {
            LightWrkConts.push_back(contours[i]); // 所有的浅皱纹，包括短的和长的
            //totalWrkLen += contours[i].size();
        }
        
        it_c++;
    }
    
    /*
    // 注意：LongWrkConts中的坐标没有校正到全局坐标系，后面也不会显示出来，仅仅用于数量统计。
    // 将LightWrkConts中的坐标由FaceRect局部坐标系，恢复到输入影像全局坐标系。
    for (unsigned int i = 0; i < LightWrkConts.size(); i++)
    {
        CONTOUR& oneCont = LightWrkConts[i];
        for (unsigned int j = 0; j < LightWrkConts[i].size(); j++)
        {
            oneCont[j].x += faceRect.x;
            oneCont[j].y += faceRect.y;
        }
    }
    */
}

Mat drawFhWrk(const Mat& canvas, const CONTOURS& LightWrkConts)
{
    Mat resImg = canvas.clone();
    
    drawContours(resImg, LightWrkConts, -1, cv::Scalar(255), 2);
    
    return resImg;
}
