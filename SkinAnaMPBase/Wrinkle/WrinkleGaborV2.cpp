
#include <algorithm>

#include "../Utils.hpp"
#include "../ImgProc.h"
#include "../Geometry.hpp"

#include "WrinkleGaborV2.h"

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
void BuildGabOptsForEb(int kerSize, int sigma, bool isLeftEye, GaborOptBank& gOptBank)
{
    // use the left eye as the reference
    float leftThetaSet[] = {73.125, 50.625, 61.875, 84.375, 95.625};
    int numTheta = sizeof(leftThetaSet) / sizeof(float);
    if(isLeftEye)
    {
        for(int i=0; i<numTheta; i++)
        {
            GaborOpt opt(kerSize, 80, sigma, 38, leftThetaSet[i], 180);
            gOptBank.push_back(opt);
        }
    }
    else // right eye
    {
        for(int i=0; i<numTheta; i++)
        {
            GaborOpt opt(kerSize, 80, sigma, 38, 180.0 - leftThetaSet[i], 180);
            gOptBank.push_back(opt);
        }
    }
}

Mat CcGabMapInOneEyebag(const Mat& grFtSrcImg, int kerSize, int sigma,
                        bool isLeftEye, const Rect& ebRect)
{
    Mat inGrFtImg = grFtSrcImg(ebRect);
    
    GaborOptBank gOptBank;
    BuildGabOptsForEb(kerSize, sigma, isLeftEye, gOptBank);
    
    Mat aggGabMapFt;
    ApplyGaborBank(gOptBank, inGrFtImg, aggGabMapFt);
    Mat aggGabMap8U = CvtFtImgTo8U_NoNega(aggGabMapFt);
    return aggGabMap8U;
}

//-----------------------------------------------------------
void BuildGabOptsForCF(int kerSize, int sigma,
                       bool isLeftEye, GaborOptBank& gOptBank)
{
    // use the left eye as the reference
    float leftThetaSet[] = {90, 78.75, 67.60, 101.25, 112.50};
    int numTheta = sizeof(leftThetaSet) / sizeof(float);
    if(isLeftEye)
    {
        for(int i=0; i<numTheta; i++)
        {
            GaborOpt opt(kerSize, 50, sigma, 41, leftThetaSet[i], 125);
            gOptBank.push_back(opt);
        }
    }
    else // right eye
    {
        for(int i=0; i<numTheta; i++)
        {
            GaborOpt opt(kerSize, 50, sigma, 41, 180.0 - leftThetaSet[i], 125);
            gOptBank.push_back(opt);
        }
    }
}

// 鱼尾纹
Mat CcGabMapInOneCrowFeet(const Mat& grFtSrcImg, int kerSize, int sigma,
                        bool isLeftEye, const Rect& cfRect)
{
    Mat inGrFtImg = grFtSrcImg(cfRect);
    
    GaborOptBank gOptBank;
    BuildGabOptsForCF(kerSize, sigma, isLeftEye, gOptBank);
    
    Mat aggGabMapFt;
    ApplyGaborBank(gOptBank, inGrFtImg, aggGabMapFt);
    Mat aggGabMap8U = CvtFtImgTo8U_NoNega(aggGabMapFt);
    return aggGabMap8U;
}

//----------------------------------------------------------------------
// forehead，前额
Mat CcGaborMapOnFh(const Mat& grFtSrcImg, int kerSize, int sigma,
                   const Rect& fhRect)
{
    Mat inGrFtImg = grFtSrcImg(fhRect);
    
    GaborOptBank gOptBank;
    // use the left eye as the reference
    float fhThetaSet[] = {73.125, 50.625, 61.875, 84.375, 95.625};
    int numTheta = sizeof(fhThetaSet) / sizeof(float);
    for(int i=0; i<numTheta; i++)
    {
        GaborOpt opt(kerSize, 57, sigma, 42, fhThetaSet[i], 105);
        gOptBank.push_back(opt);
    }
    
    Mat aggFhMapFt;
    ApplyGaborBank(gOptBank, inGrFtImg, aggFhMapFt);
    Mat aggMap8U = CvtFtImgTo8U_NoNega(aggFhMapFt);
    return aggMap8U;
}

// glabella，眉间，印堂
Mat CcGaborMapOnGlab(const Mat& grFtSrcImg, int kerSize, int sigma,
                     const Rect& glabRect)
{
    Mat inGrFtImg = grFtSrcImg(glabRect);
        
    GaborOptBank gOptBank;
    
    // use the left eye as the reference
    float glabThetaSet[] = {163.25, 152, 140.75, 174.50, 185.75};
    int numTheta = sizeof(glabThetaSet) / sizeof(float);
    
    for(int i=0; i<numTheta; i++)
    {
        GaborOpt opt(kerSize, 56, sigma, 43, glabThetaSet[i], 126);
        gOptBank.push_back(opt);
    }
    
    Mat aggGlabMapFt;
    ApplyGaborBank(gOptBank, inGrFtImg, aggGlabMapFt);
    Mat aggMap8U = CvtFtImgTo8U_NoNega(aggGlabMapFt);
    return aggMap8U;
}

///////////////////////////////////////////////////////////////////////////////////////////////

// 返回一个面颊区域的Gabor滤波响应值
void BuildGabOptsForCheek(int kerSize, int sigma, bool isLeftEye, GaborOptBank& gOptBank)
{
    // use the left eye as the reference
    float rightThetaSet[] = {103, 91.75, 80.0, 114.25, 125.5};
    int numTheta = sizeof(rightThetaSet) / sizeof(float);
    if(isLeftEye)
    {
        for(int i=0; i<numTheta; i++)
        {
            GaborOpt opt(kerSize, 53, sigma, 40, 180 - rightThetaSet[i], 131);
            gOptBank.push_back(opt);
        }
    }
    else // right eye
    {
        for(int i=0; i<numTheta; i++)
        {
            GaborOpt opt(kerSize, 53, sigma, 40, rightThetaSet[i], 131);
            gOptBank.push_back(opt);
        }
    }
}

Mat CcGaborMapInOneCheek(const Mat& grFtSrcImg, int kerSize, int sigma,
                         bool isLeft, const Rect& cheekRect)
{
    Mat inGrFtImg = grFtSrcImg(cheekRect);
    
    GaborOptBank gOptBank;
    BuildGabOptsForCheek(kerSize, sigma, isLeft, gOptBank);
    
    Mat aggGabMapFt;
    ApplyGaborBank(gOptBank, inGrFtImg, aggGabMapFt);
    Mat aggGabMap8U = CvtFtImgTo8U_NoNega(aggGabMapFt);
    return aggGabMap8U;
}

///////////////////////////////////////////////////////////////////////////////////////////////
/////output variable
void ExtLightWrk(const Mat& wrkGaborMap,
                     int minLenOfWrk,
                     int longWrkThresh,
                     CONTOURS& LightWrkConts,
                     CONTOURS& LongWrkConts,
                     int& totalWrkLen)
{
    cv::Mat lightWrkBi(wrkGaborMap.size(), CV_8UC1, cv::Scalar(0));

    //************************ light wrinkle *******************
    // light_wr_binary : binary version of the light wrinkle response map
    int ltWrkGaborRespTh = 30;
    threshold(wrkGaborMap, lightWrkBi, ltWrkGaborRespTh, 255, THRESH_BINARY);

    lightWrkBi = 255 - lightWrkBi;
    BlackLineThinInBiImg(lightWrkBi.data, lightWrkBi.cols, lightWrkBi.rows);
    lightWrkBi = 255 - lightWrkBi;
    removeBurrs(lightWrkBi, lightWrkBi);

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
}

void ExtDeepWrk(const Mat& wrkGaborRespMap,
                    //const Mat& wrkMaskInFR,
                    //const Rect& faceRect,
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
    //binaryDeep = binaryDeep & wrkMaskInFR;
    
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
}

////////////////////////////////////////////////////////////////////////////////////////
// fine wrinkle： 细皱纹
// WrinkRespMap的大小和在原始影像坐标系中的位置由Face_Rect限定
void CalcGaborMap(const Mat& grSrcImg, // in Global Source Space
                  WrkRegGroup& wrkRegGroup,
                  Mat& gaborMap,
                  Mat& fhGabMap8U,
                  Mat& glabGabMap8U)
{
    Mat grFtSrcImg;
    grSrcImg.convertTo(grFtSrcImg, CV_32F, 1.0/255, 0);

    int kerSize = int(21.0 * grSrcImg.cols / 2448.0 + 0.5);
    if(kerSize % 2 == 0)
        kerSize++;
    int sigma = int(8.8 * grSrcImg.cols / 2448.0 + 0.5);
    
    //Eb: eyebag
    Mat lEbGabMap8U = CcGabMapInOneEyebag(grFtSrcImg, kerSize, sigma,
                                          true, wrkRegGroup.lEyeBagReg.bbox);
    Mat rEbGabMap8U = CcGabMapInOneEyebag(grFtSrcImg, kerSize, sigma,
                                          false, wrkRegGroup.rEyeBagReg.bbox);

    // 前额
    fhGabMap8U = CcGaborMapOnFh(grFtSrcImg, kerSize, sigma, wrkRegGroup.fhReg.bbox);
    // glabella，眉间，印堂
    glabGabMap8U = CcGaborMapOnGlab(grFtSrcImg, kerSize, sigma, wrkRegGroup.glabReg.bbox);
    
    Mat lCFGabMap8U = CcGabMapInOneCrowFeet(grFtSrcImg, kerSize, sigma,
                                            true, wrkRegGroup.lCrowFeetReg.bbox);
    Mat rCFGabMap8U = CcGabMapInOneCrowFeet(grFtSrcImg, kerSize, sigma,
                                            false, wrkRegGroup.rCrowFeetReg.bbox);

    Mat lChkGabMap8U = CcGaborMapInOneCheek(grFtSrcImg, kerSize, sigma,
                                            true, wrkRegGroup.lCheekReg.bbox);
    Mat rChkGabMap8U = CcGaborMapInOneCheek(grFtSrcImg, kerSize, sigma,
                                            false, wrkRegGroup.rCheekReg.bbox);
    
#ifdef TEST_RUN2
    bool isSuccess;
    isSuccess = SaveTestOutImgInDir(lEbGabMap8U,  wrkOutDir,  "lEbGabMap.png");
    isSuccess = SaveTestOutImgInDir(rEbGabMap8U,  wrkOutDir,  "rEbGabMap.png");
    isSuccess = SaveTestOutImgInDir(fhGabMap8U,  wrkOutDir,  "FhGabMap.png");
    isSuccess = SaveTestOutImgInDir(glabGabMap8U,  wrkOutDir,  "glabGabMap.png");

    isSuccess = SaveTestOutImgInDir(lCFGabMap8U,  wrkOutDir,   "lCFGabMap.png");
    isSuccess = SaveTestOutImgInDir(rCFGabMap8U,  wrkOutDir,   "rCFGabMap.png");

    isSuccess = SaveTestOutImgInDir(lChkGabMap8U, wrkOutDir,  "lChkGabMap.png");
    isSuccess = SaveTestOutImgInDir(rChkGabMap8U, wrkOutDir,  "rChkGabMap.png");
    
#endif
    
    // --------把各个小区域的计算结果合并起来，存贮在WrkRespMap------------------------------------
    gaborMap = Mat(grSrcImg.size(), CV_8UC1, Scalar(0));

    lEbGabMap8U.copyTo(gaborMap(wrkRegGroup.lEyeBagReg.bbox));
    rEbGabMap8U.copyTo(gaborMap(wrkRegGroup.rEyeBagReg.bbox));
    lCFGabMap8U.copyTo(gaborMap(wrkRegGroup.lCrowFeetReg.bbox));
    rCFGabMap8U.copyTo(gaborMap(wrkRegGroup.rCrowFeetReg.bbox));
    //lChkGabMap8U.copyTo(gaborMap(wrkRegGroup.lCheekReg.bbox));
    //rChkGabMap8U.copyTo(gaborMap(wrkRegGroup.rCheekReg.bbox));
    fhGabMap8U.copyTo(gaborMap(wrkRegGroup.fhReg.bbox));
    
    // 眉间glabella与其他区域有重叠，故而处理与其他相互不重叠区域的处理有所不同。
    Mat glabeRegTemp(grSrcImg.size(), CV_8UC1, cv::Scalar(0));
    glabGabMap8U.copyTo(glabeRegTemp(wrkRegGroup.glabReg.bbox));
    
    cv::max(gaborMap, glabeRegTemp, gaborMap);
    
    /*
    cout << "gaborRespMap data type: " << openCVType2str(gaborRespMap.type()) << endl;
    uchar maxV = *max_element(gaborRespMap.begin<uchar>(), gaborRespMap.end<uchar>());
    uchar minV = *min_element(gaborRespMap.begin<uchar>(), gaborRespMap.end<uchar>());
    
    cout << "maxV in gaborRespMap:"  << (int)maxV << endl;
    cout << "minV in gaborRespMap:"  << (int)minV << endl;
    */
}

Mat drawFhWrk(const Mat& canvas, const CONTOURS& LightWrkConts)
{
    Mat resImg = canvas.clone();
    
    drawContours(resImg, LightWrkConts, -1, cv::Scalar(255), 2);
    
    return resImg;
}

////////////////////////////////////////////////////////////////////////////////////////
void ExtWrkFromFhGabMap(const Rect& fhRect,
                        const Mat& fhGabMap8U,
                        int minLenOfWrk,
                        int longWrkThresh,
                        CONTOURS& DeepWrkConts,
                        CONTOURS& LongWrkConts)
{
    cv::Mat biMap;
    int dpWrkBiTh = 90;
    cv::threshold(fhGabMap8U, biMap, dpWrkBiTh, 255, THRESH_BINARY);
    
#ifdef TEST_RUN2
    string fhGabBiMapFile = wrkOutDir + "/fhGabBiMap.png";
    imwrite(fhGabBiMapFile.c_str(), biMap);
#endif
    
    chao_thinimage(biMap); //单通道、二值化后的图像
#ifdef TEST_RUN2
    string fhThinGabBiFile = wrkOutDir + "/fhThinGabBiMap.png";
    imwrite(fhThinGabBiFile.c_str(), biMap);
#endif
    
    CONTOURS contours;
    cv::findContours(biMap, contours, RETR_EXTERNAL, CHAIN_APPROX_NONE);
    
    std::sort(contours.begin(), contours.end(),
        [](const CONTOUR& a, const CONTOUR& b){return a.size() > b.size();});
    
    cout << "Max Size of contour in fhGabMap(): " << contours[0].size() << endl;
    
    Point2i tlPt = fhRect.tl();
    for (CONTOUR cont: contours)
    {
        if (cont.size() >= minLenOfWrk ) // && it_c->size() <= sizeMax )
        {
            CONTOUR gsCt;
            transCt_LSS2GS(cont, tlPt, gsCt);
            DeepWrkConts.push_back(gsCt);
        }
        if (cont.size() >= longWrkThresh ) // && it_c->size() <= sizeMax )
        {
            CONTOUR gsCt;
            transCt_LSS2GS(cont, tlPt, gsCt);
            LongWrkConts.push_back(gsCt);
        }
    }
}

void ExtWrkFromGlabGabMap(const Rect& glabRect,
                        const Mat& glabGabMap8U,
                        int minLenOfWrk,
                        int longWrkThresh,
                        CONTOURS& DeepWrkConts,
                        CONTOURS& LongWrkConts)
{
    cv::Mat biMap;
    int dpWrkBiTh = 90;
    cv::threshold(glabGabMap8U, biMap, dpWrkBiTh, 255, THRESH_BINARY);
    
#ifdef TEST_RUN2
    string glabGabBiFile = wrkOutDir + "/glabGabBi.png";
    imwrite(glabGabBiFile.c_str(), biMap);
#endif
    
    chao_thinimage(biMap); //单通道、二值化后的图像
#ifdef TEST_RUN2
    string glabThinGabBiFile = wrkOutDir + "/glabThinGabBi.png";
    imwrite(glabThinGabBiFile.c_str(), biMap);
#endif
    
    CONTOURS contours;
    cv::findContours(biMap, contours, RETR_EXTERNAL, CHAIN_APPROX_NONE);
    
    std::sort(contours.begin(), contours.end(),
        [](const CONTOUR& a, const CONTOUR& b){return a.size() > b.size();});
    
    cout << "Max Size of contour in GlabGabMap: " << contours[0].size() << endl;
    
    Point2i tlPt = glabRect.tl();
    for (CONTOUR cont: contours)
    {
        if (cont.size() >= minLenOfWrk ) // && it_c->size() <= sizeMax )
        {
            CONTOUR gsCt;
            transCt_LSS2GS(cont, tlPt, gsCt);
            DeepWrkConts.push_back(gsCt);
        }
        if (cont.size() >= longWrkThresh ) // && it_c->size() <= sizeMax )
        {
            CONTOUR gsCt;
            transCt_LSS2GS(cont, tlPt, gsCt);
            LongWrkConts.push_back(gsCt);
        }
    }
}
