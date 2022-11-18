
#include <algorithm>

#include "../Utils.hpp"
#include "../ImgProc.h"
#include "../Geometry.hpp"

#include "WrinkleGaborV3.h"
#include "WrinkleV2.hpp"

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

Mat CcGaborMapOnFhV2(const Mat& grSrcImg, int kerSize, int sigma,
                   const Rect& fhRect)
{
    Mat inGrImg = grSrcImg(fhRect);
    Mat enhImg;
    PreprocGrImg(inGrImg, enhImg);
    
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
    ApplyGaborBank(gOptBank, enhImg, aggFhMapFt);
    Mat aggMap8U = CvtFtImgTo8U_Special(aggFhMapFt);
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

void BuildGabOptsForNagv(int kerSize, int sigma, bool isLeft, GaborOptBank& gOptBank)
{
    // use the left eye as the reference
    float leftThetaSet[] = {22, 11, 33};
    int numTheta = sizeof(leftThetaSet) / sizeof(float);
    if(isLeft)
    {
        
        for(int i=0; i<numTheta; i++)
        {
            GaborOpt opt(kerSize, 54, sigma, 42, leftThetaSet[i], 125);
            gOptBank.push_back(opt);
        }
    }
    else // right eye
    {
        for(int i=0; i<numTheta; i++)
        {
            GaborOpt opt(kerSize, 54, sigma, 42, 180 - leftThetaSet[i], 125);
            gOptBank.push_back(opt);
        }
    }
}

Mat CcGabMapInOneNagv(bool isLeft,
                      const Mat& grFtSrcImg, int kerSize, int sigma,
                          const Rect& nagvRect)
{
    Mat inGrFtImg = grFtSrcImg(nagvRect);
    
    GaborOptBank gOptBank;
    BuildGabOptsForNagv(kerSize, sigma, isLeft, gOptBank);
    
    Mat aggGabMapFt;
    ApplyGaborBank(gOptBank, inGrFtImg, aggGabMapFt);
    Mat aggGabMap8U = CvtFtImgTo8U_Special(aggGabMapFt);
    return aggGabMap8U;
}

///////////////////////////////////////////////////////////////////////////////////////////////

void BuildGabOptsForCirEye(int kerSize, int sigma, bool isLeft, GaborOptBank& gOptBank)
{
    // use the left eye as the reference
    float leftThetaSet[] = {100, 111, 84, 29, 40, 18};
    int numTheta = sizeof(leftThetaSet) / sizeof(float);
    if(isLeft)
    {
        for(int i=0; i<numTheta; i++)
        {
            GaborOpt opt(kerSize, 54, sigma, 42, leftThetaSet[i], 125);
            gOptBank.push_back(opt);
        }
    }
    else // right eye
    {
        for(int i=0; i<numTheta; i++)
        {
            GaborOpt opt(kerSize, 54, sigma, 42, 180 - leftThetaSet[i], 125);
            gOptBank.push_back(opt);
        }
    }
}

Mat CcGabMapInOneCirEye(bool isLeft, const Mat& grSrcImg,
                      int kerSize, int sigma,
                      const DetectRegion& cirEyeReg)
{
    Mat inGrImg = grSrcImg(cirEyeReg.bbox);
    
    Mat inGrFtImg;
    inGrImg.convertTo(inGrFtImg, CV_32F, 1.0/255, 0);

    GaborOptBank gOptBank;
    BuildGabOptsForCirEye(kerSize, sigma, isLeft, gOptBank);
    
    Mat aggGabMapFt;
    ApplyGaborBank(gOptBank, inGrFtImg, aggGabMapFt);
    Mat aggGabMap8U = CvtFtImgTo8U_Special(aggGabMapFt);
    return aggGabMap8U;
}
////////////////////////////////////////////////////////////////////////////////////////
// fine wrinkle： 细皱纹
// WrinkRespMap的大小和在原始影像坐标系中的位置由Face_Rect限定
void CalcGaborMap(const Mat& grSrcImg, // in Global Source Space
                  WrkRegGroup& wrkRegGroup,
                  Mat& fhGabMap8U,
                  Mat& glabGabMap8U,
                  Mat& lCirEyeGabMap8U,
                  Mat& rCirEyeGabMap8U,
                  Mat& lNagvGabMap8U,
                  Mat& rNagvGabMap8U)
{
    Mat grFtSrcImg;
    grSrcImg.convertTo(grFtSrcImg, CV_32F, 1.0/255, 0);

    int kerSize = int(21.0 * grSrcImg.cols / 2448.0 + 0.5);
    if(kerSize % 2 == 0)
        kerSize++;
    int sigma = int(8 * grSrcImg.cols / 2448.0 + 0.5);
    
    // 前额
    fhGabMap8U = CcGaborMapOnFh(grFtSrcImg, kerSize, sigma, wrkRegGroup.fhReg.bbox);
    // glabella，眉间，印堂
    glabGabMap8U = CcGaborMapOnGlab(grSrcImg, kerSize, sigma, wrkRegGroup.glabReg.bbox);
    
    lNagvGabMap8U = CcGabMapInOneNagv(true, grFtSrcImg, kerSize, sigma,
                                            wrkRegGroup.lNagvReg.bbox);

    rNagvGabMap8U = CcGabMapInOneNagv(false, grFtSrcImg, kerSize, sigma,
                                            wrkRegGroup.rNagvReg.bbox);
    
    lCirEyeGabMap8U = CcGabMapInOneCirEye(true, grSrcImg, kerSize, sigma,
                                            wrkRegGroup.lCirEyeReg);

    rCirEyeGabMap8U = CcGabMapInOneCirEye(false, grSrcImg, kerSize, sigma,
                                            wrkRegGroup.rCirEyeReg);
    
#ifdef TEST_RUN2
    bool isSuccess;
    isSuccess = SaveTestOutImgInDir(fhGabMap8U,  wrkOutDir,   "FhGabMap.png");
    isSuccess = SaveTestOutImgInDir(glabGabMap8U,  wrkOutDir,  "glabGabMap.png");

    isSuccess = SaveTestOutImgInDir(lNagvGabMap8U, wrkOutDir,  "lNagvGabMap.png");
    isSuccess = SaveTestOutImgInDir(rNagvGabMap8U, wrkOutDir,  "rNagvGabMap.png");
    
    isSuccess = SaveTestOutImgInDir(lCirEyeGabMap8U, wrkOutDir,  "lCirGabMap.png");
    isSuccess = SaveTestOutImgInDir(rCirEyeGabMap8U, wrkOutDir,  "rCirGabMap.png");
    
#endif
}

Mat drawFhWrk(const Mat& canvas, const CONTOURS& LightWrkConts)
{
    Mat resImg = canvas.clone();
    
    drawContours(resImg, LightWrkConts, -1, cv::Scalar(255), 2);
    
    return resImg;
}

////////////////////////////////////////////////////////////////////////////////////////
void ExtWrkFromFhGabMap(const DetectRegion& fhReg,
                        const Mat& fhGabMap8U,
                        int minWrkTh,
                        int longWrkThresh,
                        CONTOURS& DeepWrkConts,
                        CONTOURS& LongWrkConts)
{
    cv::Mat biMap;
    biMap = fhGabMap8U & fhReg.mask;
    int dpWrkBiTh = 40;
    cv::threshold(biMap, biMap, dpWrkBiTh, 255, THRESH_BINARY);
    
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
        
    Point2i tlPt = fhReg.bbox.tl();
    for (CONTOUR cont: contours)
    {
        if (cont.size() >= minWrkTh ) // && it_c->size() <= sizeMax )
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

void ExtWrkFromGlabGabMap(const DetectRegion& glabReg,
                        const Mat& glabGabMap8U,
                        int minWrkTh,
                        int longWrkThresh,
                        CONTOURS& DeepWrkConts,
                        CONTOURS& LongWrkConts)
{
    cv::Mat biMap;
    biMap = glabGabMap8U & glabReg.mask;
    int dpWrkBiTh = 120;
    cv::threshold(biMap, biMap, dpWrkBiTh, 255, THRESH_BINARY);
    
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
    
    Point2i tlPt = glabReg.bbox.tl();
    for (CONTOUR cont: contours)
    {
        if (cont.size() >= minWrkTh ) // && it_c->size() <= sizeMax )
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

void ExtWrkInNagvGabMap(bool isLeft,
                        const DetectRegion& nagvReg,
                        const Mat& nagvGabMap8U,
                        int minWrkTh,
                        int longWrkThresh,
                        CONTOURS& DeepWrkConts,
                        CONTOURS& LongWrkConts)
{
    cv::Mat biMap;
    int dpWrkBiTh = 15;
    biMap = nagvGabMap8U & nagvReg.mask;
    cv::threshold(biMap, biMap, dpWrkBiTh, 255, THRESH_BINARY);
    
#ifdef TEST_RUN2
    if(isLeft)
    {
        string GabBiFile = wrkOutDir + "/lnagvGabBi.png";
        imwrite(GabBiFile.c_str(), biMap);
    }
    else
    {
        string GabBiFile = wrkOutDir + "/rnagvGabBi.png";
        imwrite(GabBiFile.c_str(), biMap);
    }
#endif
    
    chao_thinimage(biMap); //单通道、二值化后的图像
#ifdef TEST_RUN2
    if(isLeft)
    {
        string ThinGabBiFile = wrkOutDir + "/lNagvThinGabBi.png";
        imwrite(ThinGabBiFile.c_str(), biMap);
    }
    else
    {
        string ThinGabBiFile = wrkOutDir + "/rNagvThinGabBi.png";
        imwrite(ThinGabBiFile.c_str(), biMap);
    }
#endif
    
    CONTOURS contours;
    cv::findContours(biMap, contours, RETR_EXTERNAL, CHAIN_APPROX_NONE);
    
    std::sort(contours.begin(), contours.end(),
        [](const CONTOUR& a, const CONTOUR& b){return a.size() > b.size();});
        
    Point2i tlPt = nagvReg.bbox.tl();
    for (CONTOUR cont: contours)
    {
        if (cont.size() >= minWrkTh)
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

void ExtWrkInCirEyeGabMap(bool isLeft,
                        const DetectRegion& cirEyeReg,
                        const Mat& cirEyeGabMap8U,
                        int minWrkTh,
                        int longWrkThresh,
                        CONTOURS& lightWrkConts,
                        CONTOURS& longWrkConts)
{
    cv::Mat biMap;
    int dpWrkBiTh = 50;
    biMap = cirEyeGabMap8U & cirEyeReg.mask;
    cv::threshold(biMap, biMap, dpWrkBiTh, 255, THRESH_BINARY);
    
#ifdef TEST_RUN2
    if(isLeft)
    {
        string GabBiFile = wrkOutDir + "/lCirEyeGabBi.png";
        imwrite(GabBiFile.c_str(), biMap);
    }
    else
    {
        string GabBiFile = wrkOutDir + "/rCirEyeGabBi.png";
        imwrite(GabBiFile.c_str(), biMap);
    }
#endif
    
    chao_thinimage(biMap); //单通道、二值化后的图像
#ifdef TEST_RUN2
    if(isLeft)
    {
        string ThinGabBiFile = wrkOutDir + "/lCirThinGabBi.png";
        imwrite(ThinGabBiFile.c_str(), biMap);
    }
    else
    {
        string ThinGabBiFile = wrkOutDir + "/rCirThinGabBi.png";
        imwrite(ThinGabBiFile.c_str(), biMap);
    }
#endif
    
    CONTOURS contours;
    cv::findContours(biMap, contours, RETR_EXTERNAL, CHAIN_APPROX_NONE);
    
    std::sort(contours.begin(), contours.end(),
        [](const CONTOUR& a, const CONTOUR& b){return a.size() > b.size();});
        
    Point2i tlPt = cirEyeReg.bbox.tl();
    for (CONTOUR cont: contours)
    {
        if (cont.size() >= minWrkTh)
        {
            CONTOUR gsCt;
            transCt_LSS2GS(cont, tlPt, gsCt);
            lightWrkConts.push_back(gsCt);
        }
        if (cont.size() >= longWrkThresh ) // && it_c->size() <= sizeMax )
        {
            CONTOUR gsCt;
            transCt_LSS2GS(cont, tlPt, gsCt);
            longWrkConts.push_back(gsCt);
        }
    }
}
