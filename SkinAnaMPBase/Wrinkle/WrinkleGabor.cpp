#include "WrinkleGabor.h"
#include "../Utils.hpp"
#include "../ImgProc.h"
#include <algorithm>

CvGabor gabor[20];

void init_gabor()
{
    double Sigma = 4.442882933081; // sqrt(2) * 3.14159265
    double F = 1.41421356237; //sqrt(2);
    
    gabor[0].Init(0.7853981625,  4, Sigma, F);   // 45度
    gabor[1].Init(1.17809724375, 3, Sigma, F);   // 67.5度
    gabor[2].Init(0.7853981625,  2, Sigma, F);   // 45度
    gabor[3].Init(1.17809724375, 1, Sigma, F);   // 67.5度
    gabor[4].Init(1.17809724375, 0, Sigma, F);   // 67.5度
    
    gabor[5].Init(2.3561944875,  4, Sigma, F);   // 135度
    gabor[6].Init(1.96349540625, 3, Sigma, F);   // 112.5度
    gabor[7].Init(2.3561944875,  2, Sigma, F);   // 135度
    gabor[8].Init(1.96349540625, 1, Sigma, F);   // 112.5度
    gabor[9].Init(1.96349540625, 0, Sigma, F);   // 112.5度
    
    gabor[10].Init(1.570796325,   4, Sigma, F);    // 90度
    gabor[11].Init(1.17809724375, 3, Sigma, F);    // 67.5度
    gabor[12].Init(1.570796325,   2, Sigma, F);    // 90度
    gabor[13].Init(1.96349540625, 1, Sigma, F);    // 112.5度
    gabor[14].Init(1.570796325,   0, Sigma, F);    // 90度
    
    // 下面的5个gabor对象用于前额的皱纹检测
    gabor[15].Init(0, 4, Sigma, F);
    gabor[16].Init(0, 3, Sigma, F);
    gabor[17].Init(0, 2, Sigma, F);
    gabor[18].Init(2.74889356875, 1, Sigma, F);  // 157.5度
    gabor[19].Init(0.39269908125, 1, Sigma, F);  // 22.5 度

    //gabor[18].Init(2.74889356875, 2, Sigma, F);  // 157.5度
    //gabor[19].Init(0.39269908125, 2, Sigma, F);  // 22.5 度
    
    /*
    gabor[15].Init(0, 3, Sigma, F);
    gabor[16].Init(0, 2, Sigma, F);
    gabor[17].Init(0, 1, Sigma, F);
    gabor[18].Init(2.74889356875, 1, Sigma, F);  // 157.5度
    gabor[19].Init(0.39269908125, 1, Sigma, F);  // 22.5 度
    */
}

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
//-----------------------------------------------------------

void InitLCheekGaborBank(vector<CvGabor*>& lGaborBank)
{
    for(int i=0; i<5; i++)
    {
        lGaborBank.push_back(gabor+i);
    }
}

void InitRCheekGaborBank(vector<CvGabor*>& rGaborBank)
{
    for(int i=5; i<10; i++)
    {
        rGaborBank.push_back(gabor+i);
    }
}

Mat ApplyGaborFilter(const vector<CvGabor*>& gaborBank, const Mat& detRegImg)
{
    //cout << "regImg data type: " << openCVType2str(regImg.type()) << endl;
    Mat detRegImgF; // F for float
    detRegImg.convertTo(detRegImgF, CV_32FC1);

    float maxV = *max_element(detRegImgF.begin<float>(), detRegImgF.end<float>());
    float minV = *min_element(detRegImgF.begin<float>(), detRegImgF.end<float>());
    
    detRegImgF = (detRegImgF - minV) / (maxV - minV); // 调整到[0.0, 1.0]

    vector<Mat> respMapGroup;
    int numBank = static_cast<int>(gaborBank.size());
#pragma omp parallel sections num_threads(numBank)
    {
#pragma omp for
        for(int i=0; i<numBank; i++)
        {
            Mat respMap;
            gaborBank[i]->conv_img(detRegImgF, respMap);  // lRegion is source, temp1 is destination
            respMapGroup.push_back(respMap);
        }
    }
    detRegImgF.release();
    
    cout << "data type of respMap: " << openCVType2str(respMapGroup[0].type()) << endl;

    //函数原型： void cv::min(InputArray  src1, InputArray  src2, OutputArray  dst)
    //功   能： 单像素操作（与领域无关）；将src1和src2同位置的像素值取小者，存贮到dst中
    cv::Mat regRespMap(detRegImg.size(), CV_32FC1, cv::Scalar(-9999.9)); //
    
    for(int i=0; i<numBank; i++)
    {
        cv::max(regRespMap, respMapGroup[i], regRespMap);  //
    }
    
    Mat regRespMapU = CvtFloatImgTo8UImg(regRespMap);
    return regRespMapU;
}

// 返回一个面颊区域的Gabor滤波响应值
Mat CalcGaborRespInOneCheek(const vector<CvGabor*>& gaborBank,
                  const Mat& grSrcImg,
                  const Rect& cheekRect)
{
    Mat imgInCheekRect = grSrcImg(cheekRect);
    Mat lRegResp = ApplyGaborFilter(gaborBank, imgInCheekRect);

    return lRegResp;
}


Mat CalcGaborRespInOneEyeBag(const vector<CvGabor*>& gaborBank,
                          const Mat& grSrcImg,
                          const Rect& eyeBagRect)
{
    Mat imgInEBRect = grSrcImg(eyeBagRect);
    Mat respMap = ApplyGaborFilter(gaborBank, imgInEBRect);

    return respMap;
}

// glabella，眉间，印堂
Mat CalcGaborRespOnGlab(const Mat& grSrcImg,
                  Rect& glabeRect)
{
    Mat imgInGlabRect = grSrcImg(glabeRect);

    vector<CvGabor*> gaborBank;
    for(int i=10; i<15; i++)
    {
        gaborBank.push_back(gabor+i);
    }
        
    Mat glabRespMap = ApplyGaborFilter(gaborBank, imgInGlabRect);
    return glabRespMap;
}

// forehead，前额
Mat CalcGaborRespOnFh(const Mat& grSrcImg,
                      const Rect& fhRect)
{
    float F = 1.414;
    float Sigma = 2*M_PI;
    
    vector<CvGabor*> gaborBank;
    
    CvGabor g1(0.0, 3, Sigma, F);
    CvGabor g2(0.0, 2, Sigma, F);
    CvGabor g3(0.0, 4, Sigma, F);
    CvGabor g4(2.74889356875, 2, Sigma, F);
    CvGabor g5(0.39269908125, 2, Sigma, F);
    
    //gaborBank.push_back(CvGabor(0.0, 4, Sigma, F));
    gaborBank.push_back(&g1);
    gaborBank.push_back(&g2);
    gaborBank.push_back(&g3);
    gaborBank.push_back(&g4); // 157.5度
    gaborBank.push_back(&g5); // 22.5 度

    Mat imgInFhRect = grSrcImg(fhRect);
    Mat fhRespMap = ApplyGaborFilter(gaborBank, imgInFhRect);

    return fhRespMap;
}

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
void CalcGaborResp(const Mat& grSrcImg,
                   WrkRegGroup& wrkRegGroup,
                   Mat& gaborRespMap)
{
    init_gabor();
   
    // 计算左面颊的Gabor滤波响应值
    vector<CvGabor*> lGaborBank;
    InitLCheekGaborBank(lGaborBank);
    Mat lCheekResp = CalcGaborRespInOneCheek(lGaborBank, grSrcImg, wrkRegGroup.lCheekReg.bbox);

    // 计算右面颊的Gabor滤波响应值
    vector<CvGabor*> rGaborBank;
    InitRCheekGaborBank(rGaborBank);
    Mat rCheekResp = CalcGaborRespInOneCheek(rGaborBank, grSrcImg, wrkRegGroup.rCheekReg.bbox);

    // 前额
    Mat fhRegResp = CalcGaborRespOnFh(grSrcImg, wrkRegGroup.fhReg.bbox);
    
    // glabella，眉间，印堂
    Mat glabeRegResp = CalcGaborRespOnGlab(grSrcImg, wrkRegGroup.glabReg.bbox);
     
    
#ifdef TEST_RUN2
    bool isSuccess;
    
    isSuccess = SaveTestOutImgInDir(fhRegResp,  wrkOutDir,   "fhGaborResp.png");
    isSuccess = SaveTestOutImgInDir(lCheekResp, wrkOutDir,  "lCheekGaborResp.png");
    isSuccess = SaveTestOutImgInDir(rCheekResp, wrkOutDir,  "rCheekGaborResp.png");
    isSuccess = SaveTestOutImgInDir(glabeRegResp, wrkOutDir,  "glabGaborResp.png");
    
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
