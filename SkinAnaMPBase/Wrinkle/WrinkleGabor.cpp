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

Mat ApplyGaborFilter(vector<CvGabor*> gaborBank, const Mat& inImgInFR_Gray,
                   const Rect& detectRect)
{
    cv::Mat regImg = inImgInFR_Gray(detectRect).clone();
    
    //cout << "regImg data type: " << openCVType2str(regImg.type()) << endl;
    
    uchar maxV = *max_element(regImg.begin<uchar>(), regImg.end<uchar>());
    uchar minV = *min_element(regImg.begin<uchar>(), regImg.end<uchar>());
    
    float alpha = 255.0 / (maxV - minV);
    regImg = (regImg - minV) * alpha;
    
    //cout << "regImg data type after scale: " << openCVType2str(regImg.type()) << endl;

    cv::Mat temp1, temp2, temp3, temp4, temp5;
    
#pragma omp parallel sections num_threads(5)
    {
#pragma omp section
        {
            gaborBank[0]->conv_img(regImg, temp1, 1);  // lRegion is source, temp1 is destination
        }
#pragma omp section
        {
            gaborBank[1]->conv_img(regImg, temp2, 1);
        }
#pragma omp section
        {
            gaborBank[2]->conv_img(regImg, temp3, 1);
        }
#pragma omp section
        {
            gaborBank[3]->conv_img(regImg, temp4, 1);
        }
#pragma omp section
        {
            gaborBank[4]->conv_img(regImg, temp5, 1);
        }
    }
    
    //函数原型： void cv::min(InputArray  src1, InputArray  src2, OutputArray  dst)
    //功   能： 单像素操作（与领域无关）；将src1和src2同位置的像素值取小者，存贮到dst中
    cv::Mat regResp(regImg.size(), CV_8S, cv::Scalar(127)); // 8S: 8位有符号数
    //cout << "lRegResp data type: " << openCVType2str(regResp.type()) << endl;
    
    cv::min(regResp, temp1, regResp);  //
    cv::min(regResp, temp2, regResp);
    cv::min(regResp, temp3, regResp);
    cv::min(regResp, temp4, regResp);
    cv::min(regResp, temp5, regResp);
    
    regResp.convertTo(regResp, CV_8U, -1.0/*,-1*Min/20*/);
    return regResp;
}

// 返回左面颊的Gabor滤波响应值
Mat CalcOneCheekGaborResp(const vector<Point2i>& wrkSpline,
                          int spPtIDs[4],
                          vector<CvGabor*> gaborBank,
                  const Rect& faceRect,
                  const Mat& inImgInFR_Gray,
                  Rect& cheekRect)
{
    vector<Point2i> cheekPg; // Pg: polygon
    for(int i=0; i<4; i++)
    {
        int spID = spPtIDs[i];
        Point2i relCd = CalcRelCdToFR(wrkSpline[spID], faceRect);
        cheekPg.push_back(relCd);
    }
    
    cheekRect = boundingRect(cheekPg);
    int margin = faceRect.width / 200;
    ClipRectByFR(faceRect.width, faceRect.height, margin, cheekRect);

    Mat lRegResp = ApplyGaborFilter(gaborBank, inImgInFR_Gray,
                                  cheekRect);

    return lRegResp;
}

// glabella，眉间，印堂
Mat CalcGlabellaGaborResp(const vector<Point2i>& wrkSpline,
                  const Rect& faceRect,
                  const Mat& inImgInFR_Gray,
                  Rect& glabeRect,
                  const Rect& lrect)
{
    int x_offset = 100 * faceRect.width / 1800;
    int y_offset = 100 * faceRect.height / 1800;

    std::vector<cv::Point2i> yFacePoly;
    yFacePoly.push_back(Point(wrkSpline[5].x - faceRect.x - x_offset, wrkSpline[5].y - faceRect.y + y_offset));
    yFacePoly.push_back(Point(wrkSpline[5].x - faceRect.x, wrkSpline[4].y - faceRect.y));
    yFacePoly.push_back(Point(wrkSpline[26].x - faceRect.x, wrkSpline[4].y - faceRect.y));
    yFacePoly.push_back(Point(wrkSpline[26].x - faceRect.x + x_offset, wrkSpline[26].y - faceRect.y + y_offset));
    
    glabeRect = boundingRect(yFacePoly);
    
    glabeRect.x = glabeRect.x > 0 ? glabeRect.x : 0;
    glabeRect.y = glabeRect.y > 0 ? glabeRect.y : 0;
    //glabeRect.y = lrect.y < inImgInFR_Gray.rows ? glabeRect.y : inImgInFR_Gray.rows - 10;
    //glabeRect.x = lrect.x < inImgInFR_Gray.cols ? glabeRect.x : inImgInFR_Gray.cols - 10;
    glabeRect.y = glabeRect.y < inImgInFR_Gray.rows ? glabeRect.y : inImgInFR_Gray.rows - 10;
    glabeRect.x = glabeRect.x < inImgInFR_Gray.cols ? glabeRect.x : inImgInFR_Gray.cols - 10;

    glabeRect.width = glabeRect.width < inImgInFR_Gray.cols - glabeRect.x ? glabeRect.width : inImgInFR_Gray.cols - glabeRect.x;
    glabeRect.height = glabeRect.height < inImgInFR_Gray.rows - glabeRect.y ? glabeRect.height : inImgInFR_Gray.rows - glabeRect.y;
    
    vector<CvGabor*> gaborBank;
    for(int i=10; i<15; i++)
    {
        gaborBank.push_back(gabor+i);
    }
        
    Mat eRegionRe = ApplyGaborFilter(gaborBank, inImgInFR_Gray, glabeRect);
    return eRegionRe;
}

// forehead，前额
Mat CalcFhGaborResp(const vector<Point2i>& wrkSpline,
                  const Rect& faceRect,
                  const Mat& inImgInFR_Gray,
                  Rect& fhRect)
{
    std::vector<cv::Point2i> fhPoly;
    fhPoly.push_back(Point(wrkSpline[0].x - faceRect.x + 20,
                           wrkSpline[3].y - faceRect.y - 40)); // y坐标采用Pt3，是有意为之，不是笔误
    fhPoly.push_back(Point(wrkSpline[3].x - faceRect.x - 20, wrkSpline[3].y - faceRect.y - 40));
    fhPoly.push_back(Point(wrkSpline[4].x - faceRect.x, wrkSpline[4].y - faceRect.y));
    fhPoly.push_back(Point(wrkSpline[5].x - faceRect.x, wrkSpline[5].y - faceRect.y));
    fhPoly.push_back(Point(wrkSpline[26].x - faceRect.x, wrkSpline[26].y - faceRect.y));
    fhPoly.push_back(Point(wrkSpline[27].x - faceRect.x, wrkSpline[27].y - faceRect.y));
    
#ifdef TEST_RUN_WRK
    Mat annoImage = inImgInFR_Gray.clone();
    AnnoPointsOnImg(annoImage, fhPoly);
    
    string fhPolyFile =  wrk_out_dir + "/fhPoly.png";
    imwrite(fhPolyFile.c_str(), annoImage);
#endif
    
    Rect rect4 = boundingRect(fhPoly);
    
    fhRect = rect4;
    ClipRectByFR(faceRect.width, faceRect.height, 10, fhRect);

    vector<CvGabor*> gaborBank;
    for(int i=15; i<20; i++)
    {
        gaborBank.push_back(gabor+i);
    }
    Mat fRegionRe = ApplyGaborFilter(gaborBank, inImgInFR_Gray, fhRect);
    return fRegionRe;
}

// for nose ?
Mat CalcUpperNoseGaborResp(const vector<Point2i>& wrkSpline,
                  const Rect& faceRect,
                  const Mat& inImgInFR_Gray,
                  Rect& nRect)
{
    std::vector<cv::Point2i> nFacePoly;
    nFacePoly.push_back(Point(wrkSpline[5].x - faceRect.x - 100, wrkSpline[5].y - faceRect.y));
    nFacePoly.push_back(Point(wrkSpline[15].x - faceRect.x, wrkSpline[15].y - faceRect.y));
    nFacePoly.push_back(Point(wrkSpline[16].x - faceRect.x, wrkSpline[16].y - faceRect.y));
    nFacePoly.push_back(Point(wrkSpline[26].x - faceRect.x + 100, wrkSpline[26].y - faceRect.y));
    
    Rect rect5 = boundingRect(nFacePoly);
    nRect = rect5;
    ClipRectByFR(faceRect.width, faceRect.height, 10, nRect);

    vector<CvGabor*> gaborBank;
    for(int i=15; i<20; i++)
    {
        gaborBank.push_back(gabor+i);
    }
    Mat nRegResp = ApplyGaborFilter(gaborBank, inImgInFR_Gray, nRect);
    
    return nRegResp;
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
void CalcGaborResp(const Mat& grFrImg,
                           const Rect& faceRect,
                           const SPLINE& wrkSpline,
                           Mat& gaborRespMap)
{
    init_gabor();
    
    Rect lCheekRect, rCheekRect, glabeRect, fhRect, noseRect;
    
    // 计算左面颊的Gabor滤波响应值
    vector<CvGabor*> lGaborBank;
    InitLCheekGaborBank(lGaborBank);
    int lcSpPtIDs[4] = {6, 10, 13, 15};
    Mat lCheekResp = CalcOneCheekGaborResp(wrkSpline,
        lcSpPtIDs, lGaborBank, faceRect, grFrImg, lCheekRect);

    // 计算右面颊的Gabor滤波响应值
    vector<CvGabor*> rGaborBank;
    InitRCheekGaborBank(rGaborBank);
    int rcSpPtIDs[4] = {25, 21, 18, 16};
    Mat rCheekResp = CalcOneCheekGaborResp(wrkSpline,
        rcSpPtIDs, rGaborBank, faceRect, grFrImg, rCheekRect);

    // glabella，眉间，印堂
    Mat glabeRegResp = CalcGlabellaGaborResp(wrkSpline,
                                         faceRect, grFrImg, glabeRect, lCheekRect);
    
    // 前额
    Mat fhRegResp = CalcFhGaborResp(wrkSpline,
                                         faceRect, grFrImg, fhRect);
    
    //鼻子的上半部分
    Mat noseRegResp = CalcUpperNoseGaborResp(wrkSpline,
                                          faceRect, grFrImg, noseRect);
    
#ifdef TEST_RUN_WRK
    bool isSuccess;
    
    isSuccess = SaveTestOutImgInDir(lCheekResp, wrk_out_dir,  "lRegResp.png");
    isSuccess = SaveTestOutImgInDir(rCheekResp, wrk_out_dir,  "rRegResp.png");
    isSuccess = SaveTestOutImgInDir(glabeRegResp, wrk_out_dir,  "glabeRegResp.png");
    isSuccess = SaveTestOutImgInDir(fhRegResp,  wrk_out_dir,   "fRegResp.png");
    isSuccess = SaveTestOutImgInDir(noseRegResp, wrk_out_dir, "nRegResp.png");
    
    Mat canvas = grFrImg.clone();
    rectangle(canvas, lCheekRect, CV_COLOR_RED, 8, 0);
    rectangle(canvas, rCheekRect, CV_COLOR_GREEN, 8, 0);
    rectangle(canvas, glabeRect, CV_COLOR_BLUE, 8, 0);
    rectangle(canvas, fhRect, CV_COLOR_YELLOW, 8, 0);
    rectangle(canvas, noseRect, CV_COLOR_WHITE, 8, 0);
    
    isSuccess = SaveTestOutImgInDir(canvas, wrk_out_dir, "fiveRects.png");
    canvas.release();

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
