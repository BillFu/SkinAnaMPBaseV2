#include <algorithm>

#include "../ImgProc.h"
#include "../Utils.hpp"
#include "../Geometry.hpp"

#include "WrinkleFrangi.h"
#include "frangi.h"



// -------------------------------------------------------------------------
// 从Frangi滤波响应中提取深皱纹和长皱纹
void PickDLWrkInFrgiMapV2(int minWrkTh, int longWrkTh,
                          Mat& frgiMap8U,
                          CONTOURS& deepWrkConts,
                          CONTOURS& longWrkConts)
{
    // DL: deep and long
    cv::Mat WrkBi(frgiMap8U.size(), CV_8UC1, cv::Scalar(0));
    threshold(frgiMap8U, WrkBi, 40, 255, THRESH_BINARY);
    chao_thinimage(WrkBi);
    
#ifdef TEST_RUN2
    string frgiThinFile = wrkOutDir + "/FrgiThin.png";
    imwrite(frgiThinFile.c_str(), WrkBi);
#endif
    
    Mat elmt = getStructuringElement(MORPH_ELLIPSE, Size(15, 1));
    Mat dilBi;
    dilate(WrkBi, dilBi, elmt, Point2i(-1,-1), 1);
    
    CONTOURS thickCts;
    findContours(dilBi, thickCts, RETR_EXTERNAL, CHAIN_APPROX_NONE);
    
    CONTOURS::const_iterator it_ct = thickCts.begin();
    unsigned long ct_size = thickCts.size();
    for (unsigned int i = 0; i < ct_size; ++i)
    {
        //DelDupPtOnCont(thickCts[i]);

        if (it_ct->size() >= minWrkTh)
        {
            deepWrkConts.push_back(thickCts[i]);
        }
        if (it_ct->size() >= longWrkTh)
        {
            longWrkConts.push_back(thickCts[i]);
        }
        it_ct++;
    }
}

////////////////////////////////////////////////////////////////////////////////////////

void CcFrgiMap(const Mat& imgGray, int scaleRatio, Mat& frgiMap8U)
{
    Mat ehGrImg; // eh: enhanced
    PreprocGrImg(imgGray, ehGrImg);
        
    Mat frgiMapRz8U;
    ApplyFrgiFilter(ehGrImg, scaleRatio, frgiMapRz8U);
    
#ifdef TEST_RUN2
    string frgiMapFile =  wrkOutDir + "/frgiMap.png";
    imwrite(frgiMapFile, frgiMapRz8U);
#endif
    
    //把响应强度图又扩大到原始影像的尺度上来，但限定在Face Rect内。
    resize(frgiMapRz8U, frgiMap8U, imgGray.size());
}

Mat CcFrgiMapInRect(const Mat& imgGray,
                    const Rect& rect,
                    int scaleRatio)
{
    Mat ehGrImg; // eh: enhanced
    PreprocGrImg(imgGray, ehGrImg);
    
    Mat ehGrImgRt = ehGrImg(rect);
    
    Mat mapRtRz8U;
    ApplyFrgiFilter(ehGrImgRt, scaleRatio, mapRtRz8U);
    
    //把响应强度图又扩大到原始影像的尺度上来，但限定在Face Rect内。
    Mat mapSSInRt8U;
    resize(mapRtRz8U, mapSSInRt8U, rect.size());
    Mat frgiMapGS(imgGray.size(), CV_8UC1, Scalar(0));
    
    mapSSInRt8U.copyTo(frgiMapGS(rect));
    return frgiMapGS;
}

void ApplyFrgiFilter(const Mat& inGrImg,
                     int scaleRatio,
                     Mat& frgiRespRzU8)
{
    Size rzSize = inGrImg.size() / scaleRatio;
    Mat rzImg;
    resize(inGrImg, rzImg, rzSize);
    
    Mat rzFtImg; // Ft: float
    rzImg.convertTo(rzFtImg, CV_32FC1);
    rzImg.release();
    
    cv::Mat respScaleRz, respAngRz;
    frangi2d_opts opts;
    opts.sigma_start = 1;
    opts.sigma_end = 5;
    opts.sigma_step = 2;
    opts.BetaOne = 0.5;  // BetaOne: suppression of blob-like structures.
    opts.BetaTwo = 12.0; // background suppression. (See Frangi1998...)
    opts.BlackWhite = true;
    
    // !!! 计算fangi2d时，使用的是缩小版的衍生影像
    Mat frgiRespRz;
    frangi2d(rzFtImg, frgiRespRz, respScaleRz, respAngRz, opts);
    rzFtImg.release();
    
    //返回的scaleRz, anglesRz没有派上实际的用场
    respScaleRz.release();
    respAngRz.release();
        
    frgiRespRzU8 = CvtFtImgTo8U_MinMax(frgiRespRz);
}

// bifurcation point: 分叉点
void FindKeyPtsOnCont(const Mat& biImg, const CONTOUR& ct,
                      KeyPtsOnCont& keyPts, int i)
{
    Rect imgRect(0, 0, biImg.cols, biImg.rows);
    for(Point2i pt: ct)
    {
        //cout << "pt: " << pt << endl;
        vector<int> neibValues;
        get16NeibValues(pt, biImg, imgRect, neibValues);
        int turnNum = getTurnNum(neibValues);
        //assert(connDeg != 0);
        if(turnNum >= 3)
        {
            if (find(keyPts.bifuPtSet.begin(), keyPts.bifuPtSet.end(), pt)
                == keyPts.bifuPtSet.end())
                keyPts.bifuPtSet.push_back(pt);
        }
        else if(turnNum == 1)
        {
            if (find(keyPts.endPtSet.begin(), keyPts.endPtSet.end(), pt)
                == keyPts.endPtSet.end())
                keyPts.endPtSet.push_back(pt);
        }
    }

    cout << "pt in keyPts.bifuPtSet " << i << endl;

    for(Point2i pt: keyPts.bifuPtSet)
    {
        cout << pt << endl;
    }
    
    cout << "----------------------------" << endl;

    if(keyPts.bifuPtSet.size() > keyPts.endPtSet.size())
    {
        cout << "index: " << i << endl;
        cout << "error ********" << endl;
        cout << "bifuPtSet.size: " << keyPts.bifuPtSet.size() << endl;
        cout << "endPtSet.size: " << keyPts.endPtSet.size() << endl;
    }
}

void PruneBurrOnConts(Mat& biMap, const CONTOURS& conts)
{
    int i = 0;
    for(CONTOUR ct: conts)
    {
        KeyPtsOnCont keyPts;
        FindKeyPtsOnCont(biMap, ct, keyPts, i);
        PruneBurrOnCont(biMap, keyPts);
        i++;
    }
}

void PruneBurrOnCont(Mat& biMap, KeyPtsOnCont& keyPts)
{
    // for each bifurcation point, choose one end-point to prune
    // this branch connecting those two points
    for(Point2i bifuPt: keyPts.bifuPtSet)
    {
        Point2i endPt = ChooseEndPtOnBranch(bifuPt, keyPts.endPtSet,
                                            M_PI*0.5, M_PI*1.5);
        PruneOneBurrOnCont(biMap, bifuPt, endPt);
        
        POINT_SET clearEndPts = EraseEndPtOnCont(keyPts.endPtSet, endPt);
        keyPts.endPtSet.clear();
        keyPts.endPtSet = clearEndPts;
    }
}

POINT_SET EraseEndPtOnCont(const POINT_SET& endPts, const Point2i& delEndPt)
{
    POINT_SET clearEndPts;
    
    for(Point2i pt: endPts)
    {
        if(pt != delEndPt)
            clearEndPts.push_back(pt);
    }
        
    return clearEndPts;
}

// orient1 < orient2
Point2i ChooseEndPtOnBranch(const Point2i& bifuPt,
                            const POINT_SET& endPts,
                            float refOrient1,
                            float refOrient2)
{
    vector<TwoPtsPose> poseSet;
    for(Point2i endPt: endPts)
    {
        TwoPtsPose pose;
        CalcTwoPtsPose(bifuPt, endPt, pose);
        poseSet.push_back(pose);
    }
    
    std::sort(poseSet.begin(), poseSet.end(), dist_less_than());
    cout << "-------------------------" << endl;
    for(TwoPtsPose pose: poseSet)
    {
        cout << "dist: " << pose.dist << endl;
        cout << "angle: " << pose.angle << endl;
    }
    cout << "-------------------------" << endl;

    float dev_orient = M_PI / 4.0;
    for(TwoPtsPose pose: poseSet)
    {
        if(isInOrientRange(pose.angle,  refOrient1, dev_orient))
            return pose.p2;
        
        if(isInOrientRange(pose.angle,  refOrient2, dev_orient))
            return pose.p2;
    }
    
    return poseSet[0].p2;
}

void PruneOneBurrOnCont(Mat& biMap, const Point2i& bifuPt, const Point2i& endPt)
{
    // starting from endPt, moving torward bifuPt, and erase the intermidate points on contour
    Point2i curPt = endPt;
    
    Rect bbox(0, 0, biMap.cols, biMap.rows);
    while(curPt != bifuPt)
    {
        POINT_SET neibPts = get8NeibCoordinates(curPt, bbox);
        Point2i nextPt(-1, -1);
        for(Point2i pt: neibPts)
            if(biMap.at<uchar>(pt) > 0)
            {
                nextPt = pt;
                break;
            }
        
        biMap.at<uchar>(curPt) = 0; // erase the current point
        curPt = nextPt;
    }
}
