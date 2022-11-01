//
//  Tie.cpp

/*******************************************************************************
本模块。

Author: Fu Xiaoqiang
Date:   2022/11/1

********************************************************************************/

#include "Geometry.hpp"
#include "Tie.hpp"

//-------------------------------------------------------------------------------------------

// 将contour多余的点（指挨得太近的点）删除掉
// 思路：计算contour上平均的点距，将小于alpha倍平均点距的点给依次删掉。
void SparsePtsOnContour(const CONTOUR& oriCont, float alpha, CONTOUR& sparCont)
{
    float avgDist = AvgPointDist(oriCont);
    float shortThresh = avgDist * alpha;
    
    int numPt = static_cast<int>(oriCont.size());

    for(int i=0; i<numPt; i++)
    {
        int prevID = (i + numPt - 1) % numPt;
        Point2i prevPt = oriCont[prevID];
        int nextID = (i + 1) % numPt;
        Point2i nextPt = oriCont[nextID];
        
        float dist0 = DisBetw2Pts(prevPt, oriCont[i]);
        float dist1 = DisBetw2Pts(oriCont[i], nextPt);
        
        if(dist0 <= shortThresh) // && dist1 <= shortThresh)
        {
            cout << "a point deleted!" << endl;
            continue;
        }
        else
        {
            sparCont.push_back(oriCont[i]);
        }
    }
}

//-------------------------------------------------------------------------------------------
int crossProduct(const Point2i& vect1, const Point2i& vect2)
{
    return (vect1.x * vect2.y - vect1.y * vect2.x);
}

bool CheckCrossOfTwoLineSegs(const LineSegment& lineSeg1, const LineSegment& lineSeg2)
{
    // <a, b> is lineSeg1; <c, d> is lineSeg2
    const Point2i& a = lineSeg1.sPt;
    const Point2i& b = lineSeg1.ePt;
    const Point2i& c = lineSeg2.sPt;
    const Point2i& d = lineSeg2.ePt;
    
    //快速排斥实验
    if( max(c.x, d.x) < min(a.x, b.x)||
        max(a.x, b.x) < min(c.x, d.x)||
        max(c.y, d.y) < min(a.y, b.y)||
        max(a.y, b.y) < min(c.y, d.y))
    {
        return false;
    }
    //跨立实验
    if(   crossProduct(a-d,c-d)*crossProduct(b-d,c-d)>0
       || crossProduct(d-b,a-b)*crossProduct(c-b,a-b)>0)
    {
        return  false;
    }
    
    return true;
}


// 1st version: just check there is a cross existed between lineSeg and
// any line segment of lineSegBuf.
// includeFinalSeg indicates whether the last one of lineSegBuf will be considered or not.
bool CheckCrossOfLineSegs(const LineSegment& lineSeg,
                          const CircularBuf<LineSegment>& lineSegBuf,
                          vector<Tie>& tieSet,
                          bool includeLastSeg)
{
    if(!includeLastSeg)
    {
        // the processing needs that there are at least two line segs existed
        if(lineSegBuf.elementNum() <= 1)
            return false;
        
        int headRevIndex = lineSegBuf.headReverseIndex();
        for(int i = -2; i>= headRevIndex; i--)
        {
            LineSegment lineSeg2 = lineSegBuf.getElement(i);
            bool isIntersect = CheckCrossOfTwoLineSegs(lineSeg, lineSeg2);
            if(isIntersect)
            {
                tieSet.push_back(Tie(lineSeg2, lineSeg));
                return true;
            }
        }
    }
    else
    {
        if(lineSegBuf.elementNum() <= 0)
            return false;
        
        int headRevIndex = lineSegBuf.headReverseIndex();
        for(int i = -1; i >= headRevIndex; i--)
        {
            LineSegment lineSeg2 = lineSegBuf.getElement(-i);
            bool isIntersect = CheckCrossOfTwoLineSegs(lineSeg, lineSeg2);
            if(isIntersect)
                return true;
        }
    }
        
    return false;
}

// 1st version: just check there is a tie existed or not
void CheckTieOnContour(const CONTOUR& oriCont, int lineSegBufSize,
                       TieGroup& tieGroup)
{
    CircularBuf<LineSegment> lineSegBuf(lineSegBufSize);
    
    int numPt = static_cast<int>(oriCont.size());
    if(numPt <= 3)
        return;

    for(int i=0; i <=numPt-1; i++)
    {
        int j = (i + 1) % numPt;
        LineSegment curLineSeg(i, j, oriCont[i], oriCont[j]);
        
        bool hasCross = CheckCrossOfLineSegs(curLineSeg, lineSegBuf,
                                             tieGroup.ties, false);
        if(hasCross)
            lineSegBuf.cleanUp(); //clean up the buf to prepare for detecting next tie
        else
            lineSegBuf.pushBack(curLineSeg);
    }
}

// clean up the points in ties recorded in tieGroup, produce a clean contour without ties.
void DelTiesOnCont(const CONTOUR& oriCont, TieGroup& tieGroup, CONTOUR& cleanCont)
{
    if(tieGroup.hasTie() == false)
    {
        cleanCont = oriCont;
        return;
    }
    
    int numPt = static_cast<int>(oriCont.size());

    int curPtIdx = 0;
    while(curPtIdx <=numPt-1)
    {
        /*
        Tie* pTie;
        bool inTile = tieGroup.ptInTie(i, pTie);
        if(!inTie)
        {
            cleanCont.push_back(oriCont[i]);
        }
        else
        {
            
        }
        */
    }
}
