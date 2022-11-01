//
//  Tie.hpp
//
//
/*
本模块。
 
Author: Fu Xiaoqiang
Date:   2022/11/1
*/

#ifndef TIE_HPP
#define TIE_HPP

#include "opencv2/opencv.hpp"

using namespace std;
using namespace cv;

#include "Common.hpp"

// contour上的线段
struct LineSegment{
    int     sIdx;
    int     eIdx;
    Point2i sPt;
    Point2i ePt;
    
    LineSegment(int sIndex0, int eIndex0,
                const Point2i& sPt0, const Point2i& ePt0):
            sIdx(sIndex0), eIdx(eIndex0),
            sPt(sPt0), ePt(ePt0)
    {
    }
    
    LineSegment(const LineSegment& lSeg):
            sIdx(lSeg.sIdx), eIdx(lSeg.eIdx),
            sPt(lSeg.sPt), ePt(lSeg.ePt)
    {
    }
    
    friend ostream &operator<<(ostream &output, const LineSegment& lSeg )
    {
        output << "LineSegment{" << endl;
        output << "\tsIdx: " << lSeg.sIdx << ", sPt: " << lSeg.sPt << endl;
        output << "\teIdx: " << lSeg.eIdx << ", ePt: " << lSeg.ePt << "}" << endl;
        return output;
    }
};

/*
// 用交叉的两段线段来表示Tie，其他点和线段目前用不到
struct Tie{
    LineSegment lineSeg1;
    LineSegment lineSeg2;
    
    Tie(const LineSegment& lineSegA, const LineSegment& lineSegB):
        lineSeg1(lineSegA) , lineSeg2(lineSegB)
    {
    }
    
    friend ostream &operator<<(ostream &output, const Tie& tie )
    {
        output << "Tie{" << endl;
        output << "\tlineSeg1: " << tie.lineSeg1;
        output << "\tlineSeg2: " << tie.lineSeg2;
        output << "}" << endl;
        return output;
    }
};
*/

enum PtLocOnTie
{
    PT_OUT_TIE,  // Not on the tie
    PT_START_TIE,
    PT_MIDWAY_TIE,
    PT_END_TIE
};

// 记录Tie上的点集，按扫描顺序
struct Tie{
    vector<int> ptIdxs;
    
    Tie()
    {
    }
    
    Tie(const vector<int>& revOrderPts)
    {
        int numPts = revOrderPts.size();
        for(int i=numPts-1; i>=0; i--)
        {
            ptIdxs.push_back(revOrderPts[i]);
        }
    }
    
    bool isValid()
    {
        if(ptIdxs.size() >= 4)
            return true;
        else
            return false;
    }
    
    friend ostream &operator<<(ostream &output, const Tie& tie )
    {
        output << "Tie{" << endl;
        //output << tie.ptIdxs;
        output << "}" << endl;
        return output;
    }
    
    PtLocOnTie GetLocPt(int ptIdx)
    {
        if(!isValid())
            return PT_OUT_TIE;
        
        if(std::find(ptIdxs.begin(), ptIdxs.end(), ptIdx) == ptIdxs.end())
            return PT_OUT_TIE;
        
        int numPt = static_cast<int>(ptIdxs.size());
        if(ptIdx == ptIdxs[0])
            return PT_START_TIE;
        else if(ptIdx == ptIdxs[numPt-1])
            return PT_END_TIE;
        else
            return PT_MIDWAY_TIE;
    }
};

struct TieGroup
{
    vector<Tie> ties;
    
    int getTieNum()
    {
        return static_cast<int>(ties.size());
    }
    
    bool hasTie()
    {
        return (getTieNum() == 0);
    }
    
    // 仅仅检查第一个Tie
    bool ptInTie(int idx, Tie*& pTie)
    {
        /*
        if(getTieNum() == 0)
        {
            pTie = NULL;
            return false;
        }
        
        if(ties[0].lineSeg1.sIdx == idx)
        {
            pTie = &(ties[0]);
            return true;
        }
        else
        {
            pTie = NULL;
            return false;
        }
        */
        return true;
    }
};

// 从头部取出，从尾部加入，环形装置；
// 取出时，将head flag往后拨一格；
// 加入新元素时，将tail flag往后拨一格。
template <class T>
class CircularBuf
{
  private:
    vector<T> buf;
    int headFlag;
    int tailFlag;
    int bufSize; // 满员时的容量
    
  public:
    CircularBuf(int bufSize0)
    {
        bufSize = bufSize0;
        buf.reserve(bufSize);
        headFlag = -1; // !!!
        tailFlag = 0;
    }
    
    void cleanUp()
    {
        headFlag = -1; // !!!
        tailFlag = 0;
    }
    
    // return how much elements stored in the buf now
    int elementNum() const
    {
        return (tailFlag + bufSize - headFlag) % bufSize;
    }
    
    bool isFull()
    {
        if(tailFlag == headFlag)
            return true;
        else
            return false;
    }
    
    // if index is a negative value, counting from tail
    // if index is zero or a positive value, coutning from head
    T getElement(int index) const
    {
        if(index >= 0) // starting from head and iterating forward
        {
            int actIndex = (headFlag + index) % bufSize;
            return buf[actIndex];
        }
        else  // starting from tail and iterating backward
        {
            int actIndex = (tailFlag + index + bufSize) % bufSize;
            return buf[actIndex];
        }
    }
    
    // 如果tail位置为0的话，head位置应该是什么
    int headReverseIndex() const
    {
        return headFlag - tailFlag;
    }
    
    T popFront()
    {
        T popElement = buf[headFlag];
        headFlag = (headFlag+1) % bufSize;
        
        if(headFlag == tailFlag) // become empty
        {
            headFlag = -1; // !!!
            tailFlag = 0;
        }
        return popElement;
    }
    void pushBack(T& t)
    {
        if(headFlag == -1)  // add the first element
            headFlag = 0;
        
        buf[tailFlag] = t;
        tailFlag = (tailFlag + 1) % bufSize;
    }
};

//-------------------------------------------------------------------------------------------
// 将contour多余的点（指挨得太近的点）删除掉
// 思路：计算contour上平均的点距，将小于alpha倍平均点距的点给依次删掉。
void SparsePtsOnContour(const CONTOUR& oriCont, float alpha, CONTOUR& sparCont);

// 1st version: just check there is a tie existed or not
void CheckTieOnContour(const CONTOUR& oriCont, int lineSegBufSize,
                       TieGroup& tieGroup);

// 1st version: just check there is a cross existed between lineSeg and
// any line segment of lineSegBuf.
// includeFinalSeg indicates whether the last one of lineSegBuf will be considered or not.
bool CheckCrossOfLineSegs(const LineSegment& lineSeg,
                          const list<LineSegment>& lineSegBuf,
                          vector<Tie>& tieSet,
                          bool includeLastSeg=false);

bool CheckCrossOfTwoLineSegs(const LineSegment& lineSeg1, const LineSegment& lineSeg2);

// clean up the points in ties recorded in tieGroup, produce a clean contour without ties.
void DelTiesOnContV2(const CONTOUR& oriCont, TieGroup& tieGroup, CONTOUR& cleanCont);

bool InBlackList(int ptIdx, const vector<int>& blackList);

void UnionPtsOnTieGroup(const TieGroup& tieGroup, vector<int>& ptBlackList);

#endif /* end of TIE_HPP */
