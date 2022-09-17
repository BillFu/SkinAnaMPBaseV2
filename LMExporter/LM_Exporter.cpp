//
//  LM_Exporter.cpp

/*******************************************************************************
Author: Fu Xiaoqiang
Date:   2022/9/15

********************************************************************************/

#include <iostream>
#include <fstream>
using namespace std;

#include "LM_Exporter.hpp"


//-------------------------------------------------------------------------------------------

/******************************************************************************************

 ******************************************************************************************/

void Export_lm_3d(FaceInfo& faceInfo, const char* outFileName)
{
    ofstream outFile;
    outFile.open (outFileName);

    for(int i = 0; i<468; i++)
    {
        outFile << faceInfo.lm_3d[i][0] << "  ";
        outFile << faceInfo.lm_3d[i][1] << "  ";
        outFile << faceInfo.lm_3d[i][2] << "  ";
        outFile << endl;
    }
    
    outFile.close();
}

void Export_lm_2d(FaceInfo& faceInfo, const char* outFileName)
{
    ofstream outFile;
    outFile.open (outFileName);

    for(int i = 0; i<468; i++)
    {
        outFile << faceInfo.lm_2d[i][0] << "  ";
        outFile << faceInfo.lm_2d[i][1] << "  ";
        outFile << endl;
    }
    
    outFile.close();
}

void Export_LipRefinePts(FaceInfo& faceInfo, const char* outFileName)
{
    ofstream outFile;
    outFile.open (outFileName);

    for(int i = 0; i<80; i++)
    {
        outFile << faceInfo.lipRefinePts[i][0] << "  ";
        outFile << faceInfo.lipRefinePts[i][1] << "  ";
        outFile << endl;
    }
    
    outFile.close();
}

void Export_TwoEyes_RefinePts(FaceInfo& faceInfo, const char* outFileName)
{
    ofstream outFile;
    outFile.open (outFileName);

    for(int i = 0; i<71; i++)
    {
        outFile << faceInfo.leftEyeRefinePts[i][0] << "  ";
        outFile << faceInfo.leftEyeRefinePts[i][1] << "  ";
        outFile << endl;
    }
    
    for(int i = 0; i<71; i++)
    {
        outFile << faceInfo.rightEyeRefinePts[i][0] << "  ";
        outFile << faceInfo.rightEyeRefinePts[i][1] << "  ";
        outFile << endl;
    }
    
    outFile.close();
}

// 将目前的所有数据（也包含位姿数据）导出到一个文本文件中
// 数据在导出文件中分布的安排如下：
// 1. 按在FaceInfo定义时的数据成员先后来排序。
// 2. 每段数据前加一行文本标题，便于人工浏览。
void ExportLM_FullData(FaceInfo& faceInfo, const char* outFileName)
{
    ofstream outFile;
    outFile.open (outFileName);

    for(int i = 0; i<468; i++)
    {
        outFile << faceInfo.lm_3d[i][0] << "  ";
        outFile << faceInfo.lm_3d[i][1] << "  ";
        outFile << faceInfo.lm_3d[i][2] << "  ";
        outFile << endl;
    }
    
    for(int i = 0; i<468; i++)
    {
        outFile << faceInfo.lm_2d[i][0] << "  ";
        outFile << faceInfo.lm_2d[i][1] << "  ";
        outFile << endl;
    }
    
    for(int i = 0; i<71; i++)
    {
        outFile << faceInfo.leftEyeRefinePts[i][0] << "  ";
        outFile << faceInfo.leftEyeRefinePts[i][1] << "  ";
        outFile << endl;
    }
    
    for(int i = 0; i<71; i++)
    {
        outFile << faceInfo.rightEyeRefinePts[i][0] << "  ";
        outFile << faceInfo.rightEyeRefinePts[i][1] << "  ";
        outFile << endl;
    }
    
    for(int i = 0; i<80; i++)
    {
        outFile << faceInfo.lipRefinePts[i][0] << "  ";
        outFile << faceInfo.lipRefinePts[i][1] << "  ";
        outFile << endl;
    }
    
    
    outFile.close();
}

//-------------------------------------------------------------------------------------------
