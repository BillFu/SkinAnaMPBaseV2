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

/*
 
 struct FaceInfo
 {
     //[n][0] for x, [n][1] for y
     // measured in source iamge coordinate system
     float lm_3d[468][3];  // x, y, z
     int   lm_2d[468][2];  // x, y，与上面的lm_3d中相同，只是数据类型不同
     
     int leftEyeRefinePts[71][2];
     int rightEyeRefinePts[71][2];
     
     int lipRefinePts[80][2];
     
     float pitch;  // rotate with x-axis
     float yaw;    // rotate with y-axis
     float roll;   // rotate with z-axis
 };
 
 */

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


//int leftEyeRefinePts[71][2];
//int rightEyeRefinePts[71][2];

void Export_TwoEyes_RefinePts(FaceInfo& faceInfo, const char* outFileName)
{
    ofstream outFile;
    outFile.open (outFileName);

    for(int i = 0; i<71; i++)
    {
        outFile << faceInfo.lipRefinePts[i][0] << "  ";
        outFile << faceInfo.lipRefinePts[i][1] << "  ";
        outFile << endl;
    }
    
    outFile.close();
}


/*
void FillData(FaceInfo& faceInfo)
{
    for(int i = 0; i<468; i++)
    {
        faceInfo.lm_2d[i][0] = general_lm[i][0];
        faceInfo.lm_2d[i][1] = general_lm[i][1];
    }
}
*/

//-------------------------------------------------------------------------------------------
