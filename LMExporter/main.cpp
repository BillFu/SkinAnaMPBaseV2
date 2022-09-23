//
//  main.cpp
//  LMExporter
//
//  Created by meicet on 2022/9/17.
//

/*************************************************************************
extract the face landmarks from the input image, and then export them to a text file.
 
Author: Fu Xiaoqiang
Date:   2022/9/17
**************************************************************************/
#include <fstream>
#include <string>
#include <sstream>
#include <vector>
#include <iostream>

#include <opencv2/opencv.hpp>

#include "nlohmann/json.hpp"

#include "../SkinAnaMPBase/Common.hpp"
#include "../SkinAnaMPBase/HeadPoseEst.hpp"
#include "../SkinAnaMPBase/FaceLmExtract.hpp"

#include "LM_Exporter.hpp"

using json = nlohmann::json;


//using namespace tflite;
using namespace std;


int main(int argc, const char * argv[])
{
    // Get Model label and input image
    if (argc != 2)
    {
        cout << "{target} config_file" << endl;
        return 0;
    }
    
    json config_json;            // 创建 json 对象
    ifstream jfile(argv[1]);
    jfile >> config_json;        // 以文件流形式读取 json 文件
        
    string faceMeshAttenModelFile = config_json.at("FaceMeshAttenModelFile");
    string srcImgFile = config_json.at("SourceImage");
    string outLmFile = config_json.at("OutLmFile");
    
    cout << "faceMeshAttenModelFile: " << faceMeshAttenModelFile << endl;
    cout << "srcImgFile: " << srcImgFile << endl;
    cout << "outLmFile: " << outLmFile << endl;

    //--------------First Stage: Extraction--------------------------------------------------
    TF_LITE_MODEL faceMeshModel = LoadFaceMeshAttenModel(faceMeshAttenModelFile.c_str());
    if(faceMeshModel == nullptr)
    {
        cout << "Failed to load face mesh with attention model file: "
            << faceMeshAttenModelFile << endl;
        return 0;
    }
    else
        cout << "Succeeded to load face mesh with attention model file: "
            << faceMeshAttenModelFile << endl;
    
    string errorMsg;
    
    // Load Input Image
    auto srcImage = cv::imread(srcImgFile.c_str());
    if(srcImage.empty())
    {
        cout << "Failed to load input iamge: " << srcImgFile << endl;
        return 0;
    }
    else
        cout << "Succeeded to load image: " << srcImgFile << endl;
    
    int srcImgWidht = srcImage.cols;
    int srcImgHeight = srcImage.rows;

    FaceInfo faceInfo;

    float confThresh = 0.75;
    bool hasFace = false;
    float confidence = 0.0;
    bool isOK = ExtractFaceLmAtten(faceMeshModel, srcImage,
                              confThresh, hasFace, confidence,
                                   faceInfo, errorMsg);
    if(!isOK)
    {
        cout << "Error Happened to extract face LM: " << errorMsg << endl;
        return 0;
    }
    else
        cout << "Succeeded to extract lm!" << endl;
    
    EstHeadPose(srcImgWidht, srcImgHeight, faceInfo);
    
    cout << "Succeeded to invoke EstHeadPose()" << endl;

    //--------------Second Stage: Export--------------------------------------------------
    
    ExportLM_FullData(faceInfo, outLmFile.c_str());
    cout << "Succeeded to export LM data into file: " << outLmFile << endl;

    return 0;
}
