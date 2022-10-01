//
//  main.cpp
//  BatchRun
//
//  Created by meicet on 2022/9/29.
//

#include <iostream>
#include <string>
#include <fstream>
#include <sstream>
#include <vector>

#include <filesystem>
#include <algorithm>

#include <opencv2/opencv.hpp>

#include "nlohmann/json.hpp"

using namespace std;
using namespace cv;

using json = nlohmann::json;
namespace fs = std::filesystem;

#include "../SkinAnaMPBase/FaceBgSeg/FaceBgSeg.hpp"
#include "../SkinAnaMPBase/FaceLmExtractV2.hpp"
#include "Batch.hpp"

int main(int argc, const char * argv[])
{
// Get Model label and input image
    if (argc != 2)
    {
        cout << "{target} config_file" << endl;
        return 0;
    }
    string errorMsg;
    
    json config_json;
    ifstream jConfFile(argv[1]);
    jConfFile >> config_json;        
        
    string segModelFile = config_json.at("SegModelFile");
    string classColorFile = config_json.at("ClassColorFile");
    string faceMeshModelFile = config_json.at("FaceMeshAttenModel");

    string srcImgRootDir = config_json.at("SrcImgRootDir");
    string annoImgRootDir = config_json.at("AnnoImgRootDir");
        
    std::cout << "---------------------------------------------------" << endl;
    std::cout << segModelFile << endl;
    std::cout << classColorFile << endl;
    std::cout << "srcImgRootDir: " << srcImgRootDir << endl;
    std::cout << "annoImgRootDir: " << annoImgRootDir << endl;
    std::cout << "---------------------------------------------------" << endl;
    
    // 这个函数在程序初始化时要调用一次，并确保返回true之后，才能往下进行
    bool isOK = FaceBgSegmentor::Initialize(segModelFile, classColorFile);
    if(!isOK)
    {
        cout << "Failed to Initialize FaceBgSegmentor: " << endl;
        return 0;
    }
    
    isOK = LoadFaceMeshModel(faceMeshModelFile.c_str(), errorMsg);
    if(!isOK)
    {
        cout << errorMsg << endl;
        return 0;
    }
    else
        cout << "Succeeded to load face mesh with attention model file: "
            << faceMeshModelFile << endl;
    
    const fs::path imgRootDir{srcImgRootDir};
    const fs::path outRootDir{annoImgRootDir};
    
    vector<string> jpgImgSet;
    ScanSrcImagesInDir(imgRootDir, jpgImgSet);
    
    //cout << "------------------------------------" << endl;
    ProImgInBatch(jpgImgSet, imgRootDir, outRootDir);
    
    return 0;
}
