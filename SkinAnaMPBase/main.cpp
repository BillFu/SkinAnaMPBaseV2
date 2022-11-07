#include <fstream>
#include <string>
#include <sstream>
#include <vector>

#include <opencv2/opencv.hpp>

#include "nlohmann/json.hpp"

#include "HeadPoseEst.hpp"
#include "FaceLmExtractV2.hpp"
#include "AnnoImage.hpp"
#include "Mask/SkinFeatureMask.hpp"
#include "Utils.hpp"

#include "FaceBgSeg/FaceBgSegV2.hpp"
#include "Wrinkle/Wrinkle.hpp"

#include "Common.hpp"

using namespace std;

/*************************************************************************
This is a experiment program to use the tensorflow lite model to
extract the face landmarks from the input image, and then to estimate the
head pose.
Now before extracting the Lms, the face/background segmentation would be invoked
fristly, and by exploiting the face primary info to shift and pad the input image
, which would be feeded into the face mesh net for inference, to improve
the outcome of the net.
 
Author: Fu Xiaoqiang
Date:   2022/9/29
**************************************************************************/

using json = nlohmann::json;

#ifdef TEST_RUN
string outDir("");
string wrkOutDir("");
#endif

int main(int argc, char **argv)
{
    // Get Model label and input image
    if (argc != 2)
    {
        cout << "{target} config_file" << endl;
        return 0;
    }
    
    string errorMsg;
    json config_json;            // 创建 json 对象
    ifstream jfile(argv[1]);
    jfile >> config_json;        // 以文件流形式读取 json 文件
        
    string segModelFile = config_json.at("SegModelFile");
    string classColorFile = config_json.at("ClassColorFile");
    
    string faceMeshAttenModelFile = config_json.at("FaceMeshAttenModel");
    string crossImgFile = config_json.at("CrossImage");
    
#ifdef TEST_RUN
    outDir = config_json.at("OutDir");
    wrkOutDir = config_json.at("WrkOutDir");
#endif
    
    // 这个函数在程序初始化时要调用一次，并确保返回true之后，才能往下进行
    bool isOK = FaceBgSegmentor::Initialize(segModelFile, classColorFile);
    if(!isOK)
    {
        cout << "Failed to Initialize FaceBgSegmentor: " << endl;
        return 0;
    }
    
    isOK = LoadFaceMeshModel(faceMeshAttenModelFile.c_str(), errorMsg);
    if(!isOK)
    {
        cout << "Failed to load face mesh model file: "
            << faceMeshAttenModelFile << endl;
        return 0;
    }
        
    // Load Input Image
    Mat crossImage = cv::imread(crossImgFile.c_str());
    if(crossImage.empty())
    {
        cout << "Failed to load input iamge: " << crossImgFile << endl;
        return 0;
    }
    else
        cout << "Succeeded to load image: " << crossImgFile << endl;
    
    //string fileBoneName = GetFileBoneName(crossImgFile);
    fs::path outParePath(outDir);
    
    // FP: full path
    string segAnnoImgFP = BuildOutImgFNV2(outParePath, "seg.png");
    FaceSegRst segResult;
    FaceBgSegmentor segmentor;
    segmentor.SegImage(crossImage, segResult);
    DrawSegOnImage(crossImage, 0.5,
        segResult, segAnnoImgFP.c_str());
    
    cout << "cross Image has been segmented!" << endl;
    
    FaceInfo faceInfo;

    float confThresh = 0.75;
    bool hasFace = false;
    isOK = ExtractFaceLm(crossImage, confThresh, segResult, hasFace,
                              faceInfo, errorMsg);
    if(!isOK)
    {
        cout << "Error Happened to extract face LM: " << errorMsg << endl;
        return 0;
    }
    else
        cout << "Succeeded to extract lm!" << endl;
    
    if(!hasFace)
    {
        cout << "No face found!" << endl;
        return 0;
    }
    
    EstHeadPose(crossImage.size(), faceInfo);
    
    Mat annoLmImage = crossImage.clone();
    
    string LmImgFile = BuildOutImgFNV2(
            outParePath, "lm.png");
    //AnnoAllLmInfo(annoLmImage, faceInfo, LmImgFile);
    AnnoGenKeyPoints(annoLmImage, faceInfo, true);
    
    Scalar yellowColor(255, 0, 0);
    //AnnoTwoEyeRefinePts(annoLmImage, faceInfo, yellowColor, true);
    imwrite(LmImgFile.c_str(), annoLmImage);

    DetRegPackage detRegPack;
    ForgeDetRegPack(crossImage, annoLmImage, outParePath, faceInfo, segResult, detRegPack);
    
    crossImage.release();
    
    string paraImgFile = config_json.at("ParallelImage");
    Mat paraImage = cv::imread(crossImgFile.c_str());
    if(paraImage.empty())
    {
        cout << "Failed to load parallel source iamge: " << paraImgFile << endl;
        return 0;
    }
    else
        cout << "Succeeded to load parallel source image: " << paraImgFile << endl;
    
    CONTOURS deepWrkConts;
    // test the wrinkle detecting algorithm
    Mat wrkGaborRespMap;
    
    DetectWrinkle(paraImage, segResult.faceBBox, detRegPack.wrkRegGroup,
                  deepWrkConts, wrkGaborRespMap);
    
    return 0;
}
