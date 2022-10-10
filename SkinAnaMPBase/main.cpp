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

#include "FaceBgSeg/FaceBgSeg.hpp"

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
    string srcImgFile = config_json.at("SourceImage");
    string outDir = config_json.at("OutDir");
    
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
    Mat srcImage = cv::imread(srcImgFile.c_str());
    if(srcImage.empty())
    {
        cout << "Failed to load input iamge: " << srcImgFile << endl;
        return 0;
    }
    else
        cout << "Succeeded to load image: " << srcImgFile << endl;
    
    string fileBoneName = GetFileBoneName(srcImgFile);
    fs::path outParePath(outDir);
    
    // FP: full path
    string segAnnoImgFP = BuildOutImgFileName(
            outParePath, fileBoneName, "seg_");
    FaceSegResult segResult;
    SegImage(srcImage, segResult);
    DrawSegOnImage(srcImage, 0.5,
        segResult, segAnnoImgFP.c_str());
    
    cout << "source image has been segmented!" << endl;

    FaceInfo faceInfo;

    float confThresh = 0.75;
    bool hasFace = false;
    isOK = ExtractFaceLm(srcImage, confThresh, segResult, hasFace,
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
    
    EstHeadPose(srcImage.size(), faceInfo);
    
    Mat annoLmImage = srcImage.clone();
    
    //string annoLmImgFile = BuildOutImgFileName(
    //        outParePath, fileBoneName, "lm_");
    //AnnoAllLmInfo(annoLmImage, faceInfo, annoLmImgFile);

    Scalar yellowColor(255, 0, 0);
    AnnoTwoEyeRefinePts(annoLmImage, faceInfo, yellowColor, true);
    //AnnoGenKeyPoints(annoLmImage, faceInfo, true);

    string annoLmImgFile = BuildOutImgFileName(
            outParePath, fileBoneName, "lm_");
    imwrite(annoLmImgFile.c_str(), annoLmImage);

    ForgeMaskAnnoPackDebug(srcImage, annoLmImage,
                      outParePath, fileBoneName,
                      faceInfo, segResult);

    
    /*
    ForgeMaskAnnoPackV2(srcImage, 
                        outParePath, fileBoneName,
                        faceInfo, segResult);
    */
    
    return 0;
}
