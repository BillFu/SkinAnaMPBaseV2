#include <fstream>
#include <string>
#include <sstream>
#include <vector>

#include <opencv2/opencv.hpp>

#include "nlohmann/json.hpp"

#include "HeadPoseEst.hpp"
#include "FaceLmExtract.hpp"
#include "AnnoImage.hpp"
#include "LM_loader.hpp"
#include "Mask/DetectRegion.hpp"
#include "Mask/EyebowMask.hpp"
#include "Utils.hpp"

//using namespace tflite;
using namespace std;

/*************************************************************************
This is a experiment program to use the tensorflow lite model to
extract the face landmarks from the input image, and then to estimate the
head pose.
 
Author: Fu Xiaoqiang
Date:   2022/9/10
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
    
    json config_json;            // 创建 json 对象
    ifstream jfile(argv[1]);
    jfile >> config_json;        // 以文件流形式读取 json 文件
        
    string faceMeshAttenModelFile = config_json.at("FaceMeshAttenModelFile");
    string srcImgFile = config_json.at("SourceImage");
    string annoPoseImgFile = config_json.at("AnnoPoseImage");
    
    //ExtractEXIF(srcImgFile.c_str());
    
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
    Mat srcImage = cv::imread(srcImgFile.c_str());
    if(srcImage.empty())
    {
        cout << "Failed to load input iamge: " << srcImgFile << endl;
        return 0;
    }
    else
        cout << "Succeeded to load image: " << srcImgFile << endl;
    
    int srcImgW = srcImage.cols;
    int srcImgH = srcImage.rows;
    FaceInfo faceInfo;

    float confThresh = 0.75;
    bool hasFace = false;
    float confidence = 0.0;
    float vertPadRatio = 0.14;
    bool isOK = ExtractFaceLm(faceMeshModel, srcImage, vertPadRatio,
                              confThresh, hasFace, confidence,
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
    
    EstHeadPose(srcImgW, srcImgH, faceInfo);
    
    Mat annoImage = srcImage.clone();
    
    AnnoGeneralKeyPoints(annoImage, faceInfo);

    Scalar yellowColor(255, 0, 0);
    AnnoTwoEyeRefinePts(annoImage, faceInfo, yellowColor, true);
    
    Scalar pinkColor(255, 0, 255);
    AnnoLipRefinePts(annoImage, faceInfo, pinkColor, true);
    
    AnnoHeadPoseEst(annoImage, faceInfo);
    
    imwrite(annoPoseImgFile, annoImage);
    
    Mat skinMask;
    string faceMaskImgFile = config_json.at("FaceContourImage");
    ForgeSkinMask(faceInfo, skinMask);
    OverlayMaskOnImage(srcImage, skinMask,
                        "face_contour", faceMaskImgFile.c_str());

    Mat mouthMask;
    string mouthMaskImgFile = config_json.at("MouthContourImage");
    ForgeMouthMask(faceInfo, mouthMask);
    OverlayMaskOnImage(srcImage, mouthMask,
                        "mouth_contour", mouthMaskImgFile.c_str());
    
    Mat eyebowsMask;
    string eyebowsMaskImgFile = config_json.at("EyebowsContourImage"); // 注意复数形式表示双眉
    ForgeTwoEyebowsMask(faceInfo, eyebowsMask);
    OverlayMaskOnImage(annoImage, eyebowsMask,
                        "eyebows_contour", eyebowsMaskImgFile.c_str());
    
    string eyeFullMaskImgFile = config_json.at("EyeFullContourImage");
    Mat eyesFullMask = ForgeTwoEyesFullMask(faceInfo);
    OverlayMaskOnImage(annoImage, eyesFullMask,
                        "eye_full_contour", eyeFullMaskImgFile.c_str());
    
    return 0;
}
