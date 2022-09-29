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
#include "Mask/FundamentalMask.hpp"
#include "Mask/EyebowMask.hpp"
#include "Mask/SkinFeatureMask.hpp"
#include "Mask/ForeheadMask.hpp"
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
    //float confidence = 0.0;
    bool needPadding = true;
    float vertPadRatio = 0.0;
    bool isOK = ExtractFaceLm(faceMeshModel, srcImage,
                              needPadding, vertPadRatio,
                              confThresh, hasFace,
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
    
    AnnoGeneralKeyPoints(annoImage, faceInfo, true);
    imwrite(annoPoseImgFile, annoImage);

    Scalar yellowColor(255, 0, 0);
    AnnoTwoEyeRefinePts(annoImage, faceInfo, yellowColor, true);
    
    Scalar pinkColor(255, 0, 255);
    AnnoLipRefinePts(annoImage, faceInfo, pinkColor, true);
    
    AnnoHeadPoseEst(annoImage, faceInfo);
    
    imwrite(annoPoseImgFile, annoImage);
    
    Mat skinMask(srcImgH, srcImgW, CV_8UC1, cv::Scalar(0));
    string faceMaskImgFile = config_json.at("FaceContourImage");
    ForgeSkinMask(faceInfo, skinMask);
    OverlayMaskOnImage(annoImage, skinMask,
                        "face_contour", faceMaskImgFile.c_str());

    Mat mouthMask(srcImgH, srcImgW, CV_8UC1, cv::Scalar(0));
    string mouthMaskImgFile = config_json.at("MouthContourImage");
    float expanRatio = 0.3;
    ForgeMouthMask(faceInfo, expanRatio, mouthMask);
    OverlayMaskOnImage(srcImage, mouthMask,
                        "mouth_contour", mouthMaskImgFile.c_str());
    
    Mat eyebowsMask(srcImgH, srcImgW, CV_8UC1, cv::Scalar(0));
    string eyebowsMaskImgFile = config_json.at("EyebowsContourImage"); // 注意复数形式表示双眉
    ForgeTwoEyebowsMask(faceInfo, eyebowsMask);
    OverlayMaskOnImage(annoImage, eyebowsMask,
                        "eyebows_contour", eyebowsMaskImgFile.c_str());
    
    Mat eyesFullMask(srcImgH, srcImgW, CV_8UC1, cv::Scalar(0));
    string eyeFullMaskImgFile = config_json.at("EyeFullContourImage");
    ForgeTwoEyesFullMask(faceInfo, eyesFullMask);
    OverlayMaskOnImage(annoImage, eyesFullMask,
                        "eye_full_contour", eyeFullMaskImgFile.c_str());
    annoImage.release();

    //string fhMaskAnnoFile = config_json.at("ForeheadMaskImage");
    Mat fhMask(srcImgH, srcImgW, CV_8UC1, cv::Scalar(0));
    ForgeForeheadMask(faceInfo, fhMask);
    
    Mat annoImage2 = srcImage.clone();
    AnnoGeneralKeyPoints(annoImage2, faceInfo, true);
    //OverlayMaskOnImage(annoImage2, fhMask,
    //                    "forehead mask", fhMaskAnnoFile.c_str());

    //string noseMaskAnnoFile = config_json.at("NoseMaskImage");
    Mat noseMask(srcImgH, srcImgW, CV_8UC1, cv::Scalar(0));
    ForgeNoseMask(faceInfo, noseMask);
    
    //Mat combinedMask = noseMask | fhMask;
    //OverlayMaskOnImage(annoImage2, combinedMask,
    //                    "combined mask", noseMaskAnnoFile.c_str());
    
    //string fleMaskAnnoFile = config_json.at("FaceLowThEyeImage");
    Mat fleMask(srcImgH, srcImgW, CV_8UC1, cv::Scalar(0));
    ForgeFaceLowThEyeMask(faceInfo, fleMask);
    //OverlayMaskOnImage(annoImage2, fleMask,
    //                    "face low th eye", fleMaskAnnoFile.c_str());
    
    string poreMaskAnnoFile = config_json.at("PoreMask");
    Mat poreMask(srcImgH, srcImgW, CV_8UC1, cv::Scalar(0));
    ForgePoreMaskV2(faceInfo, fleMask, fhMask, eyesFullMask,
                    mouthMask, noseMask,
                    poreMask);
    OverlayMaskOnImage(annoImage2, poreMask,
                        "pore mask", poreMaskAnnoFile.c_str());
    
    return 0;
}
