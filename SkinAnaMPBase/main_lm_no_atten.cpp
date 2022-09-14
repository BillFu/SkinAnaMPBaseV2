#include <fstream>
#include <string>
#include <sstream>
#include <vector>

#include <opencv2/opencv.hpp>

#include "nlohmann/json.hpp"

#include "HeadPoseEst.hpp"
//#include "FaceDetect.hpp"
#include "FaceLmExtract.hpp"
#include "AnnoImage.hpp"

using namespace tflite;
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
        
    string faceMeshModelFile = config_json.at("FaceMeshModelFile");
    string srcImgFile = config_json.at("SourceImage");
    string annoPoseImgFile = config_json.at("AnnoPoseImage");
    
    TF_LITE_MODEL faceMeshModel = LoadFaceMeshModel(faceMeshModelFile.c_str());
    if(faceMeshModel == nullptr)
    {
        cout << "Failed to load face mesh model file: " << faceMeshModelFile << endl;
        return 0;
    }

    string errorMsg;
    
    // Load Input Image
    auto srcImg = cv::imread(srcImgFile.c_str());
    if(srcImg.empty())
    {
        cout << "Failed to load input iamge: " << srcImgFile << endl;
        return 0;
    }
    else
        cout << "Succeeded to load image: " << srcImgFile << endl;
    
    int srcImgWidht = srcImg.cols;
    int srcImgHeight = srcImg.rows;

    float lm_3d[468][3];
    int lm_2d[468][2];

    float confThresh = 0.75;
    bool hasFace = false;
    float confidence = 0.0;
    bool isOK = ExtractFaceLm(faceMeshModel, srcImg,
                              confThresh, hasFace, confidence,
                              lm_3d,  lm_2d, errorMsg);
    if(!isOK)
    {
        cout << "Error Happened to extract face LM: " << errorMsg << endl;
        return 0;
    }
    else
        cout << "Succeeded to extract lm!" << endl;

    float pitch, yaw, roll;
    EstHeadPose(srcImgWidht, srcImgHeight, lm_2d, pitch, yaw, roll);
    
    Mat PoseAnnoImage;
    AnnoHeadPoseEst(srcImg, PoseAnnoImage, lm_2d, pitch, yaw, roll);

    imwrite(annoPoseImgFile, PoseAnnoImage);
    
    return 0;
}
