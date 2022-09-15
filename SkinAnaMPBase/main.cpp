#include <fstream>
#include <string>
#include <sstream>
#include <vector>

#include <opencv2/opencv.hpp>

#include "nlohmann/json.hpp"

#include "HeadPoseEst.hpp"
//#include "FaceLmAttenExtract.hpp"
#include "AnnoImage.hpp"
//#include "EXIF_Extractor.hpp"
#include "LM_loader.hpp"
#include "DetectRegion.hpp"

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
    
    /*
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
    */
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
    FillData(faceInfo);

    //EstHeadPose(srcImgWidht, srcImgHeight, faceInfo);
    
    Mat annoImage = srcImage.clone();

    AnnoGeneralKeyPoints(annoImage, faceInfo);
    
    /*
    Scalar yellowColor(0, 255, 255);
    AnnoTwoEyeRefinePts(annoImage, faceInfo, yellowColor);

    Scalar pinkColor(255, 0, 255);
    AnnoLipRefinePts(annoImage, faceInfo, pinkColor);

    AnnoHeadPoseEst(annoImage, faceInfo);
    */
    imwrite(annoPoseImgFile, annoImage);
    
    Mat outMask;
    ForgeSkinMask(srcImgWidht, srcImgHeight, faceInfo, outMask);
    
    string faceMaskImgFile = config_json.at("FaceContourImage");

    OverlayMaskOnImage(srcImage, outMask,
                        "face_contour", faceMaskImgFile.c_str());

    Mat mouthMask;
    ForgeMouthMask(srcImgWidht, srcImgHeight, faceInfo, mouthMask);
    
    string mouthMaskImgFile = config_json.at("MouthContourImage");

    OverlayMaskOnImage(srcImage, mouthMask,
                        "mouth_contour", mouthMaskImgFile.c_str());

    return 0;
}
