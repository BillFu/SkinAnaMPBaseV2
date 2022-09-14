#include <fstream>
#include <string>
#include <sstream>
#include <vector>

#include <opencv2/opencv.hpp>


#include "HeadPoseEst.hpp"
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


/*
 * Need three arguments:
 */
int main(int argc, char **argv)
{
    // Get Model label and input image
    if (argc != 4)
    {
        cout << "{target} modelfile image anno_file" << endl;
        return 0;
    }

    const char* faceMeshModelFileName = argv[1];
    const char* imageFile = argv[2];
    const char* annoFile = argv[3];

    /*
    INTERPRETER faceMeshInterpreter;
    string errorMsg;
    bool isOK = CreatInterpreter(faceMeshModelFileName, faceMeshInterpreter,
                          1, // number of threads would be used when to inference
                          errorMsg);
    if(!isOK)
    {
        cout << "Error Happened: " << errorMsg << endl;
        return 0;
    }
    else
        cout << "Succeeded to load face mesh tf lite and create the interpreter!" << endl;
    */
    
    string errorMsg;
    bool isOK = CreatInterpreter(faceMeshModelFileName,
                          1, // number of threads would be used when to inference
                          errorMsg);
    if(!isOK)
    {
        cout << "Error Happened: " << errorMsg << endl;
        return 0;
    }
    else
        cout << "Succeeded to load face mesh tf lite and create the interpreter!" << endl;

    // Load Input Image
    auto srcImg = cv::imread(imageFile);
    if(srcImg.empty())
    {
        cout << "Failed to load input iamge: " << imageFile << endl;
        return 0;
    }
    else
        cout << "Succeeded to load image!" << endl;

    int srcImgWidht = srcImg.cols;
    int srcImgHeight = srcImg.rows;

    float lm_3d[468][3];
    int lm_2d[468][2];

    ExtractFaceLm(srcImg, lm_3d, lm_2d);

    float pitch, yaw, roll;
    EstHeadPose(srcImgWidht, srcImgHeight, lm_2d, pitch, yaw, roll);
    cout << "pitch: " << pitch << endl;
    cout << "yaw: " << yaw << endl;
    cout << "roll: " << roll << endl;

    Mat PoseAnnoImage;
    AnnoHeadPoseEst(srcImg, PoseAnnoImage, lm_2d, pitch, yaw, roll);

    imwrite(annoFile, PoseAnnoImage);

    return 0;
}
