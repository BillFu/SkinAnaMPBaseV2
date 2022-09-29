//
//  AnnoImage.cpp

/*******************************************************************************
Author: Fu Xiaoqiang
Date:   2022/9/11

********************************************************************************/

#include "AnnoImage.hpp"
#include "Utils.hpp"

/******************************************************************************************
本函数的功能是，将人脸位姿估计的结果打印在输入影像的拷贝上。
*******************************************************************************************/
void AnnoHeadPoseEst(Mat& annoImage, const FaceInfo& faceInfo)
{
    string pitch_str = convertFloatToStr2DeciDigits(faceInfo.headPose.pitch);
    string yaw_str = convertFloatToStr2DeciDigits(faceInfo.headPose.yaw);
    string roll_str = convertFloatToStr2DeciDigits(faceInfo.headPose.roll);

    // Print inference ms in input image
    //cv::putText(frame, "Infernce Time in ms: " + std::to_string(inference_time), cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 0, 255), 2);
    
    Scalar redColor(0, 0, 255);  // BGR
    cv::putText(annoImage, "pitch: " + pitch_str, Point(500, 50),
                FONT_HERSHEY_SIMPLEX, 2, redColor, 2);
    
    cv::putText(annoImage, "yaw: " + yaw_str, Point(500, 100),
                FONT_HERSHEY_SIMPLEX, 2, redColor, 2);
    
    cv::putText(annoImage, "roll: " + roll_str, Point(500, 150),
                FONT_HERSHEY_SIMPLEX, 2, redColor, 2);

}

/******************************************************************************************
本函数的功能是，将人脸一般关键点(468个)的结果打印在输入影像的拷贝上。
*******************************************************************************************/
void AnnoGeneralKeyPoints(Mat& annoImage, const FaceInfo& faceInfo, bool showIndices)
{
    cv::Scalar yellow(0, 255, 255); // (B, G, R)
    cv::Scalar blue(255, 0, 0);
    for(int i = 0; i < 468; i++)
    {
        // cv::Point center(faceInfo.lm_2d[i].x, faceInfo.lm_2d[i].y);
        cv::circle(annoImage, faceInfo.lm_2d[i], 5, blue, cv::FILLED);
        if(showIndices)
        {
            cv::putText(annoImage, to_string(i), faceInfo.lm_2d[i],
                        FONT_HERSHEY_SIMPLEX, 0.5, blue, 1);
        }
    }
    
    // 对应Dlib上点的序号为18, 22, 23, 27, 37, 40, 43, 46, 32, 36, 49, 55, 58, 9
    int face_2d_pts_indices[] = {46, 55, 285, 276, 33, 173,
        398, 263, 48, 278, 61, 291, 17, 199};  // indics in face lms in mediapipe.
        
    for(int i=0; i<14; i++)
    {
        int lm_index = face_2d_pts_indices[i];
        int x = faceInfo.lm_2d[lm_index].x;
        int y = faceInfo.lm_2d[lm_index].y;
        
        cv::Point center(x, y);
        cv::circle(annoImage, center, 5, yellow, cv::FILLED);
    }
}


//-----------------------------------------------------------------------------------------

/******************************************************************************************
本函数的功能是，将一个眉毛处的71个关键点的结果打印在标注影像上。
*******************************************************************************************/
void AnnoOneEyeRefinePts(Mat& annoImage, const FaceInfo& faceInfo, EyeID eyeID,
                         const Scalar& drawColor, bool showIndices)
{
    for(int i = 0; i < 71; i++)
    {
        int x, y;
        if(eyeID == LEFT_EYE)
        {
            x = faceInfo.lEyeRefinePts[i].x;
            y = faceInfo.lEyeRefinePts[i].y;
        }
        else // RightEyeID
        {
            x = faceInfo.rEyeRefinePts[i].x;
            y = faceInfo.rEyeRefinePts[i].y;
        }
        cv::Point center(x, y);
        cv::circle(annoImage, center, 1, drawColor, cv::FILLED);
        
        if(showIndices)
        {
            //Scalar redColor(0, 0, 255);  // BGR
            cv::putText(annoImage, to_string(i), Point(x, y),
                        FONT_HERSHEY_SIMPLEX, 0.5, drawColor, 1);
        }
    }
}

//-----------------------------------------------------------------------------------------

/******************************************************************************************
本函数的功能是，将眉毛处的172个关键点的结果打印在标注影像上。
*******************************************************************************************/
void AnnoTwoEyeRefinePts(Mat& annoImage, const FaceInfo& faceInfo,
                         const Scalar& drawColor, bool showIndices)
{
    AnnoOneEyeRefinePts(annoImage, faceInfo, LEFT_EYE, drawColor, showIndices);
    AnnoOneEyeRefinePts(annoImage, faceInfo, RIGHT_EYE, drawColor, showIndices);
}

//-----------------------------------------------------------------------------------------

/******************************************************************************************
本函数的功能是，将嘴唇处80个精细关键点的结果打印在标注影像上。
*******************************************************************************************/
void AnnoLipRefinePts(Mat& annoImage, const FaceInfo& faceInfo,
                      const Scalar& drawColor, bool showIndices)
{
    for(int i = 0; i < 80; i++)
    {
        int x = faceInfo.lipRefinePts[i].x;
        int y = faceInfo.lipRefinePts[i].y;
        
        cv::Point center(x, y);
        cv::circle(annoImage, center, 1, drawColor, cv::FILLED);
        
        if(showIndices)
        {
            Scalar redColor(0, 0, 255);  // BGR
            cv::putText(annoImage, to_string(i), Point(x, y),
                        FONT_HERSHEY_SIMPLEX, 0.5, redColor, 1);
        }
    }
}

//-----------------------------------------------------------------------------------------
/******************************************************************************************
本函数的功能是，将所有信息（包括人脸关键点提取和位姿估计的结果）打印在输入影像的拷贝上。
*******************************************************************************************/
void AnnoAllLmInfo(Mat& annoImage, const FaceInfo& faceInfo,
                   const string& annoFile)
{
    AnnoGeneralKeyPoints(annoImage, faceInfo, true);
    
    Scalar yellowColor(255, 0, 0);
    AnnoTwoEyeRefinePts(annoImage, faceInfo, yellowColor, true);
    
    Scalar pinkColor(255, 0, 255);
    AnnoLipRefinePts(annoImage, faceInfo, pinkColor, true);
    
    AnnoHeadPoseEst(annoImage, faceInfo);
    
    //string poseImgFile = "pose_" + fileNameBone + ".png";
    //fs::path poseImgFullPath = outDir / poseImgFile;
    imwrite(annoFile, annoImage);
}
