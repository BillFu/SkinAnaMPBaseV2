//
//  AnnoImage.cpp

/*******************************************************************************
Author: Fu Xiaoqiang
Date:   2022/9/11

********************************************************************************/

#include "AnnoImage.hpp"
#include "Utils.hpp"

/******************************************************************************************
本函数的功能是，将人脸关键点提取和位姿估计的结果打印在输入影像的拷贝上。
*******************************************************************************************/
void AnnoHeadPoseEst(const Mat& srcImage, Mat& annoImage,
                     int lm_2d[468][2], float pitch, float yaw, float roll)
{
    annoImage = srcImage.clone();
    
    cv::Scalar colorCircle2(255, 0, 0); // (B, G, R)
    for(int i = 0; i < 468; i++)
    {
        cv::Point center(lm_2d[i][0], lm_2d[i][1]);
        cv::circle(annoImage, center, 5, colorCircle2, cv::FILLED);
    }
    
    // 对应Dlib上点的序号为18, 22, 23, 27, 37, 40, 43, 46, 32, 36, 49, 55, 58, 9
    int face_2d_pts_indices[] = {46, 55, 285, 276, 33, 173,
        398, 263, 48, 278, 61, 291, 17, 199};  // indics in face lms in mediapipe.
        
    cv::Scalar yellow(0, 255, 255); // (B, G, R)

    for(int i=0; i<14; i++)
    {
        int lm_index = face_2d_pts_indices[i];
        cv::Point center(lm_2d[lm_index][0], lm_2d[lm_index][1]);
        cv::circle(annoImage, center, 5, yellow, cv::FILLED);
    }
    
    string pitch_str = convertFloatToStr2DeciDigits(pitch);
    string yaw_str = convertFloatToStr2DeciDigits(yaw);
    string roll_str = convertFloatToStr2DeciDigits(roll);

    // Print inference ms in input image
    //cv::putText(frame, "Infernce Time in ms: " + std::to_string(inference_time), cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 0, 255), 2);
    cv::putText(annoImage, "pitch: " + pitch_str, Point(500, 50),
                FONT_HERSHEY_SIMPLEX, 2, Scalar(0, 0, 255), 2);
    
    cv::putText(annoImage, "yaw: " + yaw_str, Point(500, 100),
                FONT_HERSHEY_SIMPLEX, 2, Scalar(0, 0, 255), 2);
    
    cv::putText(annoImage, "roll: " + roll_str, Point(500, 150),
                FONT_HERSHEY_SIMPLEX, 2, Scalar(0, 0, 255), 2);

}
