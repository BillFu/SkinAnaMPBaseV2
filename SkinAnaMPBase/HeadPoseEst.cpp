//
//  HeadPoseEst.cpp

/*******************************************************************************
Author: Fu Xiaoqiang
Date:   2022/9/10

本模块采用14个点人脸关键点，利用PnP算法来计算人脸的位姿。
2D关键点来自MediaPipe中的face mesh模型的推理结果（实现是利用Tensorflow Lite），
3D人脸参考模型来自网络。
参考博文：https://www.programminghunter.com/article/58132010290/
********************************************************************************/

#include "HeadPoseEst.hpp"
#include "Utils.hpp"


// the 14 points selected from the 3D reference face model.
// 对应Dlib上点的序号为18, 22, 23, 27, 37, 40, 43, 46, 32, 36, 49, 55, 58, 9
float ref_face_3d_14[14][3] = {
                        {6.825897, 6.760612, 4.402142},
                        {1.330353, 7.122144, 6.903745},
                        {-1.330353, 7.122144, 6.903745},
                        {-6.825897, 6.760612, 4.402142},
                        {5.311432, 5.485328, 3.987654},
                        {1.789930, 5.393625, 4.413414},
                        {-1.789930, 5.393625, 4.413414},
                        {-5.311432, 5.485328, 3.987654},
                        {2.005628, 1.409845, 6.165652},
                        {-2.005628, 1.409845, 6.165652},
                        {2.774015, -2.080775, 5.048531},
                        {-2.774015, -2.080775, 5.048531},
                        {0.000000, -3.116408, 6.097667},
                        {0.000000, -7.415691, 4.070434}};


/******************************************************************************************
srcImgWidht, srcImgHeight: the width and height of the input source image.
lm_2d: as the input argument, extracted from the source image, and measured in 
the coordinate system of the source image. 
pitch, yaw, roll: are output arguments, measured in degrees.
*******************************************************************************************/
void EstHeadPose(int srcImgWidht, int srcImgHeight,
                 FaceInfo& faceInfo)
{
    // 对应Dlib上点的序号为18, 22, 23, 27, 37, 40, 43, 46, 32, 36, 49, 55, 58, 9
    int face_2d_pts_indices[] = {46, 55, 285, 276, 33, 173, 
        398, 263, 48, 278, 61, 291, 17, 199};  // indics in face lms in mediapipe. 
    
    //int face_2d_pts[14][2];  // one of the input arguments into PnP algorithm
    vector<Point2d> face_2d_pts;
    for(int i=0; i<14; i++)
    {
        int lm_index = face_2d_pts_indices[i];
        double x = faceInfo.lm_2d[lm_index][0];
        double y = faceInfo.lm_2d[lm_index][1];
        face_2d_pts.push_back(Point2d(x, y));
    }

    vector<Point3d> face_3d_pts;
    for(int i=0; i<14; i++)
    {
        face_3d_pts.push_back(Point3d(ref_face_3d_14[i][0], ref_face_3d_14[i][1], ref_face_3d_14[i][2]));
    }

    // The camera intrinsic matrix
    float theta_30_in_rad = 0.5236; //60.0 / 2 * np.pi / 180.0, half of horizontal FOV (here is 60 degree)
    float tan_theta_30 = 0.57735;  
    float focal_length = srcImgWidht / (2.0 * tan_theta_30);

    double camera_matrix[3][3] = {
        {focal_length, 0, srcImgWidht / 2.0}, 
        {0, focal_length, srcImgHeight / 2.0}, 
        {0.0, 0.0, 1.0}};

    cv::Mat cameraMatrix(3, 3, cv::DataType<double>::type, camera_matrix);
    cout << "cameraMatrix: " << cameraMatrix << std::endl;

    double zeros_four[] = {0.0, 0.0, 0.0, 0.0};
    cv::Mat distCoeffs(4, 1, cv::DataType<double>::type, zeros_four);
    
    cv::Mat rot_vec(3, 1, cv::DataType<double>::type);
    cv::Mat trans_vec(3, 1, cv::DataType<double>::type);
 
    cv::solvePnP(face_3d_pts, face_2d_pts, cameraMatrix, distCoeffs, rot_vec, trans_vec);

    cout << "solvePnP() is Done!" << std::endl;

    // get rotation matrix
    Mat rot_mat; 
    cv::Rodrigues(rot_vec, rot_mat);
    cout << "Rodrigues() is Done!" << std::endl;
    
    cout << "rot_vec: " << rot_vec << std::endl;
    cout << "rot_mat: " << rot_mat << std::endl;

    Mat mtxR, mtxQ;
    // the values in eluer_angles are measured in degree. 
    Vec3d eluer_angles= cv::RQDecomp3x3(rot_mat, mtxR, mtxQ);
    cout << "RQDecomp3x3() is Done!" << std::endl;

    faceInfo.pitch = eluer_angles[0];
    faceInfo.yaw = eluer_angles[1];
    faceInfo.roll = eluer_angles[2];
    
    cout << "pitch: " << faceInfo.pitch << std::endl;
    cout << "yaw: " << faceInfo.yaw << std::endl;
    cout << "roll: " << faceInfo.roll << std::endl;
}
