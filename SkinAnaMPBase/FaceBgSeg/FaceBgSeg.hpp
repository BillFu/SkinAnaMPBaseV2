//
//  FaceBgSeg.hpp
//
//
/****************************************************************************************
 * 本模块由王少峰的人脸/背景分割推理引擎改造而来。											    *
 * 模型也是王少峰训练出来的。																*
 *																						*
 * Author: Wang Shaofeng, Fu Xiaoqiang													*
 * Date:   2022/9/25																	*
 ****************************************************************************************/

#ifndef FACE_BG_SEG_HPP
#define FACE_BG_SEG_HPP

#include "opencv2/opencv.hpp"
#include "../Common.hpp"

using namespace std;
using namespace cv;


typedef vector<Point> CONTOUR;
typedef vector<CONTOUR> CONTOURS;

enum SEG_FACE_BG_LABELS
{
    SEG_BG_LABEL = 0,
    SEG_FACE_LABEL,
    SEG_EYEBROW_LABEL,
    SEG_EYE_LABEL,
    SEG_BEARD_LABEL
};


// if FacePrimaryInfo.eyeAreaDiffRatio > EyeAreaDiffRation_TH, then this face will be
// considered as in the profile view.
// when FacePrimaryInfo.eyeAreaDiffRatio <= threshold, we think this face is in front view
#define EyeAreaDiffRation_TH  0.35  // this value seems a bit higher, later should be dropped.

class FaceBgSegmentor
{
public:
    static bool isNetLoaded;
        
    static bool Initialize(const string& modelFileName, const string& classColorFileName);
    
    FaceBgSegmentor(); //,int width,int height);
    ~FaceBgSegmentor();

    //now make this function more thinner,
    //just extract the segmentation labels, a 512*512 matrix with one channel.
    //coloring and resize has been elimated.
    void Segment(const Mat& srcImage, FaceSegResult& segResult); //, uchar segLabel[SEG_NET_OUTPUT_SIZE][SEG_NET_OUTPUT_SIZE]);

    // 将分割结果以彩色Table渲染出来，并放大到原始图像尺度
    static Mat RenderSegLabels(const Size& imgSize, const Mat& segLabels);
    
    void CalcFaceBBox(FaceSegResult& segResult);

    // this function includes: calcuate the center point of two eys, area of two eyes,
    // and the ratio of area difference, final determine the face is in front view
    // or profile view.
    // The center point of face refers to the center of the line connecting the centers of wo eyes.
    // in the profile view, the CP of face esitmated cannot be used for the bad precision.
    void CalcEyePts(FaceSegResult& segResult);
    
    // return a binary labels image: 0 for background, and 255 for face
    // (including all its components), with the same size as the source image
    static Mat CalcFaceBgBiLabel(const FaceSegResult& segResult); 

    // FB: face and background
    static Mat CalcFBBiLabExBeard(const FaceSegResult& segResult);

private:
    
    // 这个函数在程序初始化时要调用一次，并确保返回true之后，才能往下进行
    static bool LoadSegModel(const string& modelFileName);

    // return true if OK; otherwise return false
    static bool LoadClassColorTable(const string& classColorFileName);
    
    // convert BBox in net output space into the space of source image
    void ScaleUpBBox(const Rect& inBBox, Rect& outBBox);
    void ScaleUpPoint(const Point2i& inPt, Point2i& outPt);

    // Bounding Box is in the coordinate system of the source image.
    void CalcFaceBBoxImpl(const Mat& segLabels, Rect& BBox);

    // formula: ratio = abs(a1-a2) / max(a1, a2)
    float calcEyeAreaDiffRatio(int a1, int a2);
    
private:
    int srcImgH;
    int srcImgW;
    
    //Mat segLabels; //after segmentation, would be filled with 512*512, uchar data, only one channel.
    
    static dnn::Net segNet;
    static vector<Vec3b> classColorTable;
};

// blend segment labels image with source iamge:
// result = alpha * segLabels + (1-alpha) * srcImage
// alpha lies in [0.0 1.0]
void DrawSegOnImage(const Mat& srcImg,
                    float alpha, const FaceSegResult& facePriInfo,
                    const char* outImgFileName);


// call this function outside this module
// if needToSaveAnno is true, then annoImgFile must to be a valid
// image file name, otherwise a empty string is a good choice.
void SegImage(const Mat& srcImage, FaceSegResult& facePriInfo); //,
              //bool needToSaveAnno, const string& annoImgFile);

#endif /* end of FACE_BG_SEG_HPP */
