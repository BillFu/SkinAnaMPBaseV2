//
//  FaceLmAttenExtract.cpp

/*******************************************************************************
Author: Fu Xiaoqiang
Date:   2022/9/13

********************************************************************************/

#include "FaceLmExtract.hpp"
#include "Utils.hpp"


/*
    Note: in the landmarks outputed by the tf lite model, the raw coordinate values
    NOT scaled into [0.0 1.0], instead in the [0.0 192.0].

    The returned lm_3d is measured in the input coordinate system of tf lite model,
    i.e., the values are in the range: [0.0, 192.0].

    The returned argument, lm_2d, is measured in the coordinate system of the source image!
*/
/*
void extractOutputLM(int origImgWidth, int origImgHeight,
    float* netLMOutBuffer, float lm_3d[468][3], int lm_2d[468][2])
{
    float scale_x = origImgWidth / (float)FACE_MESH_NET_INPUT_W;
    float scale_y = origImgHeight / (float)FACE_MESH_NET_INPUT_H;

    for(int i=0; i<468; i++)
    {
        float x = netLMOutBuffer[i*3];
        float y = netLMOutBuffer[i*3+1];
        float z = netLMOutBuffer[i*3+2];

        lm_3d[i][0] = x;
        lm_3d[i][1] = y;
        lm_3d[i][2] = z;

        lm_2d[i][0] = (int)(x*scale_x);
        lm_2d[i][1] = (int)(y*scale_y);
    }
}
*/

// Normal here means the coordinate value lies in [0.0 1.0]
void extractOutputLM(float* netLMOutBuffer, float lm_3d[468][3], double normal_lm_2d[468][2])
{
    double scale_x = 1.0 / (double)FACE_MESH_NET_INPUT_W;
    double scale_y = 1.0 / (double)FACE_MESH_NET_INPUT_H;

    for(int i=0; i<468; i++)
    {
        float x = netLMOutBuffer[i*3];
        float y = netLMOutBuffer[i*3+1];
        float z = netLMOutBuffer[i*3+2];

        lm_3d[i][0] = x;
        lm_3d[i][1] = y;
        lm_3d[i][2] = z;

        normal_lm_2d[i][0] = x*scale_x;
        normal_lm_2d[i][1] = y*scale_y;
    }
}

/*
void extractEyeRefinePts(int origImgWidth, int origImgHeight,
                   float* outBufEyeBow, int EyeBowPts[71][2])
{
    float scale_x = origImgWidth / (float)FACE_MESH_NET_INPUT_W;
    float scale_y = origImgHeight / (float)FACE_MESH_NET_INPUT_H;

    for(int i=0; i<71; i++)
    {
        float x = outBufEyeBow[i*2];
        float y = outBufEyeBow[i*2+1];

        EyeBowPts[i][0] = (int)(x*scale_x);
        EyeBowPts[i][1] = (int)(y*scale_y);
    }
}
*/

// Normal here means the coordinate value lies in [0.0 1.0]
void extractEyeRefinePts(float* outBufEyeBow, double NormalEyeBowPts[71][2])
{
    double scale_x = 1.0 / (double)FACE_MESH_NET_INPUT_W;
    double scale_y = 1.0 / (double)FACE_MESH_NET_INPUT_H;

    for(int i=0; i<71; i++)
    {
        float x = outBufEyeBow[i*2];
        float y = outBufEyeBow[i*2+1];

        NormalEyeBowPts[i][0] = x*scale_x;
        NormalEyeBowPts[i][1] = y*scale_y;
    }
}

/*
 outBufLipRefinePts: Input
 lipRefinePts: Output
 */
/*
void extractLipRefinePts(int origImgWidth, int origImgHeight,
                         float* outBufLipRefinePts, int lipRefinePts[80][2])
{
    float scale_x = origImgWidth / (float)FACE_MESH_NET_INPUT_W;
    float scale_y = origImgHeight / (float)FACE_MESH_NET_INPUT_H;

    for(int i=0; i<80; i++)
    {
        float x = outBufLipRefinePts[i*2];
        float y = outBufLipRefinePts[i*2+1];

        lipRefinePts[i][0] = (int)(x*scale_x);
        lipRefinePts[i][1] = (int)(y*scale_y);
    }
}
*/

void extractLipRefinePts(float* outBufLipRefinePts, double NormalLipRefinePts[80][2])
{
    double scale_x = 1.0 / (double)FACE_MESH_NET_INPUT_W;
    double scale_y = 1.0 / (double)FACE_MESH_NET_INPUT_H;

    for(int i=0; i<80; i++)
    {
        float x = outBufLipRefinePts[i*2];
        float y = outBufLipRefinePts[i*2+1];

        NormalLipRefinePts[i][0] = x*scale_x;
        NormalLipRefinePts[i][1] = y*scale_y;
    }
}

//-----------------------------------------------------------------------------------------

/******************************************************************************************
该函数的功能是，加载Face Mesh模型，生成深度网络，创建解释器并配置它。
return True if all is well done, otherwise reurn False and give the error reason.
numThreads: 解释器推理时可以使用的线程数量，最低为1.
*******************************************************************************************/

TF_LITE_MODEL LoadFaceMeshAttenModel(const char* faceMeshModelFileName)
{
    unique_ptr<FlatBufferModel> face_lm_model = FlatBufferModel::BuildFromFile(faceMeshModelFileName);
    return face_lm_model;
}
 
//-----------------------------------------------------------------------------------------

/******************************************************************************************
 目前的推理是一次性的，即从模型加载到解释器创建，到推理，到结果提取，这一流程只负责完成一副人脸的LM提取。
 以后要让前半段的结果长期存活，用于连续推理，以提高效率。
 Note: after invoking this function, return value and hasFace must be check!
*******************************************************************************************/
bool ExtractFaceLm(const TF_LITE_MODEL& face_lm_model, const Mat& srcImage,
                   bool needPadding, float vertPadRatio,
                   float confTh, bool& hasFace,
                   FaceInfo& faceInfo, string& errorMsg)
{
    //------------------******preprocessing******-----------------------------------------------------------------
    // padding the source image to get better performance
    Mat paddedImg;
    
    float alpha = 0.0;
    if(needPadding)
    {
        //float alpha = vertPadRatio; // deltaH / srcH
        //MakeSquareImageV2(srcImage, alpha,
        //                  paddedImg);
        // the values of faceBBox & faceCP taken from ISEMECO/cross_1.jpg
        Rect faceBBox(218, 867, 3407, 5039);
        Point2i faceCP(1894, 3158);
        
        GeoFixFVSrcImg(srcImage, faceBBox,
                       faceCP, 0.25, paddedImg);
    }
    else
    {
        paddedImg = srcImage.clone();
    }
    
    int padImgWidht = paddedImg.cols;
    int padImgHeight = paddedImg.rows;
    
    int netInputWidth = FACE_MESH_NET_INPUT_W;
    int netInputHeight = FACE_MESH_NET_INPUT_H;
    
    // Copy image to input tensor
    cv::Mat resized_image;  //, normal_image;
    // Not need to perform the convertion from BGR to RGB by the noticeable statements,
    // later it would be done in one trick way.
    cv::resize(paddedImg, resized_image, cv::Size(netInputWidth, netInputHeight), cv::INTER_NEAREST);

    paddedImg.release();
    
    //-------------------------*****enter inference*****---------------------------------------------------------
    // Initiate Interpreter
    INTERPRETER interpreter;
    tflite::ops::builtin::BuiltinOpResolver resolver;
    tflite::InterpreterBuilder(*face_lm_model.get(), resolver)(&interpreter);
    if (interpreter == nullptr)
    {
        errorMsg = string("failed to initiate the face lm interpreter.");
        return false;
    }
    else
        cout << "the face lm interpreter has been initialized Successfully!" << endl;

    if (interpreter->AllocateTensors() != kTfLiteOk)
    {
        errorMsg = string("Failed to allocate tensor.");
        return false;
    }
    else
        cout << "Succeeded to allocate tensor!" << endl;

    // Configure the interpreter
    interpreter->SetAllowFp16PrecisionForFp32(true);
    interpreter->SetNumThreads(1);
    
    // Get Input Tensor Dimensions
    int inTensorIndex = interpreter->inputs()[0];
    ///int netInputHeight = interpreter->tensor(inTensorIndex)->dims->data[1];
    ///int netInputWidth = interpreter->tensor(inTensorIndex)->dims->data[2];
    int channels = interpreter->tensor(inTensorIndex)->dims->data[3];
    
    
    float* inputTensorBuffer = interpreter->typed_input_tensor<float>(inTensorIndex);
    uint8_t* inImgMem = resized_image.ptr<uint8_t>(0);
    FeedInputWithNormalizedImage(inImgMem, inputTensorBuffer, netInputHeight, netInputWidth, channels);
    resized_image.release();
    cout << "ProcessInputWithFloatModel() is executed successfully!" << endl;

    // Inference
    interpreter->Invoke();  // perform the inference
    
    //int output_conf_ID = interpreter->outputs()[6];  // confidence
    float* sigmoidConfPtr = interpreter->typed_output_tensor<float>(1);
    
    float sigmoidConf = *sigmoidConfPtr;
    
    float confidence = 1.0 / (1.0 + exp(-sigmoidConf));
    cout << "face confidence: " << confidence << endl;
    if(confidence < confTh) // if confidence is too low, return immediately.
    {
        hasFace = false;
        return true;
    }
    
    faceInfo.confidence = confidence;
    hasFace = true;

    int output_lm_ID = interpreter->outputs()[0];
    cout << "output_landmarks ID: " << output_lm_ID << endl;
    
    float* LMOutBuffer = interpreter->typed_output_tensor<float>(0);

    /*
    The values in lm_3d are measured in the input coordinate system of our tf lite model,
    i.e., the values are in the range: [0.0, 192.0].
    The values in lm_2d are measured in the coordinate system of the source image! ----???
    */
    float lm_3d[NUM_PT_GENERAL_LM][3];
    NormalLmSet normalLmSet;
    extractOutputLM(LMOutBuffer,
                    lm_3d, normalLmSet.normal_lm_2d);

    cout << "extractOutputLM() is well done!" << endl;
    
    //double LNorEyeBowPts[NUM_PT_EYE_REFINE_GROUP][2];
    float* outBufLeftEyeRefinePts = interpreter->typed_output_tensor<float>(2);
    extractEyeRefinePts(outBufLeftEyeRefinePts, normalLmSet.LNorEyeBowPts);
    
    //double RNorEyeBowPts[NUM_PT_EYE_REFINE_GROUP][2];
    float* outBufRightEyeRefinePts = interpreter->typed_output_tensor<float>(3);
    extractEyeRefinePts(outBufRightEyeRefinePts, normalLmSet.RNorEyeBowPts);
    
    //double NorLipRefinePts[NUM_PT_LIP_REFINE_GROUP][2];
    float* outBufLipRefinePts = interpreter->typed_output_tensor<float>(1);
    extractLipRefinePts(outBufLipRefinePts, normalLmSet.NorLipRefinePts);
    
    //-------------------------*****exit inference*****---------------------------------------------------------

    //-------------------------*****postprocessing*****---------------------------------------------------------
    // reverse coordinate transform
    faceInfo.imgWidth = srcImage.cols;
    faceInfo.imgHeight = srcImage.rows;
    
    padCoord2SrcCoord(padImgWidht, padImgHeight,
                      srcImage.cols, srcImage.rows, alpha,
                      normalLmSet, faceInfo);

    errorMsg = "OK";
    return true;
}

//-----------------------------------------------------------------------------------------------
// dH: alpha * H
// dX: 0.5*(H + dH - W)
// H: the height of the source image
// W: the width of the source image
// dHH: 0.5 * dH
void padCoord2SrcCoord(//int srcW, int srcH,
                       int padImgWidht, int padImgHeight,
                       int dX, int dHH,
                       double normalX, double normalY, int& srcX, int& srcY)
{
    //srcX = normalX * srcW - dX;
    //srcY = normalY * srcH - dHH;
    srcX = normalX * padImgWidht - dX;
    srcY = normalY * padImgHeight - dHH;

}

void padCoord2SrcCoord(//int srcW, int srcH,
                       int padImgWidht, int padImgHeight,
                       int dX, int dHH,
                       const double normalPt[][2], int numPt, int srcPt[][2])
{
    for(int i=0; i<numPt; i++)
    {
        padCoord2SrcCoord(//srcW, srcH,
                          padImgWidht, padImgHeight,
                          dX, dHH,
                          normalPt[i][0], normalPt[i][1],
                          srcPt[i][0], srcPt[i][1]);
    }
}
/**************************************************************************************************
convert the coordinates of LM extracted from the Padded image into the coordinates
of source image space.
dummyFI: the coordiantes measured in padded image space.
srcSpaceFI: the coordinates measured in the source iamge space.
alpha: deltaH / srcH
***************************************************************************************************/
void padCoord2SrcCoord(int padImgWidht, int padImgHeight,
                       int srcW, int srcH, float alpha,
                       const NormalLmSet& normalLmSet,
                       
                       FaceInfo& srcSpaceFI)
{
    int dX = 0; // applied for the case without padding
    int dHH = 0; // applied for the case without padding
    
    if(padImgWidht == padImgHeight) // is a square, and has padding applied
    {
        int dH = (int)(alpha * srcH);
        dX = (srcH + dH - srcW)/2;
        dHH = dH/2;
    }
    
    padCoord2SrcCoord(//srcW, srcH,
                      padImgWidht, padImgHeight,
                      dX, dHH,
                      normalLmSet.normal_lm_2d, NUM_PT_GENERAL_LM, srcSpaceFI.lm_2d);

    padCoord2SrcCoord(//srcW, srcH,
                      padImgWidht, padImgHeight,
                      dX, dHH,
                      normalLmSet.LNorEyeBowPts, NUM_PT_EYE_REFINE_GROUP, srcSpaceFI.leftEyeRefinePts);
    
    padCoord2SrcCoord(//srcW, srcH,
                      padImgWidht, padImgHeight,
                      dX, dHH,
                      normalLmSet.RNorEyeBowPts, NUM_PT_EYE_REFINE_GROUP, srcSpaceFI.rightEyeRefinePts);
    
    padCoord2SrcCoord(//srcW, srcH,
                      padImgWidht, padImgHeight,
                      dX, dHH,
                      normalLmSet.NorLipRefinePts, NUM_PT_LIP_REFINE_GROUP, srcSpaceFI.lipRefinePts);

}
