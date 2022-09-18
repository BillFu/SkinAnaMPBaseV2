//
//  FaceLmAttenExtract.cpp

/*******************************************************************************
Author: Fu Xiaoqiang
Date:   2022/9/13

********************************************************************************/

#include "FaceLmAttenExtract.hpp"
#include "Utils.hpp"


/*
    Note: in the landmarks outputed by the tf lite model, the raw coordinate values
    NOT scaled into [0.0 1.0], instead in the [0.0 192.0].

    The returned lm_3d is measured in the input coordinate system of tf lite model,
    i.e., the values are in the range: [0.0, 192.0].

    The returned argument, lm_2d, is measured in the coordinate system of the source image!
*/
void extractOutputAttenLM(int origImgWidth, int origImgHeight,
    float* netLMOutBuffer, float lm_3d[468][3], int lm_2d[468][2])
{
    float scale_x = origImgWidth / 192.0f;
    float scale_y = origImgHeight / 192.0f;

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

void extractEyeRefinePts(int origImgWidth, int origImgHeight,
                   float* outBufEyeBow, int EyeBowPts[71][2])
{
    float scale_x = origImgWidth / 192.0f;
    float scale_y = origImgHeight / 192.0f;

    for(int i=0; i<71; i++)
    {
        float x = outBufEyeBow[i*2];
        float y = outBufEyeBow[i*2+1];

        EyeBowPts[i][0] = (int)(x*scale_x);
        EyeBowPts[i][1] = (int)(y*scale_y);
    }
}

/*
 outBufLipRefinePts: Input
 lipRefinePts: Output
 */
void extractLipRefinePts(int origImgWidth, int origImgHeight,
                         float* outBufLipRefinePts, int lipRefinePts[80][2])
{
    float scale_x = origImgWidth / 192.0f;
    float scale_y = origImgHeight / 192.0f;

    for(int i=0; i<80; i++)
    {
        float x = outBufLipRefinePts[i*2];
        float y = outBufLipRefinePts[i*2+1];

        lipRefinePts[i][0] = (int)(x*scale_x);
        lipRefinePts[i][1] = (int)(y*scale_y);
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
*******************************************************************************************/
bool ExtractFaceLmAtten(const TF_LITE_MODEL& face_lm_model, const Mat& srcImage,
                    float confThresh, bool& hasFace, float& confidence,
                    FaceInfo& faceInfo, string& errorMsg)
{
    faceInfo.imgWidth = srcImage.cols;
    faceInfo.imgHeight = srcImage.rows;

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
    int netInputHeight = interpreter->tensor(inTensorIndex)->dims->data[1];
    int netInputWidth = interpreter->tensor(inTensorIndex)->dims->data[2];
    int channels = interpreter->tensor(inTensorIndex)->dims->data[3];
    
    // Copy image to input tensor
    cv::Mat resized_image;  //, normal_image;
    // Not need to perform the convertion from BGR to RGB by the noticeable statements,
    // later it would be done in one trick way.
    cv::resize(srcImage, resized_image, cv::Size(netInputWidth, netInputHeight), cv::INTER_NEAREST);

    float* inputTensorBuffer = interpreter->typed_input_tensor<float>(inTensorIndex);
    uint8_t* inImgMem = resized_image.ptr<uint8_t>(0);
    FeedInputWithNormalizedImage(inImgMem, inputTensorBuffer, netInputHeight, netInputWidth, channels);
    cout << "ProcessInputWithFloatModel() is executed successfully!" << endl;

    // Inference
    //std::chrono::steady_clock::time_point start, end;
    //start = chrono::steady_clock::now();
    interpreter->Invoke();  // perform the inference
    //end = chrono::steady_clock::now();
    //auto inference_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    int output_conf_ID = interpreter->outputs()[6];  // confidence
    //cout << "output confidence ID: " << output_conf_ID << endl;
    float* confidencePtr = interpreter->typed_output_tensor<float>(1);
    
    float sigmaConf = *confidencePtr;
    
    confidence = 1.0 / (1.0 + exp(-sigmaConf));
    cout << "face confidence: " << confidence << endl;

    int output_lm_ID = interpreter->outputs()[0];
    cout << "output_landmarks ID: " << output_lm_ID << endl;
    
    float* LMOutBuffer = interpreter->typed_output_tensor<float>(0);

    /*
    The values in lm_3d are measured in the input coordinate system of our tf lite model,
    i.e., the values are in the range: [0.0, 192.0].
    The values in lm_2d are measured in the coordinate system of the source image!
    */
    extractOutputAttenLM(srcImage.cols, srcImage.rows, LMOutBuffer,
                         faceInfo.lm_3d, faceInfo.lm_2d);

    cout << "extractOutputLM() is well done!" << endl;
    
    float* outBufLeftEyeRefinePts = interpreter->typed_output_tensor<float>(2);
    extractEyeRefinePts(srcImage.cols, srcImage.rows, outBufLeftEyeRefinePts, faceInfo.leftEyeRefinePts);
    
    float* outBufRightEyeRefinePts = interpreter->typed_output_tensor<float>(3);
    extractEyeRefinePts(srcImage.cols, srcImage.rows, outBufRightEyeRefinePts, faceInfo.rightEyeRefinePts);
    
    float* outBufLipRefinePts = interpreter->typed_output_tensor<float>(1);

    extractLipRefinePts(srcImage.cols, srcImage.rows, outBufLipRefinePts, faceInfo.lipRefinePts);
    
    errorMsg = "OK";
    return true;
}
