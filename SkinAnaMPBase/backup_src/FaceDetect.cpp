//
//  FaceLmExtract.cpp

/*******************************************************************************
Author: Fu Xiaoqiang
Date:   2022/9/11

********************************************************************************/

#include "FaceDetect.hpp"
#include "Utils.hpp"


/*
从896个Scores中选出最大的那个Score和它对应的index。
 */
void extractOutputScore(float* netOutBuffer_Score, float scoreThresh,
                        vector<float>& FilteredScores,
                        vector<int>& FilteredIndics)
{
    float sigmaThresh = -log(1.0/scoreThresh - 1.0);
    
    for(int i=0; i<896; i++)
    {
        float x = netOutBuffer_Score[i];
        if(x > sigmaThresh)
        {
            FilteredScores.push_back(x);
            FilteredIndics.push_back(i);
        }
    }
}

/*
    Note: in the landmarks outputed by the tf lite model, the raw coordinate values
    NOT scaled into [0.0 1.0], instead in the [0.0 192.0].

    The returned lm_3d is measured in the input coordinate system of tf lite model,
    i.e., the values are in the range: [0.0, 192.0].

    The returned argument, lm_2d, is measured in the coordinate system of the source image!
*/
/*
void extractOutputBox(int origImgWidth, int origImgHeight,
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
*/

//-----------------------------------------------------------------------------------------

/******************************************************************************************
该函数的功能是，加载Face Mesh模型，生成深度网络，创建解释器并配置它。
return True if all is well done, otherwise reurn False and give the error reason.
numThreads: 解释器推理时可以使用的线程数量，最低为1.
*******************************************************************************************/

TF_LITE_MODEL LoadFaceDetectModel(const char* faceDetectModelFileName)
{
    unique_ptr<FlatBufferModel> face_detect_model = FlatBufferModel::BuildFromFile(faceDetectModelFileName);
    return face_detect_model;
}
 
//-----------------------------------------------------------------------------------------

/******************************************************************************************

*******************************************************************************************/
bool DetectFace(const TF_LITE_MODEL& faceDetectModel, const Mat& srcImage,
                bool& hasFace, float& confidence,
                   string& errorMsg)
{
    // Initiate Interpreter
    INTERPRETER interpreter;
    tflite::ops::builtin::BuiltinOpResolver resolver;
    tflite::InterpreterBuilder(*faceDetectModel.get(), resolver)(&interpreter);
    if (interpreter == nullptr)
    {
        errorMsg = string("failed to initiate the face datect interpreter.");
        return false;
    }
    else
        cout << "the face detect interpreter has been initialized Successfully!" << endl;

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
    int netInputHeight = interpreter->tensor(inTensorIndex)->dims->data[1];  // will be 128
    int netInputWidth = interpreter->tensor(inTensorIndex)->dims->data[2];  // will be 128
    int channels = interpreter->tensor(inTensorIndex)->dims->data[3];
    
    // Copy image to input tensor
    cv::Mat resized_image;  //, normal_image;
    // Not need to perform the convertion from BGR to RGB by the noticeable statements,
    // later it would be done in one trick way.
    cv::resize(srcImage, resized_image, cv::Size(netInputWidth, netInputHeight), cv::INTER_NEAREST);

    float* inputTensorBuffer = interpreter->typed_input_tensor<float>(inTensorIndex);
    uint8_t* inImgMem = resized_image.ptr<uint8_t>(0);
    FeedInputWithQuantizedImage(inImgMem, inputTensorBuffer, netInputHeight, netInputWidth, channels);
    cout << "ProcessInputWithFloatModel() is executed successfully!" << endl;

    // Inference
    std::chrono::steady_clock::time_point start, end;
    start = chrono::steady_clock::now();
    interpreter->Invoke();  // perform the inference
    end = chrono::steady_clock::now();
    auto inference_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    // Get the two Output items
    int output_lm_ID = interpreter->outputs()[0];
    //cout << "output_landmarks ID: " << output_lm_ID << endl;

    int output_conf_ID = interpreter->outputs()[1];
    //cout << "output confidence ID: " << output_conf_ID << endl;

    float* netOutBuffer_Score = interpreter->typed_output_tensor<float>(1);

    vector<float> filteredScores;
    vector<int> filteredIndics;
    //extractOutputScore(netOutBuffer_Score, maxScore, maxScoreIndex);
    
    float scoreThresh = 0.7;
    extractOutputScore(netOutBuffer_Score, scoreThresh,
                    filteredScores, filteredIndics);
    
    /*
    for(float f : filteredScores)
      cout << "sigma score = " << f << endl;
    for(int i : filteredIndics)
      cout << "i = " << i << endl;
    */
    
    float* netOutBuffer_BoxKeyPt = interpreter->typed_output_tensor<float>(0);

    
    //The values in lm_3d are measured in the input coordinate system of our tf lite model,
    //i.e., the values are in the range: [0.0, 192.0].
    //The values in lm_2d are measured in the coordinate system of the source image!
    
    //extractOutputBoxKeyPt(srcImage.cols, srcImage.rows, netBoxKeyPtOutBuffer, lm_3d, lm_2d);

    //cout << "extractOutputBox() is well done!" << endl;
    
    errorMsg = "OK";
    return true;
}
