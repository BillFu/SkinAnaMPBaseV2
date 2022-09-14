//
//  FaceLmExtract.cpp

/*******************************************************************************
Author: Fu Xiaoqiang
Date:   2022/9/11

********************************************************************************/

#include "FaceLmExtract.hpp"
#include "Utils.hpp"

INTERPRETER interpreter;

/*
https://github.com/bferrarini/FloppyNet_TRO/blob/master/FloppyNet_TRO/TRO_pretrained/RPI4/src/lce_cnn.cc
see the function: ProcessInputWithFloatModel()
*/
void FeedInputWithFloatData(uint8_t* imgDataPtr, float* netInputBuffer, int H, int W, int C)
{
    for (int y = 0; y < H; ++y)
    {
        for (int x = 0; x < W; ++x)
        {
            if (C == 3)
            {
                // here is TRICK: BGR ==> RGB, be careful with the channel indices coupling
                // happened on the two sides of assignment operator.
                // another thing to be noted: make the value of each channel of the pixels
                // in [0.0 1.0], dividing by 255.0
                int yWC = y * W * C;
                int xC = x * C;
                int yWC_xC = yWC + xC;
                *(netInputBuffer + yWC_xC + 0) = *(imgDataPtr + yWC_xC + 2) / 255.0f;
                *(netInputBuffer + yWC_xC + 1) = *(imgDataPtr + yWC_xC + 1) / 255.0f;
                *(netInputBuffer + yWC_xC + 2) = *(imgDataPtr + yWC_xC + 0) / 255.0f;
            }
            else
            {
                // only one channel, gray scale image
                {
                    *(netInputBuffer + y * W + x ) = *(imgDataPtr + y * W + x) / 255.0f;
                }
            }
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
void extractOutputLM(int origImgWidth, int origImgHeight,
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


//-----------------------------------------------------------------------------------------

/******************************************************************************************
该函数的功能是，加载Face Mesh模型，生成深度网络，创建解释器并配置它。
return True if all is well done, otherwise reurn False and give the error reason.
numThreads: 解释器推理时可以使用的线程数量，最低为1.
*******************************************************************************************/

bool CreatInterpreter(const char* faceMeshModelFileName,
                      int numThreads, string& errorMsg)
{
    // Load Model
    unique_ptr<FlatBufferModel> face_lm_model = FlatBufferModel::BuildFromFile(faceMeshModelFileName);
    if (face_lm_model == nullptr)
    {
        errorMsg = string("failed to load face mesh tf lite model: ") + faceMeshModelFileName;
        return false;
    }
    else
        cout << "TF lite model has been Successfully loaded!" << endl;

    // Initiate Interpreter
    tflite::ops::builtin::BuiltinOpResolver resolver;
    tflite::InterpreterBuilder(*face_lm_model.get(), resolver)(&interpreter);
    if (interpreter == nullptr)
    {
        errorMsg = string("failed to initiate the tf lite interpreter.");
        return false;
    }
    else
        cout << "the interpreter has been initialized Successfully!" << endl;

    if (interpreter->AllocateTensors() != kTfLiteOk)
    {
        errorMsg = string("Failed to allocate tensor.");
        return false;
    }
    else
        cout << "Succeeded to allocate tensor!" << endl;

    // Configure the interpreter
    interpreter->SetAllowFp16PrecisionForFp32(true);
    interpreter->SetNumThreads(numThreads);

    errorMsg = "OK";
    return true;
}

 
//-----------------------------------------------------------------------------------------

/******************************************************************************************
 目前的推理是一次性的，即从模型加载到解释器创建，到推理，到结果提取，这一流程只负责完成一副人脸的LM提取。
 以后要让前半段的结果长期存活，用于连续推理，以提高效率。
*******************************************************************************************/
void ExtractFaceLm(const Mat& srcImage,
                   float lm_3d[468][3], int lm_2d[468][2])
{
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
    FeedInputWithFloatData(inImgMem, inputTensorBuffer, netInputHeight, netInputWidth, channels);
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

    float* netLMOutBuffer = interpreter->typed_output_tensor<float>(0);

    /*
    The values in lm_3d are measured in the input coordinate system of our tf lite model,
    i.e., the values are in the range: [0.0, 192.0].
    The values in lm_2d are measured in the coordinate system of the source image!
    */
    extractOutputLM(srcImage.cols, srcImage.rows, netLMOutBuffer, lm_3d, lm_2d);

    cout << "extractOutputLM() is well done!" << endl;
}
