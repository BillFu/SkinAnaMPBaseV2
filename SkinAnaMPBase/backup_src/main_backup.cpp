#include <fstream>
#include <string>
#include <sstream>
#include <vector>

#include <opencv2/opencv.hpp>

#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/string_util.h"
#include "tensorflow/lite/examples/label_image/get_top_n.h"
#include "tensorflow/lite/model.h"

#include "HeadPoseEst.hpp"
#include "FaceLmExtract.hpp"

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

// return a string that present a float with 2 decimal digits.
// for example, return "3.14" for 3.1415927
string convertFloatToStr2DeciDigits(float value)
{
    std::stringstream stream;
    stream << std::fixed << std::setprecision(2) << value;
    std::string out_str = stream.str();
    return out_str;
}

/*
 * Need three arguments:
 */
int main(int argc, char **argv)
{
    // Get Model label and input image
    if (argc != 4)
    {
        cout << "{target} modelfile image anno_file" << endl;
        exit(-1);
    }

    const char* modelFileName = argv[1];
    const char* imageFile = argv[2];
    const char* annoFile = argv[3];

    // Load Model
    unique_ptr<FlatBufferModel> face_lm_model = FlatBufferModel::BuildFromFile(modelFileName);
    if (face_lm_model == nullptr)
    {
        fprintf(stderr, "failed to load tf lite model\n");
        exit(-1);
    }
    else
        cout << "TF lite model has been Successfully loaded!" << endl;

    // Initiate Interpreter
    unique_ptr<tflite::Interpreter> interpreter;
    tflite::ops::builtin::BuiltinOpResolver resolver;
    tflite::InterpreterBuilder(*face_lm_model.get(), resolver)(&interpreter);
    if (interpreter == nullptr)
    {
        cout << "Failed to initiate the interpreter." << endl;
        exit(-1);
    }
    else
        cout << "the interpreter has been initialized Successfully!" << endl;

    if (interpreter->AllocateTensors() != kTfLiteOk)
    {
        cout << "Failed to allocate tensor." << endl;
        exit(-1);
    }
    else
        cout << "Succeeded to allocate tensor!" << endl;

    // Configure the interpreter
    interpreter->SetAllowFp16PrecisionForFp32(true);
    interpreter->SetNumThreads(1);

    // Get Input Tensor Dimensions
    int inTensorIndex = interpreter->inputs()[0];
    auto netInputHeight = interpreter->tensor(inTensorIndex)->dims->data[1];
    auto netInputWidth = interpreter->tensor(inTensorIndex)->dims->data[2];
    auto channels = interpreter->tensor(inTensorIndex)->dims->data[3];
    //cout << "netInputWidth: " << netInputWidth << endl;
    //cout << "netInputHeight: " << netInputHeight << endl;
    //cout << "channels: " << channels << endl;

    // Load Input Image
    auto src_image = cv::imread(imageFile);
    if(src_image.empty())
    {
        fprintf(stderr, "Failed to load input iamge!\n");
        exit(-1);
    }
    else
        cout << "Succeeded to load image!" << endl;

    int srcImgWidht = src_image.cols;
    int srcImgHeight = src_image.rows;

    // Copy image to input tensor
    cv::Mat resized_image;  //, normal_image;
    // Not need to perform the convertion from BGR to RGB by the noticeable statements,
    // later it would be done in one trick way.
    cv::resize(src_image, resized_image, cv::Size(netInputWidth, netInputHeight), cv::INTER_NEAREST);

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

    if(interpreter->tensor(output_lm_ID)->type == kTfLiteFloat32)
        cout << "output lm is kTfLiteFloat32" << endl;

    //cout << "original image Width : " << src_image.cols << endl;
    //cout << "original image Height: " << src_image.rows << endl;

    float lm_3d[468][3];
    int   lm_2d[468][2];

    float* netLMOutBuffer = interpreter->typed_output_tensor<float>(0);

    /*
    The values in lm_3d are measured in the input coordinate system of our tf lite model,
    i.e., the values are in the range: [0.0, 192.0].
    The values in lm_2d are measured in the coordinate system of the source image!
    */
    extractOutputLM(src_image.cols, src_image.rows, netLMOutBuffer, lm_3d, lm_2d);

    cout << "extractOutputLM() is well done!" << endl;

    float pitch, yaw, roll;
    EstHeadPose(srcImgWidht, srcImgHeight, lm_2d, pitch, yaw, roll);
    cout << "pitch: " << pitch << endl;
    cout << "yaw: " << yaw << endl;
    cout << "roll: " << roll << endl;

    cv::Scalar colorCircle2(255, 0, 0); // (B, G, R)
    for(int i = 0; i < 468; i++)
    {
        cv::Point center(lm_2d[i][0], lm_2d[i][1]);
        cv::circle(src_image, center, 5, colorCircle2, cv::FILLED);
    }
    
    string pitch_str = convertFloatToStr2DeciDigits(pitch);
    string yaw_str = convertFloatToStr2DeciDigits(yaw);
    string roll_str = convertFloatToStr2DeciDigits(roll);

    // Print inference ms in input image
    //cv::putText(frame, "Infernce Time in ms: " + std::to_string(inference_time), cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 0, 255), 2);
    cv::putText(src_image, "pitch: " + pitch_str, Point(500, 50),
                FONT_HERSHEY_SIMPLEX, 2, Scalar(0, 0, 255), 2);
    
    cv::putText(src_image, "yaw: " + yaw_str, Point(500, 100),
                FONT_HERSHEY_SIMPLEX, 2, Scalar(0, 0, 255), 2);
    
    cv::putText(src_image, "roll: " + roll_str, Point(500, 150),
                FONT_HERSHEY_SIMPLEX, 2, Scalar(0, 0, 255), 2);

    // Display image
    //cv::imshow("Output", src_image);
    //cv::waitKey(0);
    
    imwrite(annoFile, src_image);

    return 0;
}
