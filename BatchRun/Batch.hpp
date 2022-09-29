//
//  Utils.hpp
//
//
/*
本模块提供一些简单的辅助性函数。
 
Author: Fu Xiaoqiang
Date:   2022/9/29
*/

#ifndef BATCH_HPP
#define BATCH_HPP

#include "opencv2/opencv.hpp"

using namespace std;
using namespace cv;

namespace fs = std::filesystem;

// collect all the .jpg image files under the specified directory and its sub-directories recursively
// This function will invoke itself recursively.
void ScanSrcImagesInDir(const fs::path imgRootDir, vector<string>& jpgImgSet, int level = 0);

//-------------------------------------------------------------------------------------------

// focus is saving the anno files
void ProImgInBatch(const vector<string>& jpgImgSet,
                   const fs::path& srcRootDir, const fs::path& outRootDir);


#endif /* end of BATCH_HPP */
