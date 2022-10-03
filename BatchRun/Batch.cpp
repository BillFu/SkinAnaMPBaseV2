//
//  Utils.cpp

/*******************************************************************************
Author: Fu Xiaoqiang
Date:   2022/9/27

********************************************************************************/

#include <iostream>
#include <string>
#include <fstream>
#include <sstream>
#include <vector>

#include <filesystem>
#include <algorithm>

#include "Batch.hpp"
#include "../SkinAnaMPBase/Common.hpp"
#include "../SkinAnaMPBase/FaceBgSeg/FaceBgSeg.hpp"
#include "../SkinAnaMPBase/FaceLmExtractV2.hpp"
#include "../SkinAnaMPBase/HeadPoseEst.hpp"
#include "../SkinAnaMPBase/AnnoImage.hpp"
#include "../SkinAnaMPBase/Utils.hpp"
#include "../SkinAnaMPBase/Mask/SkinFeatureMask.hpp"
#include "../SkinAnaMPBase/Geometry.hpp"


namespace fs = std::filesystem;

// collect all the .jpg image files under the specified directory and its sub-directories recursively
// This function will invoke itself recursively.
void ScanSrcImagesInDir(const fs::path imgRootDir, vector<string>& jpgImgSet, int level)
{
    for (const auto& entry : fs::directory_iterator(imgRootDir))
    {
        const auto filenameStr = entry.path().filename().string();
        const auto fullPath = entry.path().string();
        if (entry.is_directory())
        {
            //std::cout << std::setw(level * 3) << "" << filenameStr << '\n';
            ScanSrcImagesInDir(entry, jpgImgSet, level + 1);
        }
        else if (entry.is_regular_file())
        {
            string fileExt =  entry.path().extension();
            if(fileExt == ".jpg")
                jpgImgSet.push_back(fullPath);
        }
        //else
            //std::cout << std::setw(level * 3) << "" << " [?]" << filenameStr << '\n';
    }
}

//-------------------------------------------------------------------------------------------

void ProOneImg(const string& srcImgFile,
               const fs::path& outDir,
               const string& fileNameBone)
{
    string errorMsg;

    // Load Input Image
    Mat srcImage = cv::imread(srcImgFile.c_str());
    if(srcImage.empty())
    {
        cout << "Failed to load input iamge: " << srcImgFile << endl;
        return;
    }
    else
        cout << "Succeeded to load image: " << srcImgFile << endl;
    
    PadImgWithRC4Div(srcImage);

    FaceSegResult segResult;
    string segImgFile = "seg_" + fileNameBone + ".png";
    fs::path segImgFP = outDir / segImgFile;  // FP: full path
    SegImage(srcImage, segResult);
    DrawSegOnImage(srcImage, 0.5,
        segResult, segImgFP.c_str());

    FaceInfo faceInfo;

    float confThresh = 0.75;
    bool hasFace = false;
    bool isOK = ExtractFaceLm(srcImage,
                        confThresh, segResult, hasFace,
                        faceInfo, errorMsg);
    if(!isOK)
    {
        cout << "Error Happened to extract face LM: " << errorMsg << endl;
        return;
    }
    //else
    //    cout << "Succeeded to extract lm!" << endl;
    
    if(!hasFace)
    {
        cout << "No face found!" << endl;
        return;
    }
    
    EstHeadPose(srcImage.size(), faceInfo);
    
    int jawWidth = CalcLowerJawWidth(faceInfo, segResult.segLabels);
    string jawImgFile = "jaw_" + fileNameBone + ".png";
    fs::path jawImgFP = outDir / jawImgFile;
    AnnoLowerJaw(srcImage, faceInfo,
                 jawWidth, segResult.faceBBox.width, jawImgFP.string());

    Mat annoLmImage = srcImage.clone();
    
    string lmImgFile = "pose_" + fileNameBone + ".png";
    fs::path lmImgFullPath = outDir / lmImgFile;
    AnnoAllLmInfo(annoLmImage, faceInfo, lmImgFullPath.string());
    
    string fileBoneName = GetFileBoneName(srcImgFile);
    ForgeMaskAnnoPack(srcImage, annoLmImage,
                      outDir, fileBoneName,
                      faceInfo, segResult);
}

void PrepareDirFile(const string& srcImgFullPath,
                    const fs::path& srcRootDir,
                    const fs::path& outRootDir,
                    fs::path& outDir, string& fileNameBone)
{
    //cout << srcImgFullPath << endl;
    fs::path fullpath(srcImgFullPath);
    
    //cout << fullpath.parent_path().string() << endl;

    // Create path in target, if not existing.
    const auto relativeSrc = fs::relative(fullpath, srcRootDir); // rel: relative
    outDir = outRootDir / relativeSrc.parent_path();
    
    //cout << "relative path: " << relativeSrc.string() << endl;
    //cout << "target parent path: " << targetParePath.string() << endl;

    if(fs::exists(outDir) == false)
    {
        cout << "Not Existed: " << outDir << endl;
        bool isOK = fs::create_directories(outDir);
        if(!isOK)
            cout << "failed to create fold: " << outDir << endl;
        else
            cout << "succeeded to create fold: " << outDir << endl;
    }
    
    string basicFileName = fullpath.filename().string();
    //cout << "Basic File Name: " << basicFileName << endl;
    
    size_t lastindex = basicFileName.find_last_of(".");
    fileNameBone = basicFileName.substr(0, lastindex);
}

// focus is saving the anno files
void ProImgInBatch(const vector<string>& jpgImgSet,
                   const fs::path& srcRootDir, const fs::path& outRootDir)
{
    for(string srcImgFullPath: jpgImgSet)
    {
        cout << "---------------***************************--------------------" << endl;
        
        fs::path outDir;
        string fileNameBone;
        PrepareDirFile(srcImgFullPath, srcRootDir, outRootDir,
                       outDir, fileNameBone);
        
        //auto outFullFilePath = targetParePath / outFileName;
        ProOneImg(srcImgFullPath, outDir, fileNameBone);
    }
}

//-------------------------------------------------------------------------------------------

