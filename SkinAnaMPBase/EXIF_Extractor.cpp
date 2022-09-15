//
//  EXIF_Extractor.cpp

/*******************************************************************************
Author: Fu Xiaoqiang
Date:   2022/9/15

********************************************************************************/


#include <iostream> // std::cout
#include <fstream>  // std::ifstream

#include "EXIF_Extractor.hpp"
#include "EXIF/TinyEXIF.h"

using namespace std;

/**********************************************************************************************

***********************************************************************************************/
void ExtractEXIF(const char* srcImgFileName)
{
    // open a stream to read just the necessary parts of the image file
    ifstream istream(srcImgFileName, ifstream::binary);
    
    // parse image EXIF and XMP metadata
    TinyEXIF::EXIFInfo imageEXIF(istream);
    if (imageEXIF.Fields)
    {
        cout << "Image Description: " << imageEXIF.ImageDescription << endl;
        /*
        << "Image Resolution " << imageEXIF.ImageWidth << "x" << imageEXIF.ImageHeight << " pixels\n"
        << "Camera Model " << imageEXIF.Make << " - " << imageEXIF.Model << "\n"
        << "Focal Length " << imageEXIF.FocalLength << " mm" << std::endl;
         */
        
        cout << "Focal Length: " << imageEXIF.FocalLength << " mm" << endl;

    }
}
//-------------------------------------------------------------------------------------------
