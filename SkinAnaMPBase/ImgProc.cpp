
#include "ImgProc.h"

void connectEdge(cv::Mat& src)
{
    int width = src.cols;
    int height = src.rows;
    
    //uchar * data = (uchar *)src->imageData;
    for (int i = 2; i < height - 2; i++)
    {
        uchar* pData = src.ptr<uchar>(i);
        for (int j = 2; j < width - 2; j++)
        {
            //如果该中心点为255,则考虑它的八邻域
            if (*(pData + j) == 255)
            {
                int num = 0;
                for (int k = -1; k < 2; k++)
                {
                    for (int l = -1; l < 2; l++)
                    {
                        uchar* pData1 = src.ptr<uchar>(i + k);
                        //如果八邻域中有灰度值为0的点，则去找该点的十六邻域
                        if (k != 0 && l != 0 && *(pData1 + j + l) == 255)
                            num++;
                    }
                }
                //如果八邻域中只有一个点是255，说明该中心点为端点，则考虑他的十六邻域
                if (num == 1)
                {
                    for (int k = -2; k < 3; k++)
                    {
                        for (int l = -2; l < 3; l++)
                        {
                            uchar* pData1 = src.ptr<uchar>(i + k);
                            //如果该点的十六邻域中有255的点，则该点与中心点之间的点置为255
                            if (!(k < 2 && k > -2 && l < 2 && l > -2) && *(pData1 + j + l) == 255)
                            {
                                uchar* pData2 = src.ptr<uchar>(i + k / 2);
                                *(pData2 + j + l / 2) = 255;
                                
                            }
                        }
                    }
                }
            }
        }
    }
}


void removeBurrs(const cv::Mat & src, cv::Mat &dst)
{
    //找三叉点以上,分开，按长度要不要删除
    
    int DIR[8][2] = { { 0, 1 },{ -1, 1 },{ -1, 0 },{ -1, -1 },{ 0, -1 },{ 1, -1 },{ 1, 0 },{ 1, 1 } };
    cv::Mat temp;
    cv::Mat branchPoints(src.size(), CV_8U, cv::Scalar(0));
    src.copyTo(temp);
    std::vector<int> index; //像素值非零的8领域像素索引
    int rows = src.rows;
    int cols = src.cols;
    //int count = 0;
    for (int i = 1; i < rows - 1; i++)
    {
        for (int j = 1; j < cols - 1; j++)
        {
            if (src.at<uchar>(i, j) == 0)
                continue;
            //count = 0;
            
            for (int m = 0; m < 8; m++)
            {
                if (src.at<uchar>(i + DIR[m][0], j + DIR[m][1]) != 0)
                {
                    //count++;
                    index.push_back(m);
                }
            }
            if (index.size() >= 3)//暂时只处理3分支点
            {
                //branchPoints.at<uchar>(i, j) = 0;
                if ((index[0] == 0 && index[1] == 1 && index[2] == 6) || (index[0] == 0 && index[1] == 2 && index[2] == 7)
                    || (index[0] == 1 && index[1] == 2 && index[2] == 4) || (index[0] == 0 && index[1] == 2 && index[2] == 3)
                    || (index[0] == 3 && index[1] == 4 && index[2] == 6) || (index[0] == 2 && index[1] == 4 && index[2] == 5)
                    || (index[0] == 0 && index[1] == 5 && index[2] == 6) || (index[0] == 4 && index[1] == 6 && index[2] == 7))
                {
                    index.clear();
                    continue;
                }
                if (index[2] - index[0] <= 5)
                {
                    temp.at<uchar>(i + DIR[index[2]][0], j + DIR[index[2]][1]) = 0;
                    temp.at<uchar>(i + DIR[index[0]][0], j + DIR[index[0]][1]) = 0;
                    temp.at<uchar>(i, j) = 0;
                }
                else
                {
                    if (index[1] - index[0] <= 3)
                    {
                        temp.at<uchar>(i + DIR[index[2]][0], j + DIR[index[2]][1]) = 0;
                        temp.at<uchar>(i + DIR[index[1]][0], j + DIR[index[1]][1]) = 0;
                        temp.at<uchar>(i, j) = 0;
                    }
                    else
                    {
                        temp.at<uchar>(i + DIR[index[0]][0], j + DIR[index[0]][1]) = 0;
                        temp.at<uchar>(i + DIR[index[1]][0], j + DIR[index[1]][1]) = 0;
                        temp.at<uchar>(i, j) = 0;
                    }
                }
                
            }
            
            index.clear();
        }
    }
    
    temp.copyTo(dst);
}

// 给二值图像中的粗黑线“瘦身”
int BlackLineThinInBiImg(uchar *lpBits, int Width, int Height)
{
    /////////////////////////////////////////////////////////////////
    //    lpBits: [in, out] 要细化的图像数据缓冲区
    //    Width: [in] 要细化的图像宽度
    //    Height: [in] 要细化的图像高度
    /////////////////////////////////////////////////////////////////
    uchar  erasetable[256] = {
        0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1,
        1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1,
        0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1,
        1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1,
        1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 0, 1,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1,
        1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1,
        0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1,
        1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
        1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
        1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0,
        1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0
    };
    int        x, y;
    int      num;
    bool        Finished;
    uchar       nw, n, ne, w, e, sw, s, se;
    uchar       *lpPtr = NULL;
    
    //结束标志置成假
    Finished = false;
    while (!Finished)
    { //还没有结束
        //结束标志置成假
        Finished = true;
        //先进行水平方向的细化
        for (y = 1; y < Height - 1; y++)
        { //注意为防止越界，y的范围从1到高度-2
            //lpPtr指向原图数据
            lpPtr = (uchar *)lpBits + y*Width;
            
            x = 1; //注意为防止越界，x的范围从1到宽度-2
            
            while (x < Width - 1)
            {
                if (*(lpPtr + x) == 0)
                { //是黑点才做处理
                    w = *(lpPtr + x - 1);  //左邻点
                    e = *(lpPtr + x + 1);  //右邻点
                    
                    if ((w == 255) || (e == 255))
                    {
                        //如果左右两个邻居中至少有一个是白点才处理
                        nw = *(lpPtr + x + Width - 1); //左上邻点
                        n = *(lpPtr + x + Width); //上邻点
                        ne = *(lpPtr + x + Width + 1); //右上邻点
                        sw = *(lpPtr + x - Width - 1); //左下邻点
                        s = *(lpPtr + x - Width); //下邻点
                        se = *(lpPtr + x - Width + 1); //右下邻点
                        
                        //计算索引
                        num = nw / 255 + n / 255 * 2 + ne / 255 * 4 + w / 255 * 8 + e / 255 * 16 +
                        sw / 255 * 32 + s / 255 * 64 + se / 255 * 128;
                        
                        if (erasetable[num] == 1)
                        { //经查表，可以删除
                            //在原图缓冲区中将该黑点删除
                            *(lpPtr + x) = 255;
                            Finished = false; //有改动，结束标志置成假
                            x++; //水平方向跳过一个象素
                        }
                    }
                }
                
                x++; //扫描下一个象素
            }
        }
        
        //再进行垂直方向的细化
        for (x = 1; x < Width - 1; x++)
        { //注意为防止越界，x的范围从1到宽度-2
            y = 1; //注意为防止越界，y的范围从1到高度-2
            
            while (y < Height - 1)
            {
                lpPtr = lpBits + y*Width;
                if (*(lpPtr + x) == 0)
                { //是黑点才做处理
                    n = *(lpPtr + x + Width);
                    s = *(lpPtr + x - Width);
                    
                    if ((n == 255) || (s == 255))
                    {
                        //如果上下两个邻居中至少有一个是白点才处理
                        nw = *(lpPtr + x + Width - 1);
                        ne = *(lpPtr + x + Width + 1);
                        w = *(lpPtr + x - 1);
                        e = *(lpPtr + x + 1);
                        sw = *(lpPtr + x - Width - 1);
                        se = *(lpPtr + x - Width + 1);
                        
                        //计算索引
                        num = nw / 255 + n / 255 * 2 + ne / 255 * 4 + w / 255 * 8 + e / 255 * 16 +
                        sw / 255 * 32 + s / 255 * 64 + se / 255 * 128;
                        
                        if (erasetable[num] == 1)
                        { //经查表，可以删除
                            //在原图缓冲区中将该黑点删除
                            *(lpPtr + x) = 255;
                            Finished = false; //有改动，结束标志置成假
                            y++;//垂直方向跳过一个象素
                        }
                    }
                }
                
                y++; //扫描下一个象素
            }
        }
    }
    
    return 0;
}

// integral over the sub-block of the input image from (0,0) to (i, j)
Mat Integral_2(const Mat& image)
{
    Mat result;
    Mat image2;  // every pixel has a double-precision float value
    if (image.channels() == 1)
    {
        image.convertTo(image2, CV_64FC1);
        result.create(image.size(), CV_64FC1);
        
    }
    else if (image.channels() == 3)
    {
        image.convertTo(image2, CV_64FC3);
        result.create(image.size(), CV_64FC3);
    }
    
    int c = image2.channels();
    int nr = image2.rows;
    int nc = image2.cols*c;

    for (int i = 0; i < nr; i++)
    {
        const double* inCurRow = image2.ptr<double>(i); //输入图像第i行的首址
        double* outCurRow = result.ptr<double>(i);  //结果矩阵(图像)第i行的首址
        if (i != 0)  //针对除第0行的其他行
        {
            const double* outUpRow = result.ptr<double>(i - 1); //结果矩阵当前行的上一行
            for (int j = 0; j < nc; j++)
            {
                if (j >= c) //第0列右边的其他列
                {
                    // outCurRow[j - c]: 当前像素(i, j)的左邻居
                    // outUpRow[j - c] : 当前像素(i, j)的左上邻居
                    // outUpRow[j]     : 当前像素(i, j)的上方邻居
                    // Formula: out(i, j) = in(i, j) + out(i-1, j) + out(i, j-1) - out(i-1, j-1)
                    // out(i-1, j) + out(i, j-1)，将out(i-1, j-1)统计了两次，需要减去一次
                    outCurRow[j] = inCurRow[j] + outUpRow[j] + outCurRow[j - c] - outUpRow[j - c];
                }
                else //最左边的第0列
                {
                    outCurRow[j] = inCurRow[j] + outUpRow[j];
                }
            }
        }
        else //针对第0行
        {
            for (int j = 0; j < nc; j++)
            {
                if (j >= c) //第0列右边的其他列
                {
                    outCurRow[j] = inCurRow[j] + outCurRow[j - c];
                }
                else  //最左边的第0列
                {
                    outCurRow[j] = inCurRow[j];
                }
            }
        }
    }
    return result;
}

//计算(2*d+1) * (2*d+1)领域里的像素灰度值的均值、方差。
void Local_MeanStd(const Mat& image, Mat& mean, Mat& std, int d)
{
    if (image.channels() == 1)
    {
        mean.create(image.size(), CV_64FC1);
        std.create(image.size(), CV_64FC1);
        
    }
    else if (image.channels() == 3)
    {
        mean.create(image.size(), CV_64FC3);
        std.create(image.size(), CV_64FC3);
    }
    
    Mat image_bordered;
    //复制图像，并加框。框的宽度与d有关，在实际调用d居然可以达到77。
    copyMakeBorder(image, image_bordered, d + 1, d + 1, d + 1, d + 1, BORDER_REFLECT_101);
    image_bordered.convertTo(image_bordered, mean.type());
    
    Mat image_bordered_square = image_bordered.mul(image_bordered);
    Mat Integral_image = Integral_2(image_bordered);
    Mat Integral_sq_image = Integral_2(image_bordered_square);
    
    int N = (2 * d + 1)*(2 * d + 1);
    int c = image.channels();
    int nr = image.rows;
    int nc = image.cols*c;
    
    int row_offset = 2 * d + 1; 
    int col_offset = row_offset; // col_offset with the same value
    for (int i = 0; i < nr; i++)
    {
        double* mean_value = mean.ptr<double>(i);  //第i行首址
        double* std_value = std.ptr<double>(i);
        double* integral_1order_cur = Integral_image.ptr<double>(i);   //第i行(也就是当前扫描行)首址
        double* integral_2order_cur = Integral_sq_image.ptr<double>(i); //第i行(也就是当前扫描行)首址
        double* integral_1order_down = Integral_image.ptr<double>(i + row_offset);  //当前行往下再数2*d+1的那行
        double* integral_2order_down = Integral_sq_image.ptr<double>(i + row_offset); //当前行往下再数2*d+1的那行
        for (int j = 0; j < nc; j++)
        {
            double sumi1 = integral_1order_down[j + col_offset*c] + integral_1order_cur[j] 
            							 - integral_1order_cur[j + col_offset*c] - integral_1order_down[j];
            double sumi2 = integral_2order_down[j + col_offset*c] + integral_2order_cur[j] 
                           - integral_2order_cur[j + col_offset*c] - integral_2order_down[j];
            mean_value[j] = sumi1 / N;
            std_value[j] = (sumi2 - sumi1*mean_value[j]) / N;
        }
    }
    
    cv::sqrt(std, std);
}

// ACE可以表示Automatic color equalization，或者Adaptive contrast enhancement
// 这里的ACE从实现代码来看，代表的是Adaptive Contrast Enhancement
// OpenCV图像处理专栏十四 | 基于Retinex成像原理的自动色彩均衡算法(ACE).
// https://pythontechworld.com/article/detail/TbevLKxeIEjS
// OpenCV图像处理专栏五 | ACE算法论文解读及实现
// https://cloud.tencent.com/developer/article/1552856
// 自适应对比度增强（ACE）算法原理及实现
// https://codeleading.com/article/65423852945/

void ACE(const Mat& image, Mat& result, int d, float Scale, float MaxCG)
{
    Mat localMean, localStd;
    Local_MeanStd(image, localMean, localStd, d);
    
    if (image.channels() == 1)
    {
        result.create(image.size(), CV_64FC1);
    }
    else if (image.channels() == 3)
    {
        result.create(image.size(), CV_64FC3);
    }
    
    Mat mean_m, std_m;
    meanStdDev(image, mean_m, std_m);
    double std[3];
    
    int c = image.channels();
    if (c == 1) // single channel
    {
        std[0] = std_m.at<double>(0, 0); // only this one is valid!
        std[1] = 0; // Not exist, Not Used
        std[2] = 0; // Not exist, Not Used
    }
    else  // three channels
    {
        std[0] = std_m.at<double>(0, 0);
        std[1] = std_m.at<double>(1, 0);
        std[2] = std_m.at<double>(2, 0);
    }
    
    int nr = image.rows;
    int nc = image.cols*c;
    double CG; // contrast gain
    for (int i = 0; i < nr; i++)
    {
        double* local_mean = localMean.ptr<double>(i);
        double* local_std = localStd.ptr<double>(i);
        const uchar* imageData = image.ptr<uchar>(i);
        double* outData = result.ptr<double>(i);
        for (int j = 0; j < nc; j++)
        {
            // in the case of one channel, j % c always equals 0, so only std[0] be used.
            double global_std = std[j % c];
            CG =  global_std / local_std[j];
            if (CG > MaxCG)
                CG = MaxCG;  // 防止增强得过度！
            outData[j] = local_mean[j] + Scale*CG*(int(imageData[j]) - local_mean[j]);
        }
    }
    
    result.convertTo(result, CV_8UC3);
}

void FindGradient1(cv::Mat& InputImage, cv::Mat& OutputImage)
{
    cv::Mat Knernel_Y = (cv::Mat_<float>(3, 3) << -1, -1, -1, 0, 0, 0, 1, 1, 1);
    cv::Mat Knernel_X = (cv::Mat_<float>(3, 3) << -1, 0, 1, -1, 0, 1, -1, 0, 1);
    
    cv::Mat Knernel_YY = (cv::Mat_<float>(3, 3) << 1, 1, 1, 0, 0, 0, -1, -1, -1);
    cv::Mat Knernel_XX = (cv::Mat_<float>(3, 3) << 1, 0, -1, 1, 0, -1, 1, 0, -1);
    
    cv::Mat Image_y;
    cv::Mat Image_x;
    
    filter2D(InputImage, Image_y, -1, Knernel_Y, cv::Point(-1, -1), 0, cv::BORDER_DEFAULT);
    filter2D(InputImage, Image_x, -1, Knernel_X, cv::Point(-1, -1), 0, cv::BORDER_DEFAULT);
    
    cv::Mat Image_yy;
    cv::Mat Image_xx;
    
    filter2D(InputImage, Image_yy, -1, Knernel_YY, cv::Point(-1, -1), 0, cv::BORDER_DEFAULT);
    filter2D(InputImage, Image_xx, -1, Knernel_XX, cv::Point(-1, -1), 0, cv::BORDER_DEFAULT);
    
    float alpha = 1.0;
    cv::Mat I_edge = abs(Image_x)*alpha + abs(Image_y)*(1 - 0);
    cv::Mat I_edge1 = abs(Image_xx)*alpha + abs(Image_yy)*(1 - 0);
    Mat I_edge2 = I_edge + I_edge1;
    I_edge2.copyTo(OutputImage);
}

void FindGradient(cv::Mat& InputImage, cv::Mat& OutputImage)
{
    float p = 1;//15
    float q = 0;//0
    
    cv::Mat Knernel_Y = (cv::Mat_<float>(3, 3) << -1, -p, -1, 0, q, 0, 1, p, 1);
    cv::Mat Knernel_X = (cv::Mat_<float>(3, 3) << -1, 0, 1, -p, q, p, -1, 0, 1);
    
    cv::Mat Image_y;
    cv::Mat Image_x;
    
    filter2D(InputImage, Image_y, -1, Knernel_Y, cv::Point(-1, -1), 0, cv::BORDER_DEFAULT);
    filter2D(InputImage, Image_x, -1, Knernel_X, cv::Point(-1, -1), 0, cv::BORDER_DEFAULT);
    
    float alpha = 0.5;
    cv::Mat I_edge = abs(Image_x)*alpha + abs(Image_y)*(1 - alpha);
    
    I_edge.copyTo(OutputImage);
}
/*
cv::Mat worldGray(const cv::Mat& src)
{
    Mat dstImage;
    std::vector<Mat> g_vChannels;
    
    //分离通道
    split(src, g_vChannels);
    Mat imageBlueChannel = g_vChannels.at(0);
    Mat imageGreenChannel = g_vChannels.at(1);
    Mat imageRedChannel = g_vChannels.at(2);
    
    double imageBlueChannelAvg = 0;
    double imageGreenChannelAvg = 0;
    double imageRedChannelAvg = 0;
    
    //求各通道的平均值
    imageBlueChannelAvg = mean(imageBlueChannel)[0];
    imageGreenChannelAvg = mean(imageGreenChannel)[0];
    imageRedChannelAvg = mean(imageRedChannel)[0];
    
    //求出个通道所占增益
    double K = (imageBlueChannelAvg + imageGreenChannelAvg + imageRedChannelAvg) / 3;
    double Kb = K / imageBlueChannelAvg;
    double Kg = K / imageGreenChannelAvg;
    double Kr = K / imageRedChannelAvg;
    
    imageBlueChannel = Kb*imageBlueChannel;
    imageGreenChannel = Kg*imageGreenChannel;
    imageRedChannel = Kr*imageRedChannel;
    
    merge(g_vChannels, dstImage);//图像各通道合并
    g_vChannels.clear();
    return dstImage;
}
*/
cv::Mat worldGray(const cv::Mat& src)
{
    std::vector<Mat> g_vChannels;
    split(src, g_vChannels);
    
    double chanAvg[3];
    double sum_avg = 0.0;
    for(int c = 0; c<3; c++)
    {
        chanAvg[c] = mean(g_vChannels.at(c))[0];
        sum_avg += chanAvg[c];
    }
    
    double K = sum_avg / 3; // K实际就是全部像素、全部通道所有灰度值的均值

    double k_Chans[3];
    for(int c = 0; c<3; c++)
    {
        k_Chans[c] = K / chanAvg[c];
        g_vChannels[c] *= k_Chans[c];
    }
    
    Mat dstImage;
    merge(g_vChannels, dstImage);//图像各通道合并
    g_vChannels.clear();
    return dstImage;
}
