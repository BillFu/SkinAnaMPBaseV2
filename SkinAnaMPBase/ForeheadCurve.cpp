//
//  ForeheadCurve.cpp

/*******************************************************************************
 
 本模块最初的功能是，将额头的轮廓线适当地抬高一些。

Author: Fu Xiaoqiang
Date:   2022/9/16

********************************************************************************/
#include <algorithm>
#include "ForeheadCurve.hpp"

// 前额顶部轮廓线由9个lm点组成。这9个点组成第0排点集，比它们低一些的9个点组成第1排点集。
// 抬高后获得的9个点组成-1排点集。
vector<int> one_row_indices{68, 104, 69, 108, 151, 337, 299, 333, 298};
vector<int> zero_row_indices{54, 103, 67, 109, 10,  338, 297, 332, 284};  // 第0排才是MP提取出的前额顶部轮廓线

// 如果点在前额顶部轮廓线上，返回它在轮廓线点集中的index；否则返回-1
int getPtIndexOfFHCurve(int ptIndex)
{
    vector<int>& vec = zero_row_indices; // 第0排才是MP提取出的前额顶部轮廓线
    
    auto it = std::find(vec.begin(), vec.end(), ptIndex);
    if (it != vec.end())
        return distance(vec.begin(), it);
    else
        return -1;
}

//-------------------------------------------------------------------------------------------

/******************************************************************************************

 *******************************************************************************************/

void RaiseupForeheadCurve(const int lm_2d[468][2], int raisedFhCurve[9][2], float alpha)
{
    // 前额顶部轮廓线由9个lm点组成。这9个点组成第0排点集，比它们低一些的9个点组成第1排点集。
    // 抬高后获得的9个点组成-1排点集。
    // 所有的lm indices以0为起始（id为0的点是“人中”穴位的最低点，V字沟的谷底）
    // 边缘的抬升效果要逐渐减弱，还要考虑x值向中央收拢一些。
    
    //int raisedPtsY[9]; // the Y values of the -1 row
    //int raisedPtsX[9]; // the X values of the -1 row
    for(int i = 0; i<9; i++)
    {
        int id_row0 = zero_row_indices[i];
        int id_row1 = one_row_indices[i];
        
        int yi_0 = lm_2d[id_row0][1];
        int xi_0 = lm_2d[id_row0][0];
        int yi_1 = lm_2d[id_row1][1];

        // the difference of y values between the 0 row and the 1 row.
        // delta_y = row1.y - row0.y; and delta_y > 0
        int delta_y = yi_1 - yi_0;
        
        // now not consider the attenuation effect when to be far away from the center
        raisedFhCurve[i][1] = int(yi_0 - delta_y* alpha);
        raisedFhCurve[i][0] = xi_0; // now x just keep unchanged
    }
}

//-------------------------------------------------------------------------------------------
