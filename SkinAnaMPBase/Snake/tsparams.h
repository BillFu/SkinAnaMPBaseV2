#ifndef VISIONPARAMETERS_H
#define VISIONPARAMETERS_H

// Only used to pull #defines - Needed for all algos
#include "sobel.h"


/*      ALGO SWITCH

    To switch between contour algorithms, comment out
    one of the following defines, and uncomment the other.
    The build will fail if neither are defined as
    the class defined in this header (AlgoParams) is required
    by the GUI to build the parameters windows as-well-as
    the video player and display widget for passing
    algorithm parameters from the user to the algs.
*/

// --------------------------------------

/* Traditiaonal-ish snake algorithm */

/* END traditional-ish snake algorithm */

// --------------------------------------

/* Custom snake algorithm */

 //   #define CUSTOM_CONTOUR_ALG 0

/*  END custom snake algorithm*/

// --------------------------------------

/*  END ALGO SWITCH */

// Optionals
#define DRAW_LINE           1
#define DRAW_POINTS         1
#define DRAW_NEIGHBORS

// Required
#define CONTOUR_ALPHA       0.5   //越小，轮廓线越短，越紧致
#define CONTOUR_BETA        0.3   //越大，才能放大曲率的差异，曲率越小越光滑
#define CONTOUR_GAMMA       1.0   //越大，越贴近边缘
#define CONTOUR_LAMBDA      2.2   //越大，越靠近角点

#define MINIMUM_POINTS      4
#define CONTOUR_START_POINT 200

/* Active Contour */

class AlgoParams
{
public:
    AlgoParams()
    {
        this->_sobelThresh = SOBEL_THRESH;
        this->_viewSobel = false;
        this->_nPoints = CONTOUR_START_POINT;
        this->_alpha = CONTOUR_ALPHA;
        this->_beta = CONTOUR_BETA;
        this->_gamma = CONTOUR_GAMMA;
        this->_lambda = CONTOUR_LAMBDA;
    }

    ~AlgoParams(){}
    void setNumberPoints(int n){ this->_nPoints = n; }
    void setSobelThresh(int t){ this->_sobelThresh = t; }
    void setCalcSobelAngle(bool state){ this->_sobelAngle = state; }
    void setDrawSobelDeadpace(bool state){ this->_sobelDeadSpace = state; }
    void setDrawSnakePoints(bool state){ this->_drawSnakePoints = state; }
    void setDrawSnakeLines(bool state){ this->_drawSnakeLines = state; }
    void setViewSobel(bool state){ this->_viewSobel = state; }
    void setAlpha(float val){ this->_alpha = val; }
    void setBeta(float val){ this->_beta = val; }
    void setGama(float val){ this->_gamma = val; }
    int getNumberPoints(){ return this->_nPoints; }
    int getSobelThreash(){ return this->_sobelThresh; }
    bool getSobelAngle(){ return this->_sobelAngle; }
    bool getSobelDeadSpace(){ return this->_sobelDeadSpace; }
    bool getDrawSnakeLines(){ return this->_drawSnakeLines; }
    bool getDrawSnakePoints(){ return this->_drawSnakePoints; }
    bool getViewSobel(){ return this->_viewSobel; }
    float getAlpha(){ return _alpha; }
    float getBeta(){ return _beta; }
    float getGama(){ return _gamma; }

//private:
public:
    int _nPoints;
    int _sobelThresh;
    bool _sobelAngle;
    bool _sobelDeadSpace;
    bool _drawSnakeLines;
    bool _drawSnakePoints;
    bool _viewSobel;
    float _alpha;
    float _beta;
    float _gamma;
    float _lambda;
};


#endif // VISIONPARAMETERS_H

