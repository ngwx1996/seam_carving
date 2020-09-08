#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/hal/interface.h>

enum eSeamDirection { VERTICAL, HORIZONTAL };

using namespace std;
using namespace cv;

#ifndef _histogram_
#define _histogram_
void getEnergyMap(Mat& h_energy, Mat& h_energyMap, int rowSize, int colSize, eSeamDirection seamDirection);
int getMinCumulativeEnergy(Mat& h_energyMap, int rowSize, int colSize, eSeamDirection seamDirection);
Mat removeSeam(Mat& h_image, vector<int> h_seam, eSeamDirection seamDirection);
#endif