#include "opencv2\opencv.hpp"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/hal/interface.h>
#include <opencv2/cudafilters.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudaarithm.hpp>
#include <iostream>
#include <string>
#include <vector>
#include <time.h>
#include "parallel.h"

using namespace std;
using namespace cv;

float sobelEnergyTime = 0;
float cumEnergyTime = 0;
float findSeamTime = 0;
float removeSeamTime = 0;

Mat createEnergyImg(Mat& image) {
	clock_t start = clock();
	Mat grad, energy;
	cuda::GpuMat gpuImage, grayscale;
	cuda::GpuMat grad_x, grad_y;
	cuda::GpuMat abs_grad_x, abs_grad_y;
	int ddepth = CV_16S;
	int scale = 1;
	int delta = 0;

	gpuImage.upload(image);
	// Convert image to grayscale
	cuda::cvtColor(gpuImage, grayscale, COLOR_BGR2GRAY);

	// Perform sobel operator to get image gradient using cuda
	Ptr<cuda::Filter> filter;
	filter = cuda::createSobelFilter(grayscale.type(), ddepth, 1, 0, 3);
	filter->apply(grayscale, grad_x);
	filter = cuda::createSobelFilter(grayscale.type(), ddepth, 0, 1, 3);
	filter->apply(grayscale, grad_y);

	cuda::abs(grad_x, abs_grad_x);
	cuda::abs(grad_y, abs_grad_y);

	cuda::addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad);

	// Convert gradient to double
	grad.convertTo(energy, CV_64F, 1.0 / 255.0);

	clock_t end = clock();
	sobelEnergyTime += ((float)end - (float)start) / CLOCKS_PER_SEC;

	return energy;
}

Mat createEnergyMap(Mat& energy, eSeamDirection seamDirection) {
	clock_t start = clock();
	int rowSize = energy.rows;
	int colSize = energy.cols;
	// Initialize energy map
	Mat energyMap = Mat(rowSize, colSize, CV_64F, double(0));

	// Call cuda function to get energy map
	getEnergyMap(energy, energyMap, rowSize, colSize, seamDirection);

	clock_t end = clock();
	cumEnergyTime += ((float)end - (float)start) / CLOCKS_PER_SEC;

	return energyMap;
}

vector<int> findSeam(Mat& energyMap, eSeamDirection seamDirection) {
	clock_t start = clock();
	int rowSize = energyMap.rows;
	int colSize = energyMap.cols;
	int curLoc;
	vector<int> seam;
	double topCenter, topLeft, topRight;
	
	if (seamDirection == VERTICAL) {
		// Call kernel for parallel reduction to find min cumulative energy
		seam.resize(rowSize);
		curLoc = getMinCumulativeEnergy(energyMap, rowSize, colSize, VERTICAL);
		seam[rowSize - 1] = curLoc;

		// Look at top neighbors to find next minimum cumulative energy
		for (int row = rowSize - 1; row > 0; row--) {
			topCenter = energyMap.at<double>(row - 1, curLoc);
			topLeft = energyMap.at<double>(row - 1, max(curLoc - 1, 0));
			topRight = energyMap.at<double>(row - 1, min(curLoc + 1, colSize - 1));

			// find next col idx
			if (min(topLeft, topCenter) > topRight) {
				// topRight smallest
				curLoc += 1;
			}
			else if (min(topRight, topCenter) > topLeft) {
				// topLeft smallest
				curLoc -= 1;
			}
			// if topCenter smallest, curCol remain;
			// update seam
			seam[row - 1] = curLoc;
		}
	}
	else {
		// Horizontal seam, reduces height
		// Call kernel for parallel reduction to find min cumulative energy
		seam.resize(colSize);
		curLoc = getMinCumulativeEnergy(energyMap, rowSize, colSize, HORIZONTAL);
		seam[colSize - 1] = curLoc;

		// Look at top neighbors to find next minimum cumulative energy
		for (int col = colSize - 1; col > 0; col--) {
			topCenter = energyMap.at<double>(curLoc, col - 1);
			topLeft = energyMap.at<double>(max(curLoc - 1, 0), col - 1);
			topRight = energyMap.at<double>(min(curLoc + 1, rowSize - 1), col - 1);

			// find next col idx
			if (min(topLeft, topCenter) > topRight) {
				// topRight smallest
				curLoc += 1;
			}
			else if (min(topRight, topCenter) > topLeft) {
				// topLeft smallest
				curLoc -= 1;
			}
			// if topCenter smallest, curCol remain;
			// update seam
			seam[col - 1] = curLoc;
		}
	}
	
	clock_t end = clock();
	findSeamTime += ((float)end - (float)start) / CLOCKS_PER_SEC;
	return seam;
}

int main(int argc, char* argv[]) {
	clock_t start = clock();
	// Set how much to reduce width or/and height by and set image.
	int reduceWidth = 100;
	int reduceHeight = 50;
	string imageName = "../images/inputPrague.jpg";
	Mat image = imread(imageName, IMREAD_COLOR);
	if (image.empty()) {
		cout << "Invalid image. Please try again" << endl;
		return 1;
	}
	pair<int, int> imageSize = { image.cols, image.rows };
	imshow("Original", image);

	// Vertical seam, reduces width
	for (int i = 0; i < reduceWidth; i++) {
		Mat energy = createEnergyImg(image);
		Mat energyMap = createEnergyMap(energy, VERTICAL);
		vector<int> seam = findSeam(energyMap, VERTICAL);
		clock_t startRemove = clock();
		image = removeSeam(image, seam, VERTICAL);
		clock_t endRemove = clock();
		removeSeamTime += ((float)endRemove - (float)startRemove) / CLOCKS_PER_SEC;
	}

	// Horizontal seam, reduces height
	for (int j = 0; j < reduceHeight; j++) {
		Mat energy = createEnergyImg(image);
		Mat energyMap = createEnergyMap(energy, HORIZONTAL);
		vector<int> seam = findSeam(energyMap, HORIZONTAL);
		clock_t startRemove = clock();
		image = removeSeam(image, seam, HORIZONTAL);
		clock_t endRemove = clock();
		removeSeamTime += ((float)endRemove - (float)startRemove) / CLOCKS_PER_SEC;
	}

	imshow("Result", image);

	clock_t end = clock();
	float totalTime = ((float)end - (float)start) / CLOCKS_PER_SEC;
	cout << "Parallel with CUDA" << endl;
	cout << "Image name: " << imageName << endl;
	cout << "Input dimension " << imageSize.first << " x " << imageSize.second << endl;
	cout << "Output dimension " << image.cols << " x " << image.rows << endl;
	cout << "Cumulative time taken in each function for all iterations" << "s" << endl;
	cout << "Time taken to get energy of each image: " << sobelEnergyTime << "s" << endl;
	cout << "Time taken to get cumulative energy map: " << cumEnergyTime << "s" << endl;
	cout << "Time taken to find seam: " << findSeamTime << "s" << endl;
	cout << "Time taken to remove seam: " << removeSeamTime << "s" << endl;
	cout << "Total time taken: " << totalTime << "s" << endl;

	imwrite("../images/outputPrague.jpg", image);
	waitKey();
	return 0;
}