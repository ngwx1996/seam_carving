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
#include <chrono>
#include "parallel.h"

using namespace std;
using namespace cv;

float sobelEnergyTime = 0;
float cumEnergyTime = 0;
float findSeamTime = 0;
float removeSeamTime = 0;

Mat createEnergyImg(Mat& image) {
	auto start = chrono::high_resolution_clock::now();
	Mat grad, energy;
	cuda::GpuMat gpuImage(image.rows, image.cols, image.type());
	cuda::GpuMat grayscale;
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

	// Convert gradient to float
	grad.convertTo(energy, CV_32F, 1.0 / 255.0);
	
	auto end = chrono::high_resolution_clock::now();

	sobelEnergyTime += chrono::duration_cast<chrono::milliseconds>(end - start).count();

	return energy;
}

Mat createEnergyMap(Mat& energy, eSeamDirection seamDirection) {
	auto start = chrono::high_resolution_clock::now();
	int rowSize = energy.rows;
	int colSize = energy.cols;
	// Initialize energy map
	Mat energyMap = Mat(rowSize, colSize, CV_32F, float(0));

	// Call cuda function to get energy map
	getEnergyMap(energy, energyMap, rowSize, colSize, seamDirection);

	auto end = chrono::high_resolution_clock::now();
	cumEnergyTime += chrono::duration_cast<chrono::milliseconds>(end - start).count();
	return energyMap;
}

vector<int> findSeam(Mat& energyMap, eSeamDirection seamDirection) {
	auto start = chrono::high_resolution_clock::now();
	int rowSize = energyMap.rows;
	int colSize = energyMap.cols;
	int curLoc;
	vector<int> seam;
	float topCenter, topLeft, topRight;
	double minVal;
	Point minLoc;

	
	if (seamDirection == VERTICAL) {
		// Call kernel for parallel reduction to find min cumulative energy
		seam.resize(rowSize);

		curLoc = getMinCumulativeEnergy(energyMap, rowSize, colSize, VERTICAL);
		//cuda::minMaxLoc(energyMap.row(rowSize - 1), &minVal, NULL, &minLoc, NULL);
		//curLoc = minLoc.x;

		seam[rowSize - 1] = curLoc;

		// Look at top neighbors to find next minimum cumulative energy
		for (int row = rowSize - 1; row > 0; row--) {
			topCenter = energyMap.at<float>(row - 1, curLoc);
			topLeft = energyMap.at<float>(row - 1, max(curLoc - 1, 0));
			topRight = energyMap.at<float>(row - 1, min(curLoc + 1, colSize - 1));

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
		//cuda::minMaxLoc(energyMap.col(colSize - 1), &minVal, NULL, &minLoc, NULL);
		//curLoc = minLoc.y;
		
		seam[colSize - 1] = curLoc;

		// Look at top neighbors to find next minimum cumulative energy
		for (int col = colSize - 1; col > 0; col--) {
			topCenter = energyMap.at<float>(curLoc, col - 1);
			topLeft = energyMap.at<float>(max(curLoc - 1, 0), col - 1);
			topRight = energyMap.at<float>(min(curLoc + 1, rowSize - 1), col - 1);

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
	
	auto end = chrono::high_resolution_clock::now();
	findSeamTime += chrono::duration_cast<chrono::milliseconds>(end - start).count();
	return seam;
}

int main(int argc, char* argv[]) {
	for(int k = 0; k < 5; k++)
		warmUpGPU();
	auto start = chrono::high_resolution_clock::now();
	// Set how much to reduce width or/and height by and set image.
	int reduceWidth = 1000;
	int reduceHeight = 200;
	string imageName = "F:/CUDA/seam_carving/images/inputColdplayWings.jpg";
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
		auto startRemove = chrono::high_resolution_clock::now();
		removeSeam(image, seam, VERTICAL);
		auto endRemove = chrono::high_resolution_clock::now();
		removeSeamTime += chrono::duration_cast<chrono::milliseconds>(endRemove - startRemove).count();
	}

	// Horizontal seam, reduces height
	for (int j = 0; j < reduceHeight; j++) {
		Mat energy = createEnergyImg(image);
		Mat energyMap = createEnergyMap(energy, HORIZONTAL);
		vector<int> seam = findSeam(energyMap, HORIZONTAL);
		auto startRemove = chrono::high_resolution_clock::now();
		removeSeam(image, seam, HORIZONTAL);
		auto endRemove = chrono::high_resolution_clock::now();
		removeSeamTime += chrono::duration_cast<chrono::milliseconds>(endRemove - startRemove).count();
	}

	imshow("Result", image);

	auto end = chrono::high_resolution_clock::now();
	float totalTime = chrono::duration_cast<chrono::milliseconds>(end - start).count();
	cout << "Parallel with CUDA" << endl;
	cout << "Image name: " << imageName << endl;
	cout << "Input dimension " << imageSize.first << " x " << imageSize.second << endl;
	cout << "Output dimension " << image.cols << " x " << image.rows << endl;
	cout << "Cumulative time taken in each function for all iterations" << "ms" << endl;
	cout << "Time taken to get energy of each image: " << sobelEnergyTime << "ms" << endl;
	cout << "Time taken to get cumulative energy map: " << cumEnergyTime << "ms" << endl;
	cout << "Time taken to find seam: " << findSeamTime << "ms" << endl;
	cout << "Time taken to remove seam: " << removeSeamTime << "ms" << endl;
	cout << "Total time taken: " << totalTime << "ms" << endl;

	//imwrite("F:/CUDA/seam_carving/images/outputColdplayWings.jpg", image);
	
	waitKey();
	return 0;
}