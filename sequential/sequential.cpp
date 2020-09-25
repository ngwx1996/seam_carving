#include "opencv2\opencv.hpp"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/hal/interface.h>
#include <iostream>
#include <string>
#include <vector>
#include <time.h>
using namespace std;
using namespace cv;

enum eSeamDirection { VERTICAL, HORIZONTAL };

float sobelEnergyTime = 0;
float cumEnergyTime = 0;
float findSeamTime = 0;
float removeSeamTime = 0;

Mat createEnergyImg(Mat &image) {
	clock_t start = clock();
	Mat grayscale, grad, energy;
	Mat grad_x, grad_y;
	Mat abs_grad_x, abs_grad_y;
	int ddepth = CV_16S;
	int scale = 1;
	int delta = 0;

	// Convert image to grayscale
	cvtColor(image, grayscale, COLOR_BGR2GRAY);

	// Perform sobel operator to get image gradient
	Sobel(grayscale, grad_x, ddepth, 1, 0, 3, scale, delta, BORDER_DEFAULT);
	Sobel(grayscale, grad_y, ddepth, 0, 1, 3, scale, delta, BORDER_DEFAULT);

	convertScaleAbs(grad_x, abs_grad_x);
	convertScaleAbs(grad_y, abs_grad_y);

	addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad);

	// Convert gradient to double
	grad.convertTo(energy, CV_64F, 1.0 / 255.0);

	clock_t end = clock();
	sobelEnergyTime += ((float)end - (float)start) / CLOCKS_PER_SEC;

	return energy;
}

Mat createEnergyMap(Mat& energy, eSeamDirection seamDirection) {
	clock_t start = clock();
	double topCenter, topLeft, topRight;
	int rowSize = energy.rows;
	int colSize = energy.cols;

	// Initialize energy map
	Mat energyMap = Mat(rowSize, colSize, CV_64F, double(0));

	if (seamDirection == VERTICAL) {
		// Vertical Seam
		// The first row of the map should be the same as the first row of energy
		energy.row(0).copyTo(energyMap.row(0));

		for (int row = 1; row < rowSize; row++) {
			for (int col = 0; col < colSize; col++) {
				topCenter = energyMap.at<double>(row - 1, col);
				topLeft = energyMap.at<double>(row - 1, max(col - 1, 0));
				topRight = energyMap.at<double>(row - 1, min(col + 1, colSize - 1));

				// add energy at pixel with smallest of previous row neighbor's cumulative energy
				energyMap.at<double>(row, col) = energy.at<double>(row, col) + min(topCenter, min(topLeft, topRight));
			}
		}
	}
	else {
		// Horizontal Seam
		// The first col of the map should be the same as the first col of energy
		energy.col(0).copyTo(energyMap.col(0));

		for (int col = 1; col < colSize; col++) {
			for (int row = 0; row < rowSize; row++) {	
				topCenter = energyMap.at<double>(row, col - 1);
				topLeft = energyMap.at<double>(max(row - 1, 0), col - 1);
				topRight = energyMap.at<double>(min(row + 1, rowSize - 1), col - 1);

				// add energy at pixel with smallest of previous col neighbor's cumulative energy
				energyMap.at<double>(row, col) = energy.at<double>(row, col) + min(topCenter, min(topLeft, topRight));
			}
		}
	}
	clock_t end = clock();
	cumEnergyTime += ((float)end - (float)start) / CLOCKS_PER_SEC;

	return energyMap;
}

vector<int> findSeam(Mat& energyMap, eSeamDirection seamDirection) {
	clock_t start = clock();
	int rowSize = energyMap.rows;
	int colSize = energyMap.cols;
	vector<int> seam;
	double topCenter, topLeft, topRight;
	double minVal;
	Point minLoc;
	
	if (seamDirection == VERTICAL) {
		// Vertical seam, reduces width
		seam.resize(rowSize);
		// Get location of min cumulative energy
		minMaxLoc(energyMap.row(rowSize - 1), &minVal, NULL, &minLoc, NULL);
		int curCol = minLoc.x;
		seam[rowSize - 1] = curCol;

		// Look at top neighbors to find next minimum cumulative energy
		for (int row = rowSize - 1; row > 0; row--) {
			topCenter = energyMap.at<double>(row - 1, curCol);
			topLeft = energyMap.at<double>(row - 1, max(curCol - 1, 0));
			topRight = energyMap.at<double>(row - 1, min(curCol + 1, colSize - 1));

			// find next col idx
			if (min(topLeft, topCenter) > topRight) {
				// topRight smallest
				curCol += 1;
			}
			else if (min(topRight, topCenter) > topLeft) {
				// topLeft smallest
				curCol -= 1;
			}
			// if topCenter smallest, curCol remain;
			// update seam
			seam[row - 1] = curCol;
		}
	}
	else {
		// Horizontal seam, reduces height
		seam.resize(colSize);
		// Get location of min cumulative energy
		minMaxLoc(energyMap.col(colSize - 1), &minVal, NULL, &minLoc, NULL);
		int curCol = minLoc.y;
		seam[colSize - 1] = curCol;

		// Look at top neighbors to find next minimum cumulative energy
		for (int col = colSize - 1; col > 0; col--) {
			topCenter = energyMap.at<double>(curCol, col - 1);
			topLeft = energyMap.at<double>(max(curCol - 1, 0), col - 1);
			topRight = energyMap.at<double>(min(curCol + 1, rowSize - 1), col - 1);

			// find next col idx
			if (min(topLeft, topCenter) > topRight) {
				// topRight smallest
				curCol += 1;
			}
			else if (min(topRight, topCenter) > topLeft) {
				// topLeft smallest
				curCol -= 1;
			}
			// if topCenter smallest, curCol remain;
			// update seam
			seam[col - 1] = curCol;
		}
	}
	clock_t end = clock();
	findSeamTime += ((float)end - (float)start) / CLOCKS_PER_SEC;
	return seam;
}

void removeSeam(Mat& image, vector<int> seam, eSeamDirection seamDirection) {
	clock_t start = clock();
	// spare 1x1x3 to maintain matrix size
	Mat spare(1, 1, CV_8UC3, Vec3b(0, 0, 0));

	if (seamDirection == VERTICAL) {
		// Vertical seam, reduces width
		Mat tempRow(image.cols, 1, CV_8UC3);
		for (int i = 0; i < image.rows; i++) {
			tempRow.setTo(0);
			Mat beforeIdx = image.rowRange(i, i + 1).colRange(0, seam[i]);
			Mat afterIdx = image.rowRange(i, i + 1).colRange(seam[i] + 1, image.cols);

			if (beforeIdx.empty()) {
				hconcat(afterIdx, spare, tempRow);
			}
			else if (afterIdx.empty()) {
				hconcat(beforeIdx, spare, tempRow);
			}
			else {
				hconcat(beforeIdx, afterIdx, tempRow);
				hconcat(tempRow, spare, tempRow);
			}
			tempRow.copyTo(image.row(i));
		}
		image = image.colRange(0, image.cols - 1);
	}
	else {
		// Horizontal seam, reduces height
		Mat tempCol(1, image.rows, CV_8UC3);
		for (int i = 0; i < image.cols; i++) {
			tempCol.setTo(0);
			Mat beforeIdx = image.colRange(i, i + 1).rowRange(0, seam[i]);
			Mat afterIdx = image.colRange(i, i + 1).rowRange(seam[i] + 1, image.rows);

			if (beforeIdx.empty()) {
				vconcat(afterIdx, spare, tempCol);
			}
			else if (afterIdx.empty()) {
				vconcat(beforeIdx, spare, tempCol);
			}
			else {
				vconcat(beforeIdx, afterIdx, tempCol);
				vconcat(tempCol, spare, tempCol);
			}
			tempCol.copyTo(image.col(i));
		}
		image = image.rowRange(0, image.rows - 1);
	}
	
	//imshow("after cut", image);
	clock_t end = clock();
	removeSeamTime += ((float)end - (float)start) / CLOCKS_PER_SEC;
	return;
}

int main(int argc, char* argv[]) {
	clock_t start = clock();
	// Set how much to reduce width or/and height by and set image.
	int reduceWidth = 1000;
	int reduceHeight = 200;
	string imageName = "F:/CUDA/seam_carving/images/inputColdplayWings.jpg";
	Mat image = imread(imageName, IMREAD_COLOR);
	if (image.empty()) {
		cout << "Invalid image. Please try again" << endl;
		waitKey(0);
		return 1;
	}
	pair<int, int> imageSize = { image.cols, image.rows };

	imshow("Original", image);

	// Vertical seam, reduces width
	for (int i = 0; i < reduceWidth; i++) {
		Mat energy = createEnergyImg(image);
		Mat energyMap = createEnergyMap(energy, VERTICAL);
		vector<int> seam = findSeam(energyMap, VERTICAL);
		removeSeam(image, seam, VERTICAL);
	}

	// Horizontal seam, reduces height
	for (int j = 0; j < reduceHeight; j++) {
		Mat energy = createEnergyImg(image);
		Mat energyMap = createEnergyMap(energy, HORIZONTAL);
		vector<int> seam = findSeam(energyMap, HORIZONTAL);
		removeSeam(image, seam, HORIZONTAL);
	}

	imshow("Result", image);
	clock_t end = clock();
	float totalTime = ((float)end - (float)start) / CLOCKS_PER_SEC;
	cout << "Sequential on CPU" << endl;
	cout << "Image name: " << imageName << endl;
	cout << "Input dimension " << imageSize.first << " x " << imageSize.second << endl;
	cout << "Output dimension " << image.cols << " x " << image.rows << endl;
	cout << "Cumulative time taken in each function for all iterations" << "s" << endl;
	cout << "Time taken to get energy of each image: " << sobelEnergyTime << "s" << endl;
	cout << "Time taken to get cumulative energy map: " << cumEnergyTime << "s" << endl;
	cout << "Time taken to find seam: " << findSeamTime << "s" << endl;
	cout << "Time taken to remove seam: " << removeSeamTime << "s" << endl;
	cout << "Total time taken: " << totalTime << "s" << endl;

	imwrite("F:/CUDA/seam_carving/images/outputColdplayWings.jpg", image);
	waitKey(0);
	return 0;
}