#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/hal/interface.h>
#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <float.h>
#include "parallel.h"

#define MAX_THREADS 1024
using namespace std;
using namespace cv;

__global__ void cudaEnergyMap(unsigned char* energy, unsigned char* energyMap, unsigned char* prevEnergy, int rowSize, int colSize, int current, eSeamDirection seamDirection);
__global__ void cudaReduction(unsigned char* row, double* mins, int* minsIndices, int size, int blockSize, int next);
__global__ void cudaRemoveSeam(unsigned char* image, unsigned char* output, int* seam, 
	unsigned char* dummy, int rowSize, int colSize, int imageStep, int outputStep, eSeamDirection seamDirection);

int nextPowerof2(int n) {
	n--;
	n |= n >> 1;
	n |= n >> 2;
	n |= n >> 4;
	n |= n >> 8;
	n |= n >> 16;
	n++;
	return n;
}

void getEnergyMap(Mat& h_energy, Mat& h_energyMap, int rowSize, int colSize, eSeamDirection seamDirection) {
	Mat h_prevEnergy;
	unsigned char* d_energy;
	unsigned char* d_energyMap;
	unsigned char* d_prevEnergy;
	int size = rowSize * colSize;

	cudaMalloc(&d_energy, size * sizeof(double));
	cudaMalloc(&d_energyMap, size * sizeof(double));
	cudaMemcpy(d_energy, h_energy.ptr(), size * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_energyMap, h_energyMap.ptr(), size * sizeof(double), cudaMemcpyHostToDevice);

	if (seamDirection == VERTICAL) {
		// Start from first row. Copy first row of energyMap to be used in device
		h_prevEnergy = Mat(1, colSize, CV_64F, double(0));
		h_energy.row(0).copyTo(h_prevEnergy.row(0));

		cudaMalloc(&d_prevEnergy, colSize * sizeof(double));
		cudaMemcpy(d_prevEnergy, h_prevEnergy.ptr(), colSize * sizeof(double), cudaMemcpyHostToDevice);

		int blockSize = min(colSize, MAX_THREADS);
		int gridSize = ((colSize - 1) / MAX_THREADS) + 1;
		dim3 blockDim(blockSize, 1, 1);
		dim3 gridDim(gridSize, 1, 1);

		// For each row, call kernel to get cumEnergy for the row
		for (int i = 1; i < rowSize; i++) {
			cudaEnergyMap << <gridDim, blockDim >> > (d_energy, d_energyMap, d_prevEnergy, rowSize, colSize, i, VERTICAL);
		}
	}
	else {
		// Horizontal Seam, reduces height
		// Start from first row. Copy first row of energyMap to be used in device
		h_prevEnergy = Mat(rowSize, 1, CV_64F, double(0));
		h_energy.col(0).copyTo(h_prevEnergy.col(0));

		cudaMalloc(&d_prevEnergy, rowSize * sizeof(double));
		cudaMemcpy(d_prevEnergy, h_prevEnergy.ptr(), rowSize * sizeof(double), cudaMemcpyHostToDevice);

		int blockSize = min(rowSize, MAX_THREADS);
		int gridSize = ((rowSize - 1) / MAX_THREADS) + 1;
		dim3 blockDim(1, blockSize, 1);
		dim3 gridDim(1, gridSize, 1);

		// For each col, call kernel to get cumEnergy for the col
		for (int i = 1; i < colSize; i++) {
			cudaEnergyMap << <gridDim, blockDim >> > (d_energy, d_energyMap, d_prevEnergy, rowSize, colSize, i, HORIZONTAL);
		}
	}

	cudaMemcpy(h_energyMap.ptr(), d_energyMap, size * sizeof(double), cudaMemcpyDeviceToHost);
	cudaFree(d_energy);
	cudaFree(d_energyMap);
	cudaFree(d_prevEnergy);
}

int getMinCumulativeEnergy(Mat& h_energyMap, int rowSize, int colSize, eSeamDirection seamDirection) {
	// Require block size to be a multiple of 2 for parallel reduction
	// Sequential addressing ensures bank conflict free
	int blockSize;
	int gridSize;
	int lastSize;
	int sharedSize;
	Mat h_last;
	if (seamDirection == VERTICAL) {
		blockSize = min(nextPowerof2(colSize / 2), MAX_THREADS);
		gridSize = (nextPowerof2(colSize / 2) - 1) / MAX_THREADS + 1;
		sharedSize = blockSize * 2 * (sizeof(double) + sizeof(int));
		lastSize = colSize;
		// Copy last row of energyMap to be used in device
		h_last = Mat(1, colSize, CV_64F, double(0));
		h_energyMap.row(rowSize - 1).copyTo(h_last.row(0));
	}
	else {
		// Horizontal Seam, reduces height
		blockSize = min(nextPowerof2(rowSize / 2), MAX_THREADS);
		gridSize = (nextPowerof2(rowSize / 2) - 1) / MAX_THREADS + 1;
		sharedSize = blockSize * 2 * (sizeof(double) + sizeof(int));
		lastSize = rowSize;
		// Copy last row of energyMap to be used in device
		h_last = Mat(rowSize, 1, CV_64F, double(0));
		h_energyMap.col(colSize - 1).copyTo(h_last.col(0));
	}
	

	// Allocate memory for host and device variables
	double* h_mins = new double[gridSize];
	int* h_minIndices = new int[gridSize];
	unsigned char* d_last;
	double* d_mins;
	int* d_minIndices;

	cudaMalloc(&d_mins, gridSize * sizeof(double));
	cudaMalloc(&d_minIndices, gridSize * sizeof(int));
	cudaMalloc(&d_last, lastSize * sizeof(double));
	cudaMemcpy(d_last, h_last.ptr(), lastSize * sizeof(double), cudaMemcpyHostToDevice);

	cudaReduction << <gridSize, blockSize, sharedSize >> > (d_last, d_mins, d_minIndices, lastSize, blockSize, blockSize * gridSize);

	cudaMemcpy(h_mins, d_mins, gridSize * sizeof(double), cudaMemcpyDeviceToHost);
	cudaMemcpy(h_minIndices, d_minIndices, gridSize * sizeof(int), cudaMemcpyDeviceToHost);

	// Compare mins of different blocks
	pair<unsigned char, int> min = {h_mins[0], h_minIndices[0]};
	for (int i = 1; i < gridSize; i++) {
		if (min.first > h_mins[i]) {
			min.first = h_mins[i];
			min.second = h_minIndices[i];
		}
	}
	free(h_mins);
	free(h_minIndices);
	cudaFree(d_last);
	cudaFree(d_mins);
	cudaFree(d_minIndices);
	return min.second;
}

Mat removeSeam(Mat& h_image, vector<int> h_seam, eSeamDirection seamDirection) {
	// dummy 1x1x3 to maintain matrix size;
	Mat h_dummy(1, 1, CV_8UC3, Vec3b(0, 0, 0));
	Mat h_output;
	// Initialize output Mat image
	if (seamDirection == VERTICAL)
		h_output = Mat(h_image.rows, h_image.cols - 1, CV_8UC3, Vec3b(0, 0, 0));
	else
		h_output = Mat(h_image.rows - 1, h_image.cols, CV_8UC3, Vec3b(0, 0, 0));
	unsigned char* d_dummy;
	unsigned char* d_image;
	unsigned char* d_output;
	int* d_seam;

	int rowSize = h_image.rows;
	int colSize = h_image.cols;
	int dummySize = h_dummy.rows * h_dummy.step;
	int size = h_image.rows * h_image.step;
	int outputSize = h_output.rows * h_output.step;
	dim3 blockDim(32, 32);
	dim3 gridDim((h_image.cols + blockDim.x - 1) / blockDim.x, (h_image.rows + blockDim.y - 1) / blockDim.y);

	cudaMalloc(&d_dummy, dummySize);
	cudaMalloc(&d_image, size);
	cudaMalloc(&d_output, outputSize);
	cudaMalloc(&d_seam, h_seam.size() * sizeof(int));

	cudaMemcpy(d_dummy, h_dummy.ptr(), dummySize, cudaMemcpyHostToDevice);
	cudaMemcpy(d_image, h_image.ptr(), size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_output, h_output.ptr(), outputSize, cudaMemcpyHostToDevice);
	cudaMemcpy(d_seam, &h_seam[0], h_seam.size() * sizeof(int), cudaMemcpyHostToDevice);

	cudaRemoveSeam << <gridDim, blockDim >> > (d_image, d_output, d_seam, d_dummy, rowSize, colSize, h_image.step, h_output.step, seamDirection);

	cudaMemcpy(h_output.ptr(), d_output, outputSize, cudaMemcpyDeviceToHost);

	cudaFree(d_dummy);
	cudaFree(d_output);
	cudaFree(d_image);
	cudaFree(d_seam);

	return h_output;
}

__global__ void cudaEnergyMap(unsigned char* energy, unsigned char* energyMap, unsigned char* prevEnergy, int rowSize, int colSize, int current, eSeamDirection seamDirection) {
	int idx;
	double topCenter, topLeft, topRight, minEnergy, cumEnergy;

	if (seamDirection == VERTICAL) {
		idx = blockIdx.x * MAX_THREADS + threadIdx.x;

		if (idx >= colSize) {
			return;
		}
		// Find min value of prev row neighbors and add to the current idx's cumEnergy
		topCenter = ((double*)prevEnergy)[idx];
		topLeft = (idx > 0) ? ((double*)prevEnergy)[idx - 1] : ((double*)prevEnergy)[0];
		topRight = (idx < colSize - 1) ? ((double*)prevEnergy)[idx + 1] : ((double*)prevEnergy)[colSize - 1];
		minEnergy = min(topCenter, min(topLeft, topRight));
		cumEnergy = minEnergy + ((double*)energy)[current * colSize + idx];
		__syncthreads();
		//Update cumEnergy in map and prevRow array
		((double*)prevEnergy)[idx] = cumEnergy;
		((double*)energyMap)[current * colSize + idx] = cumEnergy;
	}
	else {
		idx = blockIdx.y * MAX_THREADS + threadIdx.y;

		if (idx >= rowSize) {
			return;
		}
		
		// Find min value of prev row neighbors and add to the current idx's cumEnergy
		topCenter = ((double*)prevEnergy)[idx];
		topLeft = (idx > 0) ? ((double*)prevEnergy)[idx - 1] : ((double*)prevEnergy)[0];
		topRight = (idx < rowSize - 1) ? ((double*)prevEnergy)[idx + 1] : ((double*)prevEnergy)[rowSize - 1];
		minEnergy = min(topCenter, min(topLeft, topRight));
		cumEnergy = minEnergy + ((double*)energy)[idx * colSize + current];
		__syncthreads();
		//Update cumEnergy in map and prevRow array
		((double*)prevEnergy)[idx] = cumEnergy;
		((double*)energyMap)[idx * colSize + current] = cumEnergy;
	}
	
}

__global__ void cudaReduction(unsigned char* last, double* mins, int* minsIndices, int size, int blockSize, int next) {
	// Global index
	int idx = blockIdx.x * blockSize + threadIdx.x;
	// Initialize shared memory arrays
	extern __shared__ unsigned char sharedMemory[];
	double* sharedMins = (double*)sharedMemory;
	int* sharedMinIndices = (int*)(&(sharedMins[blockSize * 2]));
	
	// Since shared memory is shared in a block, the local idx is used while storing the value of the global idx cumEnergy
	sharedMins[threadIdx.x] = (idx < size) ? ((double*)last)[idx] : DBL_MAX;
	sharedMins[threadIdx.x + blockSize] = (idx + next < size) ? ((double*)last)[idx + next] : DBL_MAX;
	sharedMinIndices[threadIdx.x] = (idx < size) ? idx : INT_MAX;
	sharedMinIndices[threadIdx.x + blockSize] = (idx + next < size) ? idx + next : INT_MAX;

	__syncthreads();
	
	// Parallel reduction to get the min of the block
	for (int i = blockSize; i > 0; i >>= 1) {
		if (threadIdx.x < i) {
			if (sharedMins[threadIdx.x] > sharedMins[threadIdx.x + i]) {
				sharedMins[threadIdx.x] = sharedMins[threadIdx.x + i];
				sharedMinIndices[threadIdx.x] = sharedMinIndices[threadIdx.x + i];
			}
		}
		__syncthreads();
	}
	// local idx 0 has the min of the block
	if (threadIdx.x == 0) {
		mins[blockIdx.x] = sharedMins[0];
		minsIndices[blockIdx.x] = sharedMinIndices[0];
	}
}

__global__ void cudaRemoveSeam(unsigned char* image, unsigned char* output, int* seam, 
	unsigned char* dummy, int rowSize, int colSize, int imageStep, int outputStep, eSeamDirection seamDirection) {
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int row = blockIdx.y * blockDim.y + threadIdx.y;

	if (col < colSize && row < rowSize) {
		// Location of colored pixel in input
		int tidImage = row * imageStep + (3 * col);
		int tidOutput;
		if (seamDirection == VERTICAL) {
			// Skip pixel if part of seam
			if (col < seam[row])
				tidOutput = row * outputStep + (3 * col);
			else if (col > seam[row])
				tidOutput = row * outputStep + 3 * (col - 1);
			else return;
		}
		else {
			// Horizontal
			if (row < seam[col])
				// Skip pixel if part of seam
				tidOutput = row * outputStep + (3 * col);
			else if (row > seam[col])
				tidOutput = (row - 1) * outputStep + (3 * col);
			else return;
		}
		// Add pixel from image to output
		output[tidOutput] = image[tidImage];
		output[tidOutput + 1] = image[tidImage + 1];
		output[tidOutput + 2] = image[tidImage + 2];
	}
}
