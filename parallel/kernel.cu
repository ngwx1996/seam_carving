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

__global__ void warm_up_gpu();
__global__ void cudaEnergyMap(unsigned char* energy, unsigned char* energyMap, unsigned char* prevEnergy, int rowSize, int colSize, eSeamDirection seamDirection);
__global__ void cudaEnergyMapLarge(unsigned char* energy, unsigned char* energyMap, unsigned char* prevEnergy, int rowSize, int colSize, int current, eSeamDirection seamDirection);
__global__ void cudaReduction(unsigned char* row, float* mins, int* minsIndices, int size, int blockSize, int next);
__global__ void cudaRemoveSeam(unsigned char* image, int* seam, int rowSize, int colSize, int imageStep, eSeamDirection seamDirection);

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

void warmUpGPU() {
	warm_up_gpu << <1, 1024 >> > ();
}

void getEnergyMap(Mat& h_energy, Mat& h_energyMap, int rowSize, int colSize, eSeamDirection seamDirection) {
	Mat h_prevEnergy;
	unsigned char* d_energy;
	unsigned char* d_energyMap;
	unsigned char* d_prevEnergy;
	int size = rowSize * colSize;

	cudaMalloc(&d_energy, size * sizeof(float));
	cudaMalloc(&d_energyMap, size * sizeof(float));
	cudaMemcpy(d_energy, h_energy.ptr(), size * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_energyMap, h_energyMap.ptr(), size * sizeof(float), cudaMemcpyHostToDevice);

	if (seamDirection == VERTICAL) {
		// Start from first row. Copy first row of energyMap to be used in device
		h_prevEnergy = Mat(1, colSize, CV_32F, float(0));
		h_energy.row(0).copyTo(h_prevEnergy.row(0));

		cudaMalloc(&d_prevEnergy, colSize * sizeof(float));
		cudaMemcpy(d_prevEnergy, h_prevEnergy.ptr(), colSize * sizeof(float), cudaMemcpyHostToDevice);

		int blockSize = min(colSize, MAX_THREADS);
		int gridSize = ((colSize - 1) / MAX_THREADS) + 1;

		if (gridSize == 1) {
			cudaEnergyMap << <gridSize, blockSize >> > (d_energy, d_energyMap, d_prevEnergy, rowSize, colSize, VERTICAL);
		}
		else {
			for (int i = 1; i < rowSize; i++) {
				cudaEnergyMapLarge << <gridSize, blockSize >> > (d_energy, d_energyMap, d_prevEnergy, rowSize, colSize, i, VERTICAL);
			}
		}
	}
	else {
		// Horizontal Seam, reduces height
		// Start from first row. Copy first row of energyMap to be used in device
		h_prevEnergy = Mat(rowSize, 1, CV_32F, float(0));
		h_energy.col(0).copyTo(h_prevEnergy.col(0));

		cudaMalloc(&d_prevEnergy, rowSize * sizeof(float));
		cudaMemcpy(d_prevEnergy, h_prevEnergy.ptr(), rowSize * sizeof(float), cudaMemcpyHostToDevice);

		int blockSize = min(rowSize, MAX_THREADS);
		int gridSize = ((rowSize - 1) / MAX_THREADS) + 1;

		if (gridSize == 1) {
			cudaEnergyMap << <gridSize, blockSize >> > (d_energy, d_energyMap, d_prevEnergy, rowSize, colSize, HORIZONTAL);

		}
		else {
			for (int i = 1; i < colSize; i++) {
				cudaEnergyMapLarge << <gridSize, blockSize >> > (d_energy, d_energyMap, d_prevEnergy, rowSize, colSize, i, HORIZONTAL);
			}
		}
	}

	cudaMemcpy(h_energyMap.ptr(), d_energyMap, size * sizeof(float), cudaMemcpyDeviceToHost);
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
		sharedSize = blockSize * 2 * (sizeof(float) + sizeof(int));
		lastSize = colSize;
		// Copy last row of energyMap to be used in device
		h_last = Mat(1, colSize, CV_32F, float(0));
		h_energyMap.row(rowSize - 1).copyTo(h_last.row(0));
	}
	else {
		// Horizontal Seam, reduces height
		blockSize = min(nextPowerof2(rowSize / 2), MAX_THREADS);
		gridSize = (nextPowerof2(rowSize / 2) - 1) / MAX_THREADS + 1;
		sharedSize = blockSize * 2 * (sizeof(float) + sizeof(int));
		lastSize = rowSize;
		// Copy last row of energyMap to be used in device
		h_last = Mat(rowSize, 1, CV_32F, float(0));
		h_energyMap.col(colSize - 1).copyTo(h_last.col(0));
	}
	

	// Allocate memory for host and device variables
	float* h_mins = new float[gridSize];
	int* h_minIndices = new int[gridSize];
	unsigned char* d_last;
	float* d_mins;
	int* d_minIndices;

	cudaMalloc(&d_mins, gridSize * sizeof(float));
	cudaMalloc(&d_minIndices, gridSize * sizeof(int));
	cudaMalloc(&d_last, lastSize * sizeof(float));
	cudaMemcpy(d_last, h_last.ptr(), lastSize * sizeof(float), cudaMemcpyHostToDevice);

	cudaReduction << <gridSize, blockSize, sharedSize >> > (d_last, d_mins, d_minIndices, lastSize, blockSize, blockSize * gridSize);

	cudaMemcpy(h_mins, d_mins, gridSize * sizeof(float), cudaMemcpyDeviceToHost);
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

void removeSeam(Mat& h_image, vector<int> h_seam, eSeamDirection seamDirection) {
	// dummy 1x1x3 to maintain matrix size;
	Mat h_output;
	unsigned char* d_image;
	int* d_seam;

	int rowSize = h_image.rows;
	int colSize = h_image.cols;
	int size = h_image.rows * h_image.step;
	dim3 blockDim(32, 32);
	dim3 gridDim((h_image.cols + blockDim.x - 1) / blockDim.x, (h_image.rows + blockDim.y - 1) / blockDim.y);

	cudaMalloc(&d_image, size);
	cudaMalloc(&d_seam, h_seam.size() * sizeof(int));

	cudaMemcpy(d_image, h_image.ptr(), size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_seam, &h_seam[0], h_seam.size() * sizeof(int), cudaMemcpyHostToDevice);

	cudaRemoveSeam << <gridDim, blockDim >> > (d_image, d_seam, rowSize, colSize, h_image.step, seamDirection);

	cudaMemcpy(h_image.ptr(), d_image, size, cudaMemcpyDeviceToHost);

	if (seamDirection == VERTICAL)
		h_image = h_image.colRange(0, h_image.cols - 1);
	else
		h_image = h_image.rowRange(0, h_image.rows - 1);

	cudaFree(d_image);
	cudaFree(d_seam);
}

__global__ void warm_up_gpu() {
	unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
	float ia, ib;
	ia = ib = 0.0f;
	ib += ia + tid;
}

__global__ void cudaEnergyMap(unsigned char* energy, unsigned char* energyMap, unsigned char* prevEnergy, int rowSize, int colSize, eSeamDirection seamDirection) {
	int idx;
	float topCenter, topLeft, topRight, minEnergy, cumEnergy;

	if (seamDirection == VERTICAL) {
		idx = blockIdx.x * MAX_THREADS + threadIdx.x;

		for (int current = 1; current < rowSize; current++) {
			if (idx < colSize) {
				// Find min value of prev row neighbors and add to the current idx's cumEnergy
				topCenter = ((float*)prevEnergy)[idx];
				topLeft = (idx > 0) ? ((float*)prevEnergy)[idx - 1] : ((float*)prevEnergy)[0];
				topRight = (idx < colSize - 1) ? ((float*)prevEnergy)[idx + 1] : ((float*)prevEnergy)[colSize - 1];
				minEnergy = min(topCenter, min(topLeft, topRight));
				cumEnergy = minEnergy + ((float*)energy)[current * colSize + idx];
			}
			__syncthreads();
			if (idx < colSize) {
				//Update cumEnergy in map and prevRow array
				((float*)prevEnergy)[idx] = cumEnergy;
				((float*)energyMap)[current * colSize + idx] = cumEnergy;
			}
			__syncthreads();
		}
	}
	else {
		idx = blockIdx.x * MAX_THREADS + threadIdx.x;

		for (int current = 1; current < colSize; current++) {
			if (idx < rowSize) {
				// Find min value of prev row neighbors and add to the current idx's cumEnergy
				topCenter = ((float*)prevEnergy)[idx];
				topLeft = (idx > 0) ? ((float*)prevEnergy)[idx - 1] : ((float*)prevEnergy)[0];
				topRight = (idx < rowSize - 1) ? ((float*)prevEnergy)[idx + 1] : ((float*)prevEnergy)[rowSize - 1];
				minEnergy = min(topCenter, min(topLeft, topRight));
				cumEnergy = minEnergy + ((float*)energy)[idx * colSize + current];
			}
			__syncthreads();
			if (idx < rowSize) {
				//Update cumEnergy in map and prevRow array
				((float*)prevEnergy)[idx] = cumEnergy;
				((float*)energyMap)[idx * colSize + current] = cumEnergy;
			}
			__syncthreads();

		}
	}

}

__global__ void cudaEnergyMapLarge(unsigned char* energy, unsigned char* energyMap, unsigned char* prevEnergy, int rowSize, int colSize, int current, eSeamDirection seamDirection) {
	int idx;
	float topCenter, topLeft, topRight, minEnergy, cumEnergy;

	if (seamDirection == VERTICAL) {
		idx = blockIdx.x * MAX_THREADS + threadIdx.x;

		if (idx >= colSize) {
			return;
		}
		// Find min value of prev row neighbors and add to the current idx's cumEnergy
		topCenter = ((float*)prevEnergy)[idx];
		topLeft = (idx > 0) ? ((float*)prevEnergy)[idx - 1] : ((float*)prevEnergy)[0];
		topRight = (idx < colSize - 1) ? ((float*)prevEnergy)[idx + 1] : ((float*)prevEnergy)[colSize - 1];
		minEnergy = min(topCenter, min(topLeft, topRight));
		cumEnergy = minEnergy + ((float*)energy)[current * colSize + idx];
		__syncthreads();
		//Update cumEnergy in map and prevRow array
		((float*)prevEnergy)[idx] = cumEnergy;
		((float*)energyMap)[current * colSize + idx] = cumEnergy;
	}
	else {
		idx = blockIdx.x * MAX_THREADS + threadIdx.x;

		if (idx >= rowSize) {
			return;
		}

		// Find min value of prev row neighbors and add to the current idx's cumEnergy
		topCenter = ((float*)prevEnergy)[idx];
		topLeft = (idx > 0) ? ((float*)prevEnergy)[idx - 1] : ((float*)prevEnergy)[0];
		topRight = (idx < rowSize - 1) ? ((float*)prevEnergy)[idx + 1] : ((float*)prevEnergy)[rowSize - 1];
		minEnergy = min(topCenter, min(topLeft, topRight));
		cumEnergy = minEnergy + ((float*)energy)[idx * colSize + current];
		__syncthreads();
		//Update cumEnergy in map and prevRow array
		((float*)prevEnergy)[idx] = cumEnergy;
		((float*)energyMap)[idx * colSize + current] = cumEnergy;
	}

}

__global__ void cudaReduction(unsigned char* last, float* mins, int* minsIndices, int size, int blockSize, int next) {
	// Global index
	int idx = blockIdx.x * blockSize + threadIdx.x;
	// Initialize shared memory arrays
	extern __shared__ unsigned char sharedMemory[];
	float* sharedMins = (float*)sharedMemory;
	int* sharedMinIndices = (int*)(&(sharedMins[blockSize * 2]));
	
	// Since shared memory is shared in a block, the local idx is used while storing the value of the global idx cumEnergy
	sharedMins[threadIdx.x] = (idx < size) ? ((float*)last)[idx] : DBL_MAX;
	sharedMins[threadIdx.x + blockSize] = (idx + next < size) ? ((float*)last)[idx + next] : DBL_MAX;
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

__global__ void cudaRemoveSeam(unsigned char* image, int* seam, int rowSize, int colSize, int imageStep, eSeamDirection seamDirection) {
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	// Location of colored pixel in input
	int tidImage = row * imageStep + (3 * col);
	float temp[3] = { 0 };

	if (col < colSize && row < rowSize) {
		if (seamDirection == VERTICAL) {
			if (col >= seam[row] && col != colSize - 1) {
				temp[0] = image[tidImage + 3];
				temp[1] = image[tidImage + 4];
				temp[2] = image[tidImage + 5];
			}
			else {
				temp[0] = image[tidImage];
				temp[1] = image[tidImage + 1];
				temp[2] = image[tidImage + 2];
			}
		}
		else {
			if (row >= seam[col] && row < rowSize - 1) {
				temp[0] = image[tidImage + imageStep];
				temp[1] = image[tidImage + imageStep + 1];
				temp[2] = image[tidImage + imageStep + 2];
			}
			else {
				temp[0] = image[tidImage];
				temp[1] = image[tidImage + 1];
				temp[2] = image[tidImage + 2];
			}
		}
	}
	
	__syncthreads();
	if (col < colSize && row < rowSize) {
		image[tidImage] = temp[0];
		image[tidImage + 1] = temp[1];
		image[tidImage + 2] = temp[2];
	}
}
