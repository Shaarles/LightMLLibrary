#pragma once
#include "GeLU.hpp"


__global__ void tanh(float* input, float* output, int n) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < n) {
		output[idx] = tanhf(input[idx]);
	}
}

__global__ void sqrt(float* input, float* output, int n) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < n) {
		output[idx] = sqrtf(input[idx]);
	}
}

__global__ void gelu(float* input, float* output, int n) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < n) {
		float x = input[idx];
		output[idx] = 0.5f * x * (1.0f + tanhf(0.79788456f * (x + 0.044715f * x * x * x)));
	}
}


Tensor GeLU::forward(Tensor input) {
	this->input = input.copy();
	Tensor output(input.getDimensions(), input.getNdim());
	int blockSize = 256;
	int numBlocks = (input.getnbEle() + blockSize - 1) / blockSize;
	gelu<<<numBlocks, blockSize>>>(input.getDevData(), output.getDevData(), input.getnbEle());
	cudaError_t c_err = cudaMemcpy(output.getData(), output.getDevData(), input.getnbEle() * sizeof(float), cudaMemcpyDeviceToHost);
	if (c_err != cudaSuccess) {
		std::cerr << "cudaMemcpy failed: " << cudaGetErrorString(c_err) << std::endl;
	}
	return output;
}

void GeLU::backward(Tensor input,Tensor gradOutput) {
	Tensor gradInput(input.getDimensions(), input.getNdim());
	int blockSize = 256;
	int numBlocks = (input.getnbEle() + blockSize - 1) / blockSize;
	sqrt<<<numBlocks, blockSize>>>(input.getDevData(), gradInput.getDevData(), input.getnbEle());
	cudaError_t c_err = cudaMemcpy(gradInput.getData(), gradInput.getDevData(), input.getnbEle() * sizeof(float), cudaMemcpyDeviceToHost);
	if (c_err != cudaSuccess) {
		std::cerr << "cudaMemcpy failed: " << cudaGetErrorString(c_err) << std::endl;
	}

	gradOutput = gradInput.copy();
}	

