#pragma once
#include "SoftMax.hpp"



//implementation of a kind of reduced max, not the most efficient
// and I think it should be done closely to what was done for batchNorm2d(I might be wrong)
//but let's hope not


__global__ void max_search(int num_features, float* input_data, float* output_data, int B, int lasts) {
	int b = blockIdx.x;
	int c = blockIdx.y;
	float max_value = -FLT_MAX;
	if (b>=B || c>=num_features) return;
	for (int i = 0; i < lasts; i++) {
		int idx = b * num_features * lasts + c * lasts + i;
		if (max_value < input_data[idx]) {
			max_value = input_data[idx];
		}
	}
	output_data[b * num_features + c] = max_value;
}

__global__ void t_to_t_substraction(float* input_data, float* max_data, int B, int C, int lasts) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int total = B * C * lasts;
	if (idx >= total) return;

	int i = idx % lasts;
	int tmp = idx / lasts;
	int b = tmp / C;

	input_data[idx] -= max_data[b * lasts + i];
}

__global__ void exp_and_sum(float* input_data, float* sum_data, int B, int C, int lasts) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int total = B * C * lasts;
	if (idx >= total) return;

	int i = idx % lasts;
	int tmp = idx / lasts;
	int b = tmp / C;

	float val = expf(input_data[idx]);
	input_data[idx] = val;

	atomicAdd(&sum_data[b * lasts + i], val);
}


__global__ void div_by_sum(float* input_data, float* sum_data, int B, int C, int lasts) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int total = B * C * lasts;
	if (idx >= total) return;

	int i = idx % lasts;
	int tmp = idx / lasts;
	int b = tmp / C;

	input_data[idx] /= sum_data[b * lasts + i];
}


SoftMax::SoftMax(int dim) : dim(dim){
	training = true;

}

Tensor SoftMax::forward(Tensor input) {
	//input is of shape (B, num_features, lasts)
	//output is of shape (B, num_features, lasts)

	int B = input.getDimensions()[0];
	int C = input.getDimensions()[1];

	int maxcdims[] = { B, C };
	maxc = Tensor(maxcdims, 2);

	int lasts = input.getDimensions()[2];
	max_search << <dim3(B, C), 1 >> > (C, input.getDevData(), maxc.getDevData(), B, lasts);
	cudaDeviceSynchronize();

	cudaError_t err = cudaMemcpy(maxc.getData(), maxc.getDevData(), maxc.getnbEle() * sizeof(float), cudaMemcpyDeviceToHost);
	if (err != cudaSuccess) {
		std::cerr << "Error copying data from maxc to device for max_search: " << cudaGetErrorString(err) << std::endl;
	}

	t_to_t_substraction << <dim3((B * C * lasts + 255) / 256), 256 >> > (input.getDevData(), maxc.getDevData(), B , C , lasts);
		

	int sumcdims[] = { B, lasts };
	sumc = Tensor(sumcdims, 2);


	err = cudaMemcpy(maxc.getData(), maxc.getDevData(), maxc.getnbEle() * sizeof(float), cudaMemcpyDeviceToHost);
	if (err != cudaSuccess) {
		std::cerr << "Error copying data from input to device for t_to_t_substraction: " << cudaGetErrorString(err) << std::endl;
	}



	exp_and_sum << <dim3((B * C * lasts + 255) / 256), 256 >> > (input.getDevData(), sumc.getDevData(), B, C, lasts);

	err = cudaMemcpy(sumc.getData(), sumc.getDevData(), sumc.getnbEle() * sizeof(float), cudaMemcpyDeviceToHost);
	if (err != cudaSuccess) {
		std::cerr << "Error copying data from input to device for exp_and_sum: " << cudaGetErrorString(err) << std::endl;
	}




	div_by_sum << <dim3((B * C * lasts + 255) / 256), 256 >> > (input.getDevData(), sumc.getDevData(), B, C, lasts);
	err = cudaMemcpy(input.getData(), input.getDevData(), input.getnbEle() * sizeof(float), cudaMemcpyDeviceToHost);
	if (err != cudaSuccess) {
		std::cerr << "Error copying data from input to device for div_by_sum: " << cudaGetErrorString(err) << std::endl;
	}


	return input;
}
