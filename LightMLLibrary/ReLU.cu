#pragma once 
#include "ReLU.hpp"


__global__ void reluForwardKernel(float* input, float* output, float* mask, int size) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < size) {
		float value = input[idx];
		if (value > 0) {
			output[idx] = value;
			mask[idx] = 1.0f; // Mark as active
			
		}
		else {
			output[idx] = 0.0f;
			mask[idx] = 0.0f; // Mark as inactive
		}
	}


}

__global__ void reluBackwardKernel(float* gradOutput, float* mask, float* gradInput, int size) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < size) {
		gradInput[idx] = gradOutput[idx] * mask[idx]; // Only propagate gradients for active neurons
	}
}

Tensor ReLU::forward(Tensor input)  {
	// Implement the forward pass using the weights and bias
	Tensor output(input.getDimensions(), input.getNdim()); // Create an output tensor with the same dimensions as input

	new (&mask) Tensor(input.getDimensions(), input.getNdim()); // Create a mask tensor with the same dimensions as input
	int vec_size = 1;
	for (int i = 0; i < input.getNdim(); i++) {
		vec_size = vec_size * input.getDimensions()[i];
	}


		
	cudaError_t c_err = cudaMemcpy(input.getDevData(), input.getData(), vec_size * sizeof(float), cudaMemcpyHostToDevice);
	if (c_err != cudaSuccess) {
		std::cerr << "cudaMemcpy failed: " << cudaGetErrorString(c_err) << std::endl;
	}
	reluForwardKernel << <(vec_size + 255) / 256, 256 >> > (input.getDevData(), output.getDevData(), mask.getDevData(), vec_size);
		
	cudaDeviceSynchronize();
	c_err = cudaMemcpy(output.getData(), output.getDevData(), vec_size * sizeof(float), cudaMemcpyDeviceToHost);                                                                                                                                                                                                                                                                                        

	if (c_err != cudaSuccess) {
		std::cerr << "cudaMemcpy failed for output data: " << cudaGetErrorString(c_err) << std::endl;
	}

	//veryfing the mask values
	c_err = cudaMemcpy( mask.getData(), mask.getDevData(), vec_size * sizeof(float), cudaMemcpyDeviceToHost);
	if (c_err != cudaSuccess) {
		std::cerr << "cudaMemcpy failed for mask copy from device to host: " << cudaGetErrorString(c_err) << std::endl;
	}

	return output;
}

void ReLU::backward(Tensor input, Tensor gradOutput)  {
	int vec_size = 1;
	for (int i = 0; i < gradOutput.getNdim(); i++) {
		vec_size = vec_size * gradOutput.getDimensions()[i];
	}
	reluBackwardKernel << <(vec_size + 255) / 256, 256 >> > (gradOutput.getDevData(), mask.getDevData(), input.getDevData(), vec_size);
}


vector<Tensor*> ReLU::parameters()  {
	// Return the parameters of the module (weights and bias)
	vector<Tensor*> outputVec = {&mask};
	return outputVec;
}

Tensor ReLU::getMask() {
	return mask;
}
