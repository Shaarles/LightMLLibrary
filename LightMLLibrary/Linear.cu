#pragma once
#include "Linear.hpp"

__global__ void multiply21(float* dev_data_a, float* dev_datab, float* dev_data_result, int dimensions1_a, int strides_a_0, int strides_a_1, int strides_b_0) {
	//Assuming it's 2d*1D (a*b)
	printf("beginning to multiply 2D and 1D tensors in kernel with thread %d\n", threadIdx.x);
	dev_data_result[threadIdx.x] = 0;
	for (int j = 0; j < dimensions1_a; j++) {
		dev_data_result[threadIdx.x] += dev_data_a[threadIdx.x * strides_a_0 + j * strides_a_1] * dev_datab[j * strides_b_0];
	}

}


__global__ void addd(float* dev_data_a, float* dev_data_b, float* dev_data_result) {
	//Assuming it's 1D+1D (a+b)
	dev_data_result[threadIdx.x] = dev_data_a[threadIdx.x] + dev_data_b[threadIdx.x];
}


Linear::Linear(int input_size, int output_size) {
	cout << "Initializing layer with input size " << input_size << " and output size " << output_size << endl;
	int weights_dimension[] = { output_size, input_size };
	int bias_dimension[] = { output_size };


	this->weights = Tensor(weights_dimension, 2);
	cout << "Initialized weights tensor :[" << weights.getDimensions()[0] << ", " << weights.getDimensions()[1] << "]" << endl;

	this->bias = Tensor(bias_dimension, 1);
}


Linear::Linear() {
	this->weights = Tensor();
	this->bias = Tensor();
}


Linear::~Linear() {
	//The tensors will be automatically freed when the layer is destroyed
}



void Linear::forward(const Tensor& input, Tensor& output) {
	cout << "device forward called with input dimensions: " << input.getDimensions()[0] << " and output dimensions: " << output.getDimensions()[0] << endl;
	cout << output.getDimensions()[0] << " " << output.getNdim() << endl;
	cout << bias.getDimensions()[0] << " " << bias.getNdim() << endl;
	Tensor result(output.getDimensions(), output.getNdim());
	cout << "Launching kernel with " << this->weights.getDimensions()[0] << " threads" << endl;
	multiply21 << <1, bias.getDimensions()[0] >> > (weights.getDevData(), input.getDevData(), result.getDevData(), weights.getDimensions()[1], weights.getStrides()[0], weights.getStrides()[1], bias.getStrides()[0]);
	cudaDeviceSynchronize();
	cout << "Finished multiplying, now adding bias" << endl;
	addd << <1, bias.getDimensions()[0] >> > (result.getDevData(), this->bias.getDevData(), output.getDevData());
	cudaDeviceSynchronize();

	cudaMemcpy(output.getData(), output.getDevData(), output.getnbEle() * sizeof(float), cudaMemcpyDeviceToHost);
}

void Linear::gradW(const Tensor& input, const Tensor& grad_output, Tensor& grad_weights) {

}

void Linear::gradB(const Tensor& input, const Tensor& grad_output, Tensor& grad_bias) {

}

void Linear::gradI(const Tensor& input, const Tensor& grad_output, Tensor& grad_input) {

}

void Linear::backward(const Tensor& input, const Tensor& grad_output, Tensor& grad_input) {

}


