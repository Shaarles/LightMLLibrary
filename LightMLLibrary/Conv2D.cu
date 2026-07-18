#pragma once
#include "Conv2D.hpp"

//with padding and stride
__global__ void conv2d( float* input, float* output, float* weights, float* bias, int in_channels, int out_channels, int kernel_height, int kernel_width, int input_width, int input_height,int stride, int padding, int out_Height, int out_Width) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;

	int bz = blockIdx.z;
	int b = bz / out_channels;
	int k = bz % out_channels;



	float sum = 0.0f;

	if (i >= out_Height || j >= out_Width || k >= out_channels) return;//bound check
	int batch_offset = b * in_channels * input_height * input_width;
	for(int ic=0; ic<in_channels; ic++) {
		// Implementation for each input channel
		for (int m = 0; m < kernel_height; m++) {
			for (int n = 0; n < kernel_width; n++) {
				// Calculate the convolution for the current position
				int input_x = stride * i+m-padding;
				int input_y = stride * j+n-padding;
				if (input_x >= 0 && input_x < input_width && input_y >= 0 && input_y < input_height) {
					sum+= input[batch_offset + ic * input_width * input_height + input_y * input_width + input_x] *
						weights[k*kernel_height*kernel_width*in_channels + ic * kernel_height * kernel_width + m * kernel_width + n];
				}
			}
		}
	}

	output[i * out_Width + j + k * out_Height * out_Width+b * out_channels * out_Height * out_Width] = sum + bias[k];
}

__global__ void conv2d_backward(float* input, float* gradOutput, float* weights, float* gradInput, float* gradWeights, float* gradBias, int in_channels, int out_channels, int kernel_height, int kernel_width, int input_width, int input_height, int stride, int padding, int out_Height, int out_Width) {
	// Implement the backward pass to compute gradients for weights and bias
	// This is a placeholder implementation
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;

	int bz = blockIdx.z;
	int b = bz / out_channels;
	int k = bz % out_channels;



	float sum = 0.0f;


	if (i >= out_Height || j >= out_Width) return;

	//Computing the gradient for the bias
	int go_index = i * out_Width + j + k * out_Height * out_Width + b * out_channels * out_Height * out_Width;
	float go = gradOutput[go_index];
	atomicAdd(&gradBias[k], go); 

	//Computing the gradient for the weights and in the input

	int batch_offset_in = b * in_channels * input_height * input_width;

	for (int ic = 0; ic < in_channels; ic++) {
		// Implementation for each input channel
		for (int m = 0; m < kernel_height; m++) {
			for (int n = 0; n < kernel_width; n++) {
				int input_x = stride * i + m - padding;
				int input_y = stride * j + n - padding;
				if (input_x >= 0 && input_x < input_width && input_y >= 0 && input_y < input_height) {
					int in_index =batch_offset_in +ic * (input_height * input_width) +input_y * input_width +input_x;

					int w_index =k * (in_channels * kernel_height * kernel_width) +ic * (kernel_height * kernel_width) +m * kernel_width + n;

					float x = input[in_index];
					float w = weights[w_index];

					atomicAdd(&gradWeights[w_index], go * x);


					atomicAdd(&gradInput[in_index], go * w);
				}
			}
		}
	}

	
}



Conv2D::Conv2D(int in_channels, int out_channels, int kernel_size, int stride, int padding) :
	in_channels(in_channels), out_channels(out_channels), kernel_size(new int[2] {kernel_size, kernel_size}),
	weights(new int[4] {out_channels, in_channels, this->kernel_size[0], this->kernel_size[1]}, 4),
	bias(new int[1] {out_channels}, 1), stride(stride), padding(padding) {
	//learn to initialize like this but I'm too lazy to change the initalization in like other classes, I'll try to do it here

}

void Conv2D::forward(Tensor& input, int batch_size,int height, int width, Tensor& output) {

	int out_Height = 1 + (height - kernel_size[0] + 2 * padding) / stride;
	int out_Width = 1 + (width - kernel_size[1] + 2 * padding) / stride;


	dim3 block(16, 16, 1);
	dim3 grid( (out_Width + block.x - 1) / block.x,  (out_Height + block.y - 1) / block.y, out_channels *batch_size	);
	conv2d << <grid, block >> > (input.getDevData(), output.getDevData(), weights.getDevData(), bias.getDevData(), in_channels, out_channels, kernel_size[0], kernel_size[1], input.getDimensions()[3], input.getDimensions()[2], stride, padding, out_Height, out_Width);

	cudaDeviceSynchronize();

	cudaMemcpy(output.getData(), output.getDevData(), output.getnbEle() * sizeof(float), cudaMemcpyDeviceToHost);

}

void Conv2D::backward(Tensor& input, Tensor& gradOutput, Tensor& gradInput, Tensor& gradWeights, Tensor& gradBias, int batch_size, int height, int width) {
	// on suppose que gradInput, gradWeights, gradBias sont déjà alloués
	// et initialisés à 0 (important pour les atomicAdd)

	int out_Height = 1 + (height - kernel_size[0] + 2 * padding) / stride;
	int out_Width = 1 + (width - kernel_size[1] + 2 * padding) / stride;

	dim3 block(16, 16, 1);
	dim3 grid((out_Width + block.x - 1)  / block.x,	(out_Height + block.y - 1) / block.y,out_channels * batch_size );

	conv2d_backward << <grid, block >> > (input.getDevData(), gradOutput.getDevData(), weights.getDevData(), gradInput.getDevData(), gradWeights.getDevData(), gradBias.getDevData(), in_channels, out_channels,kernel_size[0], kernel_size[1],width, height,	stride, padding, out_Height, out_Width);
	cudaDeviceSynchronize();
}
float* Conv2D::getWeights() {
	return weights.getData();
}

float* Conv2D::getBias() {
	return bias.getData();
}

float* Conv2D::getDevWeights() {
	return weights.getDevData();
}

float* Conv2D::getDevBias() {
	return bias.getDevData();
}
