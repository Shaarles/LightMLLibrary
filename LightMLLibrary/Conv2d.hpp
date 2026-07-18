#pragma once
#include <cstdio>
#include "Module.cu"
#include <cstdlib>
#include <cublas.h>

using namespace std;



class Conv2D : public Module {
private:
	int in_channels;
	int out_channels;
	int* kernel_size;

	//No implementation of different size of stride and padding as Loukka uses int stride and padding
	int stride;
	int padding;

	//No dilation used as Loukka does not use it, but it can be added later if needed
	Tensor bias;
public:

	Tensor weights;
	Conv2D(int in_channels, int out_channels, int kernel_size, int stride, int padding);
	

	void forward(Tensor& input, int batch_size, int height, int width, Tensor& output);

	void backward(Tensor& input, Tensor& gradOutput, Tensor& gradInput, Tensor& gradWeights, Tensor& gradBias, int batch_size, int height, int width);
		// on suppose que gradInput, gradWeights, gradBias sont déjà alloués
		// et initialisés à 0 (important pour les atomicAdd)

	float* getWeights();

	float* getBias();

	float* getDevWeights();

	float* getDevBias();
};
