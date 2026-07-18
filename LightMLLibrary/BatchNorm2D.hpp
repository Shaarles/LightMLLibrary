#pragma once

#include <cstdio>
#include <cstdlib>
#include "Module.cu"

using namespace std;


class BatchNorm2D : public Module {

private:
	//used only in inference
	Tensor running_mean;
	Tensor running_var;

	//used both in inference and training
	Tensor gamma;
	Tensor beta;


	//used only in training
	Tensor batch_mean;
	Tensor batch_var;

	int num_features;
public:


	BatchNorm2D(int num_features);

	Tensor forward(Tensor input) override;
		/*
		Implementing the forward pass for batch normalization


		input are assumed to be in the format(B, C, H, W, ...)
		but for Loukka's network input are assumed to be in the format(B, C, H, W)
			B -> the batch index
			C -> the channel index
			H -> the height index
			W -> the width index
			but just in case I'm going to make it work for any number of dimensions

		*/

	// Implement the backward pass 	
	void backward(Tensor input, Tensor gradOutput) override;

	bool is_training() const;
	void set_training(bool t);

};