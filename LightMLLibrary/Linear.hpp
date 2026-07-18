#pragma once
#include <functional>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "Module.cu"

using namespace std;



class Linear : public Module {
private:
	Tensor weights;
public:

	Tensor bias;
	Linear(int input_size, int output_size);


	Linear();


	~Linear();
		//The tensors will be automatically freed when the layer is destroyed



	void forward(const Tensor& input, Tensor& output);

	void gradW(const Tensor& input, const Tensor& grad_output, Tensor& grad_weights);

	void gradB(const Tensor& input, const Tensor& grad_output, Tensor& grad_bias);

	void gradI(const Tensor& input, const Tensor& grad_output, Tensor& grad_input);

	void backward(const Tensor& input, const Tensor& grad_output, Tensor& grad_input);


};
