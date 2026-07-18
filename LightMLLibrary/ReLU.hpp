#pragma once
#include <cstdio>
#include <cstdlib>
#include "Module.cu"

using namespace std;


class ReLU : public Module {
private:
	Tensor mask; //sert à stocker les mask pour la rétropagation

public:


	Tensor forward(Tensor input) override;
		// Implement the forward pass using the weights and bias

	void backward(Tensor input, Tensor gradOutput) override;
		// Implement the backward pass to compute gradients for weights and bias on device

	vector<Tensor*> parameters() override;
		// Return the parameters of the module (weights and bias)
		// 
		// This is a placeholder implementation

	Tensor getMask();
};
