#pragma once
#include "Module.hpp"
#include "Tensor.cu"

class Module {

protected:
	bool training;
	void eval() {
		training = false;
	}
	void train() {
		training = true;
	}
public:
	Module() {

	}
	virtual ~Module() {

	}
	virtual void forward(Tensor input);
	virtual void backward(Tensor input, Tensor gradOutput);
	virtual vector<Tensor*> parameters();

};

void Module::forward(Tensor input) {
	// Implement the forward pass for the module
	// This is a placeholder implementation
	return;
}

void Module::backward(Tensor input, Tensor gradOutput) {
	// Implement the backward pass to compute gradients for weights and bias
	// This is a placeholder implementation
	return;
}


vector<Tensor*> Module::parameters() {
	// Return the parameters of the module (weights and bias)
	// This is a placeholder implementation
	return vector<Tensor*>();
}


