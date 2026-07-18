#pragma once
#include "Module.hpp"
#include "Tensor.cu"

Tensor Module::forward(Tensor input) {
	// Implement the forward pass using the weights and bias
	// This is a placeholder implementation
	return Tensor();
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
