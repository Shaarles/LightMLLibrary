#pragma once
#include <cstdio>
#include <cstdlib>
#include "Module.cu"

using namespace std;

class GeLU : public Module {
private:
	Tensor input;

public:

	Tensor forward(Tensor input) override;

	void backward(Tensor input, Tensor gradOutput) override;
};
