#pragma once
#include <cstdio>
#include <cstdlib>
#include "Module.cu"

using namespace std;


class SoftMax : public Module {
private:
	Tensor weights;
	bool training;
	Tensor maxc;
	int dim; //not sure I understand what this "dimension of normalization" is.
	Tensor sumc;


public:
	SoftMax(int dim);
	Tensor forward(Tensor input);
		//input is of shape (B, num_features, lasts)
		//output is of shape (B, num_features, lasts)

};

