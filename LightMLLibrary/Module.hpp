#pragma once
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <functional>
#include "Tensor.hpp"

using namespace std;

class Module {
protected://enables access to derived classes -> had problem when it was private in  BatchNorm2D where it's particularly useful so :(
	bool training;

public:
	virtual Tensor forward(Tensor input);


	virtual void backward(Tensor input, Tensor gradOutput);


	virtual vector<Tensor*> parameters();

};