#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <random>



class Tensor {
	//2D and 3D tensors are the only ones we need for our network

private:
	int ndim;
	int* dimensions;
	int* strides = nullptr;//kinda like coordinates?
	float* data = nullptr;
	float* dev_data = nullptr; //for GPU tensors
	int nbEle;
public:



	Tensor(int dimensions[], int ndim);
		//Constructor for nD tensors, the random initialization is only for weights and bias but I implemented it here for simplicity


	Tensor(float* dataT, int size);
		//Constructor(?) for tensors with datas, size as an argument is for convenience, it didn't always work to compute it


	Tensor();
		//minimal constructor

	Tensor(const Tensor& other);
		//constructor based on other Tensor instances

	~Tensor();
		//destructor to free memory
	static bool canMultiply(Tensor a, Tensor b);



	static Tensor add(float* a, float* b, int n);



	void toString();

	Tensor copy();

	Tensor& operator=(const Tensor& other);

	void zero();

	void one();

	// Getters 

	int getNdim() const;

	int getnbEle() const;

	int* getDimensions() const;

	float* getData() const;

	float* getDevData() const;

	int* getStrides() const;

	/*
	void sync_data(const char* from, const char* to,const size_t size)
		Almost the end of the project and finally thought of something that would synchronize at once rather than doing it by hand everytime lol


		cudaError_t err = cudaMemcpy(to, from, size, cudaMemcpyDefault);
		if (err != cudaSuccess) {
			std::cerr << "cudaMemcpy failed: " << cudaGetErrorString(err) << std::endl;
		}
	}

	*/
};
