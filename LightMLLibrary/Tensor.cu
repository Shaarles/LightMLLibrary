#pragma once
#include "Tensor.hpp"

using namespace std;

__global__ void zero_kernel(float* data, int size) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < size) {
		data[idx] = 0.0f;
	}
}

__global__ void one_kernel(float* data, int size) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < size) {
		data[idx] = 1.0f;
	}
}

__global__ void add(float*a, float*b, float* results, int size) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < size) {
		results[idx] = a[idx] + b[idx];
	}
}



Tensor::Tensor(int dimensions[], int ndim) {
	//Constructor for nD tensors, the random initialization is only for weights and bias but I implemented it here for simplicity


	//cout << "Incoming dims: ";
	for (int i = 0; i < ndim; i++)
		cout << dimensions[i] << " ";
	cout << endl;


	this->ndim = ndim;
	this->dimensions = new int[ndim];
	for (int i = 0; i < ndim; i++) {
		this->dimensions[i] = dimensions[i];
	}
	this->strides = new int[ndim];
	nbEle = 1;
	for (int i = 0; i < ndim; i++) {
		strides[ndim - 1 - i] = nbEle;
		//cout << dimensions[i] << " nbEle: " << nbEle << endl;
		nbEle *= dimensions[i];
	}
	//cout << "nbEle: " << nbEle << endl;


	cudaError_t err = cudaMallocHost(&data, nbEle * sizeof(float));
	if (err != cudaSuccess) {
		std::cerr << "cudaMallocHost failed: " << cudaGetErrorString(err) << std::endl;
		data = nullptr;
	}

	err = cudaMalloc(&dev_data, nbEle * sizeof(float));
	if (err != cudaSuccess) {
		std::cerr << "cudaMalloc failed: " << cudaGetErrorString(err) << std::endl;
		dev_data = nullptr;
	}

	for (int i = 0; i < nbEle; i++) {
		data[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
	}

	err = cudaMemcpy(dev_data, data, nbEle * sizeof(float), cudaMemcpyHostToDevice);
	//cout << "Initialized tensor with random values" << endl;
}


	
Tensor::Tensor(float* dataT, int size) {
	//Constructor(?) for tensors with datas, size as an argument is for convenience, it didn't always work to compute it
	this->ndim = 1;
	this->dimensions = new int[1];
	this->dimensions[0] = size;
	this->strides = new int[ndim];
	nbEle = 1;
	for (int i = 0; i < ndim; i++) {
		this->strides[ndim - 1 - i] = nbEle;
		cout << nbEle << endl;
		nbEle *= dimensions[i];
	}
	//cout << "nbEle: " << nbEle << endl;

	cudaError_t err = cudaMallocHost(&data, nbEle * sizeof(float));
	if (err != cudaSuccess) {
		std::cerr << "On host cudaMallocHost failed: " << cudaGetErrorString(err) << std::endl;
		data = nullptr;
	}

	err = cudaMalloc(&dev_data, nbEle * sizeof(float));
	if (err != cudaSuccess) {
		std::cerr << "On host cudaMalloc failed: " << cudaGetErrorString(err) << std::endl;
		dev_data = nullptr;
	}


	//cout << "Copying data to tensor" << endl;
	cudaMemcpy(data, dataT, nbEle * sizeof(float), cudaMemcpyHostToHost);
	//cout << "Copied data to host memory" << endl;
	cudaMemcpy(dev_data, data, nbEle * sizeof(float), cudaMemcpyHostToDevice);
	//cout << "Copied data to tensor" << endl;

}


Tensor::Tensor() {
	//minimal constructor
	this->ndim = 0;
	this->dimensions = nullptr;
	this->strides = nullptr;
	this->data = nullptr;
	this->dev_data = nullptr;
}

Tensor::Tensor(const Tensor& other) {
	//constructor based on other Tensor instances
	this->ndim = other.ndim;
	this->dimensions = new int[ndim];
	this->strides = new int[ndim];
	
	memcpy(dimensions, other.dimensions, ndim * sizeof(int));
	memcpy(strides, other.strides, ndim * sizeof(int));
	this->nbEle = other.nbEle;
	cudaError_t err = cudaMallocHost(&data, nbEle * sizeof(float));
	if (err != cudaSuccess) {
		std::cerr << "cudaMallocHost failed: " << cudaGetErrorString(err) << std::endl;
		data = nullptr;
	}
	err = cudaMalloc(&dev_data, nbEle * sizeof(float));
	if (err != cudaSuccess) {
		std::cerr << "cudaMalloc failed: " << cudaGetErrorString(err) << std::endl;
		dev_data = nullptr;
	}
	cudaMemcpy(data, other.data, nbEle * sizeof(float), cudaMemcpyHostToHost);
	cudaMemcpy(dev_data, other.dev_data, nbEle * sizeof(float), cudaMemcpyDeviceToDevice);
}

Tensor::~Tensor() {
	//destructor to free memory
	cudaFreeHost(data);
	cudaFree(dev_data);
}
bool Tensor::canMultiply(Tensor* a, Tensor* b) {
	return a->getDimensions()[a->getNdim() - 1] == b->getDimensions()[0];
}



	

void Tensor::toString() {
	printf("Tensor with %d dimensions\n", getNdim());
	for (int i = 0; i < getNdim(); i++) {
		printf("Dimension %d: %d\n", i, getDimensions()[i]);
	}
	printf("(");
	for (int i = 0; i < (this->nbEle)-1; i++) {
		printf("%f,", this->data[i]);
	}
	printf("%f)", this->data[this->nbEle - 1]);
}

Tensor Tensor::copy() {
	
	return Tensor(*this);
}

Tensor& Tensor::operator=(const Tensor& other) {
	if (this != &other) {
		// Free existing resources
		cudaFreeHost(data);
		cudaFree(dev_data);
		delete[] dimensions;
		delete[] strides;
		// Copy dimensions and strides
		ndim = other.ndim;
		dimensions = new int[ndim];
		strides = new int[ndim];
		memcpy(dimensions, other.dimensions, ndim * sizeof(int));
		memcpy(strides, other.strides, ndim * sizeof(int));
		nbEle = other.nbEle;
		// Allocate new memory and copy data
		cudaError_t err = cudaMallocHost(&data, nbEle * sizeof(float));
		if (err != cudaSuccess) {
			std::cerr << "cudaMallocHost failed: " << cudaGetErrorString(err) << std::endl;
			data = nullptr;
		}
		err = cudaMalloc(&dev_data, nbEle * sizeof(float));
		if (err != cudaSuccess) {
			std::cerr << "cudaMalloc failed: " << cudaGetErrorString(err) << std::endl;
			dev_data = nullptr;
		}
		err = cudaMemcpy(data, other.data, nbEle * sizeof(float), cudaMemcpyHostToHost);
		if (err != cudaSuccess) {
			std::cerr << "cudaMemcpy failed for host to host copy: " << cudaGetErrorString(err) << std::endl;
		}
		err = cudaMemcpy(dev_data, other.dev_data, nbEle * sizeof(float), cudaMemcpyDeviceToDevice);
		if (err != cudaSuccess) {
			std::cerr << "cudaMemcpy failed for device to device copy: " << cudaGetErrorString(err) << std::endl;
		}
	}
	return *this;
}

void Tensor::zero() {
	int blockSize = 256;
	zero_kernel << < (nbEle + blockSize - 1) / blockSize, blockSize >>> (dev_data, nbEle);
	cudaDeviceSynchronize();
	cudaError_t err = cudaMemcpy(data, dev_data, nbEle * sizeof(float), cudaMemcpyDeviceToHost);
	if (err != cudaSuccess) {
		std::cerr << "cudaMemcpy failed for device to host copy: " << cudaGetErrorString(err) << std::endl;
	}
}

void Tensor::one() {
	int blockSize = 256;
	one_kernel << < (nbEle + blockSize - 1) / blockSize, blockSize >>> (dev_data, nbEle);
	cudaDeviceSynchronize();
	cudaError_t err = cudaMemcpy(data, dev_data, nbEle * sizeof(float), cudaMemcpyDeviceToHost);
	if (err != cudaSuccess) {
		std::cerr << "cudaMemcpy failed for device to host copy: " << cudaGetErrorString(err) << std::endl;
	}
}

// Getters 

int Tensor::getNdim() const{
	return ndim;
}

int Tensor::getnbEle() const {
	return nbEle;
}

int* Tensor::getDimensions() const {
	return dimensions;
}

float* Tensor::getData() const {
	return data;
}

float* Tensor::getDevData() const {
	return dev_data;
}

int* Tensor::getStrides() const {
	return strides;
}

void add(const Tensor& a, const Tensor& b, const Tensor& result) {

	cudaError_t err = cudaMemcpy(a.getDevData(), a.getData(), a.getnbEle() * sizeof(float), cudaMemcpyHostToDevice);
	if (err != cudaSuccess) {
		std::cerr << "cudaMemcpy failed for host to device copy: " << cudaGetErrorString(err) << std::endl;
	}

	err = cudaMemcpy(b.getDevData(), b.getData(), b.getnbEle() * sizeof(float), cudaMemcpyHostToDevice);
	if (err != cudaSuccess) {
		std::cerr << "cudaMemcpy failed for host to device copy: " << cudaGetErrorString(err) << std::endl;
	}
	
	err = cudaMemcpy(result.getDevData(), result.getData(), result.getnbEle() * sizeof(float), cudaMemcpyHostToDevice);
	if (err != cudaSuccess) {
		std::cerr << "cudaMemcpy failed for host to device copy: " << cudaGetErrorString(err) << std::endl;
	}


	add << <(result.getnbEle() + 255) / 256, 256 >> > (a.getDevData(), b.getDevData(), result.getDevData(), result.getnbEle());


	err = cudaMemcpy(a.getData(), a.getDevData(), a.getnbEle() * sizeof(float), cudaMemcpyDeviceToHost);
	if (err != cudaSuccess) {
		std::cerr << "cudaMemcpy failed for device to host copy: " << cudaGetErrorString(err) << std::endl;
	}

	err = cudaMemcpy(b.getData(), b.getDevData(), b.getnbEle() * sizeof(float), cudaMemcpyDeviceToHost);
	if (err != cudaSuccess) {
		std::cerr << "cudaMemcpy failed for device to host copy: " << cudaGetErrorString(err) << std::endl;
	}

	err = cudaMemcpy(result.getData(), result.getDevData(), result.getnbEle() * sizeof(float), cudaMemcpyDeviceToHost);
	if (err != cudaSuccess) {
		std::cerr << "cudaMemcpy failed for device to host copy: " << cudaGetErrorString(err) << std::endl;
	}
}
