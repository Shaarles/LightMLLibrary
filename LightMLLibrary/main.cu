#include "Tensor.cu"
#include "Conv2d.cu"
#include <iostream>

int main() {
	// Example usage of the Conv2d class
	float* aData = new float[3];

	float* bData = new float[3];

	for (int i = 0; i < 3; i++) {
		aData[i] = 1.0f;
	}
	Tensor a = Tensor(aData,3);
	
	Conv2D conv(1, 1, 3, 1, 0); // Example parameters: in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=0
	Tensor output = Tensor(new float[1], 1); // Placeholder for output tensor
	conv.forward(a, 1, 3, 3, output); 

	output.toString(); // Print the output tensor
	conv.getWeights(); // Get the weights of the convolutional layer

	return 0;
}