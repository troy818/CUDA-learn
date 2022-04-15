#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda_device_set.h"

#include <stdio.h>

__global__ void hello_cuda()
{
	printf("Hello CUDA world \n");
}

int main()
{
	cudaDeviceInit();
	
	int nx, ny;
	nx = 4;
	ny = 4;

	dim3 block(2, 2);
	dim3 grid(nx / block.x, ny / block.y);

	hello_cuda << < grid, block >> > ();
	cudaDeviceSynchronize();

	cudaDeviceReset();
	return 0;
}