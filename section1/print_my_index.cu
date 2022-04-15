#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda_device_set.h"

#include <stdio.h>


__global__ void print_my_index()
{
	int tid = threadIdx.x;
	int bid = blockIdx.x;
	printf("my id :%d , block_id :%d \n",tid,bid);
}


int main()
{
	cudaDeviceInit();
	
	printf("hello from main \n");
	print_my_index << <4, 10 >> > ();
	cudaDeviceSynchronize();
	return 0;
}