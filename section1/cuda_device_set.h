#include <iostream>

int cudaDeviceInit()
{
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    if (deviceCount == 0)
    {
        std::cout << "CUDA error: no devices supporting CUDA." << std::endl;
        exit(EXIT_FAILURE);
    }

    cudaSetDevice(0);

    return 0;
}
