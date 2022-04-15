# Section1

1. cudaDeviceInit
the cudaDeviceInit function is defined in `cuda_device_set.h`, you can change the Nvidia card you want to use by setting `cudaSetDevice` with different number. The number can be seen by the result of the command `nvidia-smi`. BTW, you can run the code without cudaDeviceInit function, which the code will run on the default Nvidia card.

1. how to build
change the code name of *.cu in [`build.sh`](./build.sh) with the code you want to build.
*Tips: if the code is 14_sum_array.cu, you should add common.cpp too.*

1. how to run
```bash
./cuda_learn
```