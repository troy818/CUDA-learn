
#include <chrono>
#include <ctime>
#include <iostream>
#include <iomanip>  // std::setw
#include <stdlib.h>
#include <string.h>

#include <cuda_runtime.h>
#include <curand.h>
#include <cuda_profiler_api.h>

#include "cpu.hpp"
#include "gpu.hpp"

// clock defination
using CLOCK = std::chrono::high_resolution_clock;
using NS = std::chrono::nanoseconds;


inline int cudaDeviceInit(int argc, const char **argv)
{
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    if (deviceCount == 0)
    {
        std::cerr << "CUDA error: no devices supporting CUDA." << std::endl;
        exit(EXIT_FAILURE);
    }

    cudaSetDevice(0);

    return 0;
}

void populateImage(int * image, int width, int height) {
    srand(time(NULL));  // avoid generate pseudorandom value
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            image[i * width + j] = rand() % 256;
        }
    }
}

void diff(int *himage, int *dimage, int width, int height) {
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            if (himage[i * width + j] != dimage[i * width + j]) {
                std::cout << "Expected: " << himage[i * width + j] << ", actual: " << dimage[i * width + j] << ", on: " << i << ", " << j << std::endl;
                exit(0);
            }
        }
    }
}

int main(int argc, char *argv[])
{
    cudaDeviceInit(argc, (const char **)argv);

    int * dimage_src, *dimage_dst, *dimage_tmp;
    int * himage_src, *himage_dst, *himage_tmp, *himage_dst_2;  // himage_dst_2: cpu 2 step
    // Width and height of the image
    int width = 3840, height = 2160, radio = 7;  // for (radio = 2; radio <= 15; radio++)
    std::cout << "\nErosion Process\nwidth: " << width << ", height: " << height << ", radio: " << radio << std::endl;

    (cudaMalloc(&dimage_src, width * height * sizeof(int)));
    (cudaMalloc(&dimage_dst, width * height * sizeof(int)));
    (cudaMalloc(&dimage_tmp, width * height * sizeof(int)));
    (cudaMallocHost(&himage_src, width * height * sizeof(int)));
    (cudaMallocHost(&himage_dst, width * height * sizeof(int)));
    (cudaMallocHost(&himage_dst_2, width * height * sizeof(int)));
    (cudaMallocHost(&himage_tmp, width * height * sizeof(int)));

    // Randomly populate the image
    populateImage(himage_src, width, height);

    std::chrono::time_point<CLOCK> start, end;
    uint64_t elapsed_seconds;

    {  // 1 step cpu
        start = CLOCK::now();
        // Calculate the eroded image on the host
        erosionCPUOneStep(himage_src, himage_dst, width, height, radio);
        end = CLOCK::now();
        elapsed_seconds = std::chrono::duration_cast<NS>(end - start).count();
        std::cout << "\n" << std::right << std::setw(20) << "CPU 1 step: " << std::right << std::setw(12) << elapsed_seconds << " ns, " << \
            std::right << std::setw(12) << (double)elapsed_seconds / 1e+6 << " ms.\n";
    }

    {  // 2 steps cpu
        start = CLOCK::now();
        // Calculate the eroded image on the host
        erosionCPUTwoStep(himage_src, himage_dst_2, width, height, radio);
        end = CLOCK::now();
        elapsed_seconds = std::chrono::duration_cast<NS>(end - start).count();
        std::cout << std::right << std::setw(20) << "CPU 2 step: " << std::right << std::setw(12) << elapsed_seconds << " ns, " << \
            std::right << std::setw(12) << (double)elapsed_seconds / 1e+6 << " ms.\n";
        // Diff the images
        diff(himage_dst, himage_dst_2, width, height);
    }

    for (int num = 0; num < 10; num++) {  // wcy: just run one time
        std::cout << "\nloop num: " << num << std::endl;

        if (num == 4) {
            cudaProfilerStart();  // CUDA Profiler Start
        }

        {  // test different grid and block setting
            start = std::chrono::system_clock::now();
            // Copy the image from the host to the GPU
            (cudaMemcpy(dimage_src, himage_src, width * height * sizeof(int), cudaMemcpyHostToDevice));
            // Calculate the eroded image on the GPU
            NaiveErosionOneStepMod(dimage_src, dimage_dst, width, height, radio);
            // Copy the eroded image to the host
            (cudaMemcpy(himage_tmp, dimage_dst, width * height * sizeof(int), cudaMemcpyDeviceToHost));
            end = std::chrono::system_clock::now();
            elapsed_seconds = std::chrono::duration_cast<NS>(end - start).count();
            std::cout << std::right << std::setw(20) << "GPU 1 step (mod): " << std::right << std::setw(12) << elapsed_seconds << " ns, " << \
                std::right << std::setw(12) << (double)elapsed_seconds / 1e+6 << " ms.\n";
            // Diff the images
            diff(himage_dst, himage_tmp, width, height);
        }

        {  // 1 step gpu (best)
            start = CLOCK::now();
            // Copy the image from the host to the GPU
            (cudaMemcpy(dimage_src, himage_src, width * height * sizeof(int), cudaMemcpyHostToDevice));
            // Calculate the eroded image on the GPU
            NaiveErosionOneStep(dimage_src, dimage_dst, width, height, radio);
            // Copy the eroded image to the host
            (cudaMemcpy(himage_tmp, dimage_dst, width * height * sizeof(int), cudaMemcpyDeviceToHost));
            end = CLOCK::now();
            elapsed_seconds = std::chrono::duration_cast<NS>(end - start).count();
            std::cout << std::right << std::setw(20) << "GPU 1 step (bst): " << std::right << std::setw(12) << elapsed_seconds << " ns, " << \
                std::right << std::setw(12) << (double)elapsed_seconds / 1e+6 << " ms.\n";
            // Diff the images
            diff(himage_dst, himage_tmp, width, height);

        }

        {  // 2 steps gpu
            start = CLOCK::now();
            // Copy the image from the host to the GPU
            (cudaMemcpy(dimage_src, himage_src, width * height * sizeof(int), cudaMemcpyHostToDevice));
            ErosionTwoSteps(dimage_src, dimage_dst, dimage_tmp, width, height, radio);
            // Copy the eroded image to the host
            (cudaMemcpy(himage_tmp, dimage_dst, width * height * sizeof(int), cudaMemcpyDeviceToHost));
            end = CLOCK::now();
            elapsed_seconds = std::chrono::duration_cast<NS>(end - start).count();
            std::cout << std::right << std::setw(20) << "GPU 2 step: " << std::right << std::setw(12) << elapsed_seconds << " ns, " << \
                std::right << std::setw(12) << (double)elapsed_seconds / 1e+6 << " ms.\n";
            // Diff the images
            diff(himage_dst, himage_tmp, width, height);
        }

        {  // 2 steps gpu shared
            start = CLOCK::now();
            // Copy the image from the host to the GPU
            (cudaMemcpy(dimage_src, himage_src, width * height * sizeof(int), cudaMemcpyHostToDevice));
            ErosionTwoStepsShared(dimage_src, dimage_dst, dimage_tmp, width, height, radio);
            // Copy the eroded image to the host
            (cudaMemcpy(himage_tmp, dimage_dst, width * height * sizeof(int), cudaMemcpyDeviceToHost));
            end = CLOCK::now();
            elapsed_seconds = std::chrono::duration_cast<NS>(end - start).count();
            std::cout << std::right << std::setw(20) << "GPU 2 step shr: " << std::right << std::setw(12) << elapsed_seconds << " ns, " << \
                std::right << std::setw(12) << (double)elapsed_seconds / 1e+6 << " ms.\n";
            // Diff the images
            diff(himage_dst, himage_tmp, width, height);
        }

        if (num == 4) {
            cudaProfilerStop();  // CUDA Profiler Stop
        }
    }
    std::cout << "Great!!" << std::endl;

    cudaFree(dimage_src);
    cudaFree(dimage_dst);
    cudaFree(dimage_tmp);
    cudaFreeHost(himage_src);
    cudaFreeHost(himage_dst);
    cudaFreeHost(himage_dst_2);
    cudaFreeHost(himage_tmp);
    cudaDeviceReset();
    return 0;
}