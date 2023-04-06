#include <iostream>

#include "utils.hpp"

/**
 * Naive erosion kernel with each thread processing a square area.
 */
__global__ void NaiveErosionKernel(int * src, int * dst, int width, int height, int radio) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (y >= height || x >= width) {
        return;
    }
    unsigned int start_i = max(y - radio, 0);
    unsigned int end_i = min(height - 1, y + radio);
    unsigned int start_j = max(x - radio, 0);
    unsigned int end_j = min(width - 1, x + radio);
    int value = 255;
    for (int i = start_i; i <= end_i; i++) {
        for (int j = start_j; j <= end_j; j++) {
            value = min(value, src[i * width + j]);
        }
    }
    dst[y * width + x] = value;
}

void NaiveErosionOneStep(int * src, int * dst, int width, int height, int radio) {
    dim3 block(32, 32);
    dim3 grid(ceil((float)width / block.x), ceil((float)height / block.y));
    NaiveErosionKernel<<<grid,block>>>(src, dst, width, height, radio);
    cudaError_t cudaerr = cudaDeviceSynchronize();
}

// modify 'block' and 'grid' size
void NaiveErosionOneStepMod(int * src, int * dst, int width, int height, int radio) {
    dim3 block(32, 32);
    dim3 grid(ceil((float)width / block.x), ceil((float)height / block.y));
    NaiveErosionKernel<<<grid,block>>>(src, dst, width, height, radio);
    cudaCheckErrors("Kernel execution failed");  // error handling
    cudaError_t cudaerr = cudaDeviceSynchronize();
}


/**
 * Two steps erosion using separable filters
 */
__global__ void ErosionStep2(int * src, int * dst, int width, int height, int radio) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (y >= height || x >= width) {
        return;
    }
    unsigned int start_i = max(y - radio, 0);
    unsigned int end_i = min(height - 1, y + radio);
    int value = 255;
    for (int i = start_i; i <= end_i; i++) {
        value = min(value, src[i * width + x]);
    }
    dst[y * width + x] = value;
}

__global__ void ErosionStep1(int * src, int * dst, int width, int height, int radio) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (y >= height || x >= width) {
        return;
    }
    unsigned int start_j = max(x - radio, 0);
    unsigned int end_j = min(width - 1, x + radio);
    int value = 255;
    for (int j = start_j; j <= end_j; j++) {
        value = min(value, src[y * width + j]);
    }
    dst[y * width + x] = value;
}

void ErosionTwoSteps(int * src, int * dst, int * temp, int width, int height, int radio) {
    dim3 block(16, 16);
    // dim3 block(32, 8);
    dim3 grid(ceil((float)width / block.x), ceil((float)height / block.y));
    ErosionStep1<<<grid,block>>>(src, temp, width, height, radio);
    cudaCheckErrors("Kernel execution failed");  // error handling
    cudaError_t cudaerr = cudaDeviceSynchronize();
    ErosionStep2<<<grid,block>>>(temp, dst, width, height, radio);
    cudaCheckErrors("Kernel execution failed");  // error handling
    cudaerr = cudaDeviceSynchronize();

    // // Occupancy calculatation //
    // int num_blocks;
    // int block_size = 16 * 16;
    // int device;
    // cudaDeviceProp prop;
    // int active_warps;
    // int max_warps;
    // cudaGetDevice(&device);
    // cudaGetDeviceProperties(&prop, device);
    // cudaOccupancyMaxActiveBlocksPerMultiprocessor(
    //     &num_blocks,
    //     ErosionStep1,
    //     block_size,
    //     0
    // );
    // active_warps = num_blocks * block_size / prop.warpSize;
    // max_warps = prop.maxThreadsPerMultiProcessor / prop.warpSize;
    // std::cout << "Calculate Occupancy: " << (double)active_warps / max_warps * 100 << "%" << std::endl;
}


/**
 * Two steps erosion using separable filters with shared memory.
 */
__global__ void ErosionSharedStep2(int * src, int *src_src, int * dst, int radio, int width, int height, int tile_w, int tile_h) {
    extern __shared__ int smem[];
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int x = bx * tile_w + tx;
    int y = by * tile_h + ty - radio;
    smem[ty * blockDim.x + tx] = 255;
    __syncthreads();
    if (x >= width || y < 0 || y >= height) {
        return;
    }
    smem[ty * blockDim.x + tx] = src[y * width + x];
    __syncthreads();
    if (y < (by * tile_h) || y >= ((by + 1) * tile_h)) {
        return;
    }
    int * smem_thread = &smem[(ty - radio) * blockDim.x + tx];
    int val = smem_thread[0];
    for (int yy = 1; yy <= 2 * radio; yy++) {
        val = min(val, smem_thread[yy * blockDim.x]);
    }
    dst[y * width + x] = val;
}

__global__ void ErosionSharedStep1(int * src, int * dst, int radio, int width, int height, int tile_w, int tile_h) {
    extern __shared__ int smem[];
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int x = bx * tile_w + tx - radio;
    int y = by * tile_h + ty;
    smem[ty * blockDim.x + tx] = 255;
    __syncthreads();
    if (x < 0 || x >= width || y >= height) {
        return;
    }
    smem[ty * blockDim.x + tx] = src[y * width + x];
    __syncthreads();
    if (x < (bx * tile_w) || x >= ((bx + 1) * tile_w)) {
        return;
    }
    int * smem_thread = &smem[ty * blockDim.x + tx - radio];
    int val = smem_thread[0];
    for (int xx = 1; xx <= 2 * radio; xx++) {
        val = min(val, smem_thread[xx]);
    }
    dst[y * width + x] = val;
}

void ErosionTwoStepsShared(int * src, int * dst, int * temp, int width, int height, int radio) {
    int tile_w = 640;
    int tile_h = 1;
    // int tile_w = 128;
    // int tile_h = 5;
    dim3 block2(tile_w + (2 * radio), tile_h);
    dim3 grid2(ceil((float)width / tile_w), ceil((float)height / tile_h));
    ErosionSharedStep1<<<grid2,block2,block2.y*block2.x*sizeof(int)>>>(src, temp, radio, width, height, tile_w, tile_h);
    cudaCheckErrors("Kernel execution failed");  // error handling
    cudaError_t cudaerr = cudaDeviceSynchronize();

    // // get function registers
    // cudaFuncAttributes funcAttrib;
    // checkCudaErrors(cudaFuncGetAttributes(&funcAttrib, ErosionSharedStep1));
    // printf("%s numRegs=%d\n", "ErosionSharedStep1", funcAttrib.numRegs);

    // tile_w = 1;
    // tile_h = 1009;
    tile_w = 5;
    tile_h = 128;
    dim3 block3(tile_w, tile_h + (2 * radio));
    dim3 grid3(ceil((float)width / tile_w), ceil((float)height / tile_h));
    ErosionSharedStep2<<<grid3,block3,block3.y*block3.x*sizeof(int)>>>(temp, src, dst, radio, width, height, tile_w, tile_h);
    cudaCheckErrors("Kernel execution failed");  // error handling
    cudaerr = cudaDeviceSynchronize();
}

