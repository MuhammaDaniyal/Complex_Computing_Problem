#include <cuda_runtime.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>
#include <stdlib.h>
#include "base.h"
#include "error.h"
#include "convolve.h"
#include "klt_util.h"

#define TILE_WIDTH 32
#define TILE_HEIGHT 16 // Can be changed for performance tuning
#define MAX_RADIUS 35 // Half of maximum kernel size (71)

// Fixed CUDA error checking
static void checkCudaError(cudaError_t err, const char *file, int line) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s (error code %d) at %s:%d\\n",
                cudaGetErrorString(err), err, file, line);
        exit(EXIT_FAILURE);
    }
}

#define CUDA_CHECK(call) checkCudaError(call, __FILE__, __LINE__)
__constant__ float device_kernel[71];
__constant__ float device_kernel_ver[71];

__global__ void _convolveImageVertSharedKernel(
    int radius, 
    int width,
    const float *imgin,
    float *imgout,
    int ncols,
    int nrows)
{
    extern __shared__ float tile[]; // shared memory for vertical strip

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int col = blockIdx.x * TILE_WIDTH + tx;
    int row = blockIdx.y * TILE_HEIGHT + ty;

    if (col >= ncols) return;

    // Each column has its own vertical slice in shared memory
    int tile_pitch = TILE_HEIGHT + 2 * radius;
    int tile_offset = tx * tile_pitch;

    // Global row of first shared pixel (includes top halo)
    int base_row = blockIdx.y * TILE_HEIGHT - radius;

    // Load full column (main region + halos) into shared memory
    for (int s = ty; s < tile_pitch; s += blockDim.y) {
        int global_row = base_row + s;
        float val = 0.0f;
        if (global_row >= 0 && global_row < nrows) {
            val = imgin[global_row * ncols + col];
        }
        tile[tile_offset + s] = val;
    }

    __syncthreads();

    // Compute convolution for valid region
    if (row < nrows) {
        int idx = row * ncols + col;
        // Border pixels ? zero (same as CPU)
        if (row < radius || row >= nrows - radius) {
            imgout[idx] = 0.0f;
            return;
        }

        float sum = 0.0f;
        int local_row = ty + radius;
        // Apply kernel in reverse order (to match CPU loop)
        for (int k = 0; k < width; ++k) {
            sum += tile[tile_offset + local_row - radius + k] * device_kernel_ver[width - 1 - k];
        }

        imgout[idx] = sum;
    }
}


extern "C" {
    void _convolveImageVertUsingGPU(_KLT_FloatImage imgin, int width, float* kerneldata, _KLT_FloatImage imgout) {
        int radius = width / 2;
        int ncols = imgin->ncols, nrows = imgin->nrows;
    
        assert(width % 2 == 1);
        assert(imgin != imgout);
        assert(imgout->ncols >= imgin->ncols);
        assert(imgout->nrows >= imgin->nrows);
    
        float *d_imgin, *d_imgout;
        size_t img_size = sizeof(float) * ncols * nrows;
    
        CUDA_CHECK(cudaMalloc(&d_imgin, img_size));
        CUDA_CHECK(cudaMalloc(&d_imgout, img_size));
    
        CUDA_CHECK(cudaMemcpy(d_imgin, imgin->data, img_size, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpyToSymbol(device_kernel_ver, kerneldata, sizeof(float) * width));
    
        dim3 block(TILE_WIDTH, TILE_HEIGHT);
        dim3 grid((ncols + TILE_WIDTH - 1) / TILE_WIDTH,
                  (nrows + TILE_HEIGHT - 1) / TILE_HEIGHT);
    
        int shared_mem_size = TILE_WIDTH * (TILE_HEIGHT + 2 * radius) * sizeof(float);
    
        _convolveImageVertSharedKernel<<<grid, block, shared_mem_size>>>(
            radius, width, d_imgin, d_imgout, ncols, nrows);
    
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
            fprintf(stderr, "Kernel launch error: %s\n", cudaGetErrorString(err));
    
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(imgout->data, d_imgout, img_size, cudaMemcpyDeviceToHost));
    
        cudaFree(d_imgin);
        cudaFree(d_imgout);
    }
}

__global__ void _convolveImageHorizGPU(float *ptrrow, int radius, int width, float *ptrout, int ncols, int nrows) {
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (j < nrows && i < ncols) {
        int idx = j * ncols + i;
        if (i < radius)
            ptrout[idx] = 0.0f;
        else if (i >= radius && i < ncols - radius) {
            float sum = 0.0f;
            int x = 0;
            for (int k = width-1; k >= 0; k--) {
                sum += ptrrow[idx - radius + x] * device_kernel[k];
                x++;
            }
            ptrout[idx] = sum;
        }
        else if (i >= ncols - radius)
            ptrout[idx] = 0.0f;
    }
}

extern "C" {
    void _convolveImageHorizUsingGPU(_KLT_FloatImage imgin, int width, float* kerneldata, _KLT_FloatImage imgout) {
        int radius = width / 2;
        int ncols = imgin->ncols, nrows = imgin->nrows;
        
        assert(width % 2 == 1);
        assert(imgin != imgout);
        assert(imgout->ncols >= imgin->ncols);
        assert(imgout->nrows >= imgin->nrows);

        float *ptrrow_device;
        float *ptrout_device;
        int size_of_imgin = sizeof(float) * ncols * nrows;
        int size_of_imgout = sizeof(float) * imgout->ncols * imgout->nrows;

        CUDA_CHECK(cudaMalloc((void**)&ptrrow_device, size_of_imgin));
        CUDA_CHECK(cudaMalloc((void**)&ptrout_device, size_of_imgout));

        CUDA_CHECK(cudaMemcpy(ptrrow_device, imgin->data, size_of_imgin, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpyToSymbol(device_kernel, kerneldata, sizeof(float)*71));    //copy to constant memory

        dim3 block(32, 32, 1);
        dim3 grid((ncols + block.x - 1) / block.x, (nrows + block.y - 1) / block.y, 1);
        _convolveImageHorizGPU<<<grid, block>>>(ptrrow_device, radius, width, ptrout_device, ncols, nrows);

        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(imgout->data, ptrout_device, size_of_imgout, cudaMemcpyDeviceToHost));
        
        cudaFree(ptrrow_device);
        cudaFree(ptrout_device);
    }
}
