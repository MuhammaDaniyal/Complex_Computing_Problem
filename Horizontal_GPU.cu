#include <cuda_runtime.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>
#include <stdlib.h>
#include "base.h"
#include "error.h"
#include "convolve.h"
#include "klt_util.h"

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

__global__ void _convolveImageVertKernel(
    int radius, 
    int width,
    float *imgin,      
    float *imgout,     
    int ncols,
    int nrows)
{
    // Calculate which pixel (row, col) this thread is responsible for
    int col_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int row_idx = blockIdx.y * blockDim.y + threadIdx.y;
    
    // Boundary check (MUST use OR, not AND)
    if (col_idx >= ncols || row_idx >= nrows)
        return;
    
    // Calculate linear index for this pixel
    int idx = row_idx * ncols + col_idx;
    
    if (row_idx < radius) {
        imgout[idx] = 0.0f;
        return;  // Done with this thread
    }
    
    if (row_idx >= nrows - radius) {
        imgout[idx] = 0.0f;
        return;  // Done with this thread
    }
    
    
    float sum = 0.0f;
    int x = 0;
    
    // Start position: row_idx - radius, same column
    int start_idx = (row_idx - radius) * ncols + col_idx;
    
   for (int k = 0; k < width; k++) {

        int current_idx = start_idx + k * ncols;  // Move down one row each iteration
   }

    for (int k = width - 1; k >= 0; k--) {
        int current_row = row_idx - radius + x;
        int current_idx = current_row * ncols + col_idx;
        sum += imgin[current_idx] * device_kernel_ver[k];
        x++;
    }
    
    // Store result
    imgout[idx] = sum;
}

extern "C" {
    void _convolveImageVertUsingGPU(_KLT_FloatImage imgin, int width, float* kerneldata, _KLT_FloatImage imgout) {
        int radius = width / 2;
        int ncols = imgin->ncols, nrows = imgin->nrows;
       
        assert(width % 2 == 1);
        assert(imgin != imgout);
        assert(imgout->ncols >= imgin->ncols);
        assert(imgout->nrows >= imgin->nrows);

        float *d_imgin, *d_imgout, *d_kernel;
        int img_size = sizeof(float) * ncols * nrows;
        int out_size = sizeof(float) * imgout->ncols * imgout->nrows;

        CUDA_CHECK(cudaMalloc(&d_imgin, img_size));
        CUDA_CHECK(cudaMalloc(&d_imgout, out_size));
        //CUDA_CHECK(cudaMalloc(&d_kernel, sizeof(float)*71));

        CUDA_CHECK(cudaMemcpy(d_imgin, imgin->data, img_size, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_imgout, imgout->data, out_size, cudaMemcpyHostToDevice));
        //CUDA_CHECK(cudaMemcpy(d_kernel, kerneldata, sizeof(float)*71, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpyToSymbol(device_kernel_ver, kerneldata, sizeof(float)*71));

        dim3 block(32, 32);
        dim3 grid((ncols + block.x - 1) / block.x, (nrows + block.y - 1) / block.y);

        _convolveImageVertKernel<<<grid, block>>>(
             radius,width, d_imgin, d_imgout,ncols, nrows
        );

        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(imgout->data, d_imgout, out_size, cudaMemcpyDeviceToHost));

        cudaFree(d_imgin);
        cudaFree(d_imgout);
        //cudaFree(d_kernel);
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
        CUDA_CHECK(cudaMemcpy(ptrout_device, imgout->data, size_of_imgout, cudaMemcpyHostToDevice));
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
