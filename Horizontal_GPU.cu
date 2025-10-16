#include <cuda.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>
#include <stdlib.h>
#include "base.h"
#include "error.h"
#include "convolve.h"
#include "klt_util.h"

#define CUDA_CHECK(call)                                                      \
do {                                                                          \
    cudaError_t err = call;                                                   \
    if (err != cudaSuccess) {                                                 \
        fprintf(stderr, "CUDA Error: %s (error code %d) at %s:%d\n",          \
                cudaGetErrorString(err), err, __FILE__, __LINE__);            \
        exit(EXIT_FAILURE);                                                   \
    }                                                                         \
} while (0)



__global__ void _convolveImageVertKernel(
    float *kernel, 
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
    
    // =====================================================================
    // PHASE 1: Zero the top edge
    // =====================================================================
    if (row_idx < radius) {
        imgout[idx] = 0.0f;
        return;  // Done with this thread
    }
    
    // =====================================================================
    // PHASE 3: Zero the bottom edge
    // =====================================================================
    if (row_idx >= nrows - radius) {
        imgout[idx] = 0.0f;
        return;  // Done with this thread
    }
    
    // =====================================================================
    // PHASE 2: Convolve middle rows with kernel
    // =====================================================================
    // If we're here, row_idx is in the middle (not top or bottom edge)
    
    float sum = 0.0f;
    
    // Start position: row_idx - radius, same column
    int start_idx = (row_idx - radius) * ncols + col_idx;
    
    // Apply kernel vertically
    for (int k = 0; k < width; k++) {
        int current_idx = start_idx + k * ncols;  // Move down one row each iteration
        sum += imgin[current_idx] * kernel[k];
    }
    
    // Store result
    imgout[idx] = sum;
}


extern "C" {
    void _convolveImageVertUsingGPU(_KLT_FloatImage imgin, int width, float* kerneldata, _KLT_FloatImage imgout) {
        float *ptrcol = imgin->data;
        float *ptrout = imgout->data;
        int radius = width / 2;
        int ncols = imgin->ncols, nrows = imgin->nrows;
        
        /* Kernel width must be odd */
        assert(width % 2 == 1);
        /* Must read from and write to different images */
        assert(imgin != imgout);
        /* Output image must be large enough to hold result */
        assert(imgout->ncols >= imgin->ncols);
        assert(imgout->nrows >= imgin->nrows);

        float *ptrcol_device;
        float *ptrout_device;
        float *kerneldata_device;
        int size_of_imgin = sizeof(float) * ncols * nrows;
        int size_of_imgout = sizeof(float) * imgout->ncols * imgout->nrows;

        CUDA_CHECK(cudaMalloc((void**)&ptrcol_device, size_of_imgin));
        CUDA_CHECK(cudaMalloc((void**)&ptrout_device, size_of_imgout));
        CUDA_CHECK(cudaMalloc((void**)&kerneldata_device, sizeof(float)*71));

        CUDA_CHECK(cudaMemcpy(ptrcol_device, imgin->data, size_of_imgin, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(ptrout_device, imgout->data, size_of_imgout, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(kerneldata_device, kerneldata, sizeof(float)*71, cudaMemcpyHostToDevice));

        dim3 block(32, 32, 1);
        dim3 grid((ncols + block.x - 1) / block.x, (nrows + block.y - 1) / block.y, 1);
        
        _convolveImageVertKernel<<<grid, block>>>(kerneldata_device, radius, width, ptrcol_device, ptrout_device, ncols, nrows);
        
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(imgout->data, ptrout_device, size_of_imgout, cudaMemcpyDeviceToHost));
        
        cudaFree(ptrcol_device);
        cudaFree(ptrout_device);
        cudaFree(kerneldata_device);
    }
}



__global__ void _convolveImageHorizGPU(float *kernel,float *ptrrow, int radius, int width, float *ptrout, int ncols, int nrows){
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (j < nrows && i < ncols) {
		int idx = j * ncols + i;
		if (i < radius)
			ptrout[idx] = 0.0f;
		if (i >= radius && i < ncols - radius){
			float sum = 0.0f;
			int x = 0;
			for (int k = width-1; k >= 0; k--) {
				sum += ptrrow[idx - radius + x] * kernel[k];
				x++;
			}
			ptrout[idx] = sum;
		}
		if (i >= ncols - radius)
			ptrout[idx] = 0.0f;
	}
}

extern "C" {
	void _convolveImageHorizUsingGPU(_KLT_FloatImage imgin, int width, float* kerneldata, _KLT_FloatImage imgout) {
	    float *ptrrow = imgin->data;
	    float *ptrout = imgout->data;
	    int radius = width / 2;
	    int ncols = imgin->ncols, nrows = imgin->nrows;
	    
	    /* Kernel width must be odd */
	    assert(width % 2 == 1);
	    /* Must read from and write to different images */
	    assert(imgin != imgout);
	    /* Output image must be large enough to hold result */
	    assert(imgout->ncols >= imgin->ncols);
	    assert(imgout->nrows >= imgin->nrows);

		float *ptrrow_device;
		float *ptrout_device;
		float *kerneldata_device;
		int size_of_imgin = sizeof(float) * ncols * nrows;
		int size_of_imgout = sizeof(float) * imgout->ncols * imgout->nrows;

		cudaMalloc((void**)&ptrrow_device, size_of_imgin);
		CUDA_CHECK(cudaMalloc((void**)&ptrout_device, size_of_imgout));
		cudaMalloc((void**)&kerneldata_device, sizeof(float)*71);

		cudaMemcpy(ptrrow_device, imgin->data, size_of_imgin, cudaMemcpyHostToDevice);
		CUDA_CHECK(cudaMemcpy(ptrout_device, imgout->data, size_of_imgout, cudaMemcpyHostToDevice));
		cudaMemcpy(kerneldata_device, kerneldata, sizeof(float)*71, cudaMemcpyHostToDevice);

		dim3 block(32, 32, 1);
		dim3 grid((ncols + block.x - 1) / block.x, (nrows + block.y - 1) / block.y, 1);
		_convolveImageHorizGPU <<<grid,block>>> (kerneldata_device, ptrrow_device, radius, width, ptrout_device, ncols, nrows);
		CUDA_CHECK(cudaDeviceSynchronize());
		CUDA_CHECK(cudaMemcpy(imgout->data, ptrout_device, size_of_imgout, cudaMemcpyDeviceToHost));
		cudaFree(ptrrow_device);
		cudaFree(ptrout_device);
		cudaFree(kerneldata_device);
	}
}
