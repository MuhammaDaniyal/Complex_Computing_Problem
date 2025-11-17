/*********************************************************************
 * convolve.c - Enhanced with comprehensive timing and metrics
 *********************************************************************/

/* Standard includes */
#include <assert.h>
#include <math.h>
#include <openacc.h>
#include <stdlib.h> /* malloc(), realloc() */
/* Our includes */
#include "base.h"
#include "convolve.h"
#include "error.h"
#include "klt_util.h" /* printing */
#include <time.h>

#define MAX_KERNEL_WIDTH 71

// =============== GLOBAL TIMING ACCUMULATORS ===============
static double total_gpu_compute_time = 0.0;
static double total_cpu_compute_time = 0.0;
static double total_memory_transfer_time = 0.0;
static int total_convolution_calls = 0;
static int first_image_width = 0;
static int first_image_height = 0;
static int image_size_recorded = 0;

typedef struct {
  int width;
  float data[MAX_KERNEL_WIDTH];
} ConvolutionKernel;

/* Kernels */
static ConvolutionKernel gauss_kernel;
static ConvolutionKernel gaussderiv_kernel;
static float sigma_last = -10.0;

/*********************************************************************
 * _KLTToFloatImage
 *
 * Given a pointer to image data (probably unsigned chars), copy
 * data to a float image.
 */

void _KLTToFloatImage(KLT_PixelType *img, int ncols, int nrows,
                      _KLT_FloatImage floatimg) {
  KLT_PixelType *ptrend = img + ncols * nrows;
  float *ptrout = floatimg->data;

  /* Output image must be large enough to hold result */
  assert(floatimg->ncols >= ncols);
  assert(floatimg->nrows >= nrows);

  floatimg->ncols = ncols;
  floatimg->nrows = nrows;

  while (img < ptrend)
    *ptrout++ = (float)*img++;
}

/*********************************************************************
 * _computeKernels
 */

static void _computeKernels(float sigma, ConvolutionKernel *gauss,
                            ConvolutionKernel *gaussderiv) {
  const float factor = 0.01f; /* for truncating tail */
  int i;

  assert(MAX_KERNEL_WIDTH % 2 == 1);
  assert(sigma >= 0.0);

  /* Compute kernels, and automatically determine widths */
  {
    const int hw = MAX_KERNEL_WIDTH / 2;
    float max_gauss = 1.0f, max_gaussderiv = (float)(sigma * exp(-0.5f));

    /* Compute gauss and deriv */
    for (i = -hw; i <= hw; i++) {
      gauss->data[i + hw] = (float)exp(-i * i / (2 * sigma * sigma));
      gaussderiv->data[i + hw] = -i * gauss->data[i + hw];
    }

    /* Compute widths */
    gauss->width = MAX_KERNEL_WIDTH;
    for (i = -hw; fabs(gauss->data[i + hw] / max_gauss) < factor;
         i++, gauss->width -= 2)
      ;
    gaussderiv->width = MAX_KERNEL_WIDTH;
    for (i = -hw; fabs(gaussderiv->data[i + hw] / max_gaussderiv) < factor;
         i++, gaussderiv->width -= 2)
      ;
    if (gauss->width == MAX_KERNEL_WIDTH ||
        gaussderiv->width == MAX_KERNEL_WIDTH)
      KLTError("(_computeKernels) MAX_KERNEL_WIDTH %d is too small for "
               "a sigma of %f",
               MAX_KERNEL_WIDTH, sigma);
  }

  /* Shift if width less than MAX_KERNEL_WIDTH */
  for (i = 0; i < gauss->width; i++)
    gauss->data[i] = gauss->data[i + (MAX_KERNEL_WIDTH - gauss->width) / 2];
  for (i = 0; i < gaussderiv->width; i++)
    gaussderiv->data[i] =
        gaussderiv->data[i + (MAX_KERNEL_WIDTH - gaussderiv->width) / 2];
  /* Normalize gauss and deriv */
  {
    const int hw = gaussderiv->width / 2;
    float den;

    den = 0.0;
    for (i = 0; i < gauss->width; i++)
      den += gauss->data[i];
    for (i = 0; i < gauss->width; i++)
      gauss->data[i] /= den;
    den = 0.0;
    for (i = -hw; i <= hw; i++)
      den -= i * gaussderiv->data[i + hw];
    for (i = -hw; i <= hw; i++)
      gaussderiv->data[i + hw] /= den;
  }

  sigma_last = sigma;
}

/*********************************************************************
 * _KLTGetKernelWidths
 *
 */

void _KLTGetKernelWidths(float sigma, int *gauss_width, int *gaussderiv_width) {
  _computeKernels(sigma, &gauss_kernel, &gaussderiv_kernel);
  *gauss_width = gauss_kernel.width;
  *gaussderiv_width = gaussderiv_kernel.width;
}

/*********************************************************************
 * _convolveImageHoriz
 */
//#include "Horizontal_GPU.h"
//#include <cuda.h>
#include <sys/time.h>

static void _convolveImageHoriz(_KLT_FloatImage imgin, ConvolutionKernel kernel, _KLT_FloatImage imgout) {
  total_convolution_calls++;
  if (!image_size_recorded) {
    first_image_width = imgin->ncols;
    first_image_height = imgin->nrows;
    image_size_recorded = 1;
  }

  // ====================== GPU TIMING START ======================
  //cudaEvent_t startGPU, stopGPU;
  float gpuTime = 0.0f;
  /*cudaEventCreate(&startGPU);
  cudaEventCreate(&stopGPU);
  cudaEventRecord(startGPU, 0);*/

  // Step 1: Run GPU convolution (includes memory transfers)
 //_convolveImageHorizUsingGPU(imgin, kernel.width, kernel.data, imgout);

  // Stop GPU timer
  //cudaEventRecord(stopGPU, 0);
  //cudaEventSynchronize(stopGPU);
  //cudaEventElapsedTime(&gpuTime, startGPU, stopGPU);

  total_gpu_compute_time += gpuTime;

  //printf("\n[GPU HORIZ] Time: %.3f ms (Image: %dx%d)\n", gpuTime, imgin->ncols, imgin->nrows);

  //cudaEventDestroy(startGPU);
  //cudaEventDestroy(stopGPU);
  // ====================== GPU TIMING END ========================

  // Step 2: Create a temporary CPU output image of same size
  _KLT_FloatImage cpuOut = _KLTCreateFloatImage(imgin->ncols, imgin->nrows);

  // ====================== CPU TIMING START ======================
  struct timeval startCPU, endCPU;
  gettimeofday(&startCPU, NULL);
  // ====================== Run CPU ===============================
    float *restrict ptrrow = imgin->data;
    float *restrict ptrout = cpuOut->data;
    int radius = kernel.width / 2;
    int w = kernel.width;
    float *restrict ker = kernel.data;
    int ncols = imgin->ncols, nrows = imgin->nrows;
#pragma acc data copyin(ker[0:w], ptrrow[0:ncols*nrows]) copyout(ptrout[0:ncols*nrows])  //Data Construct and Clauses
{
   #pragma acc parallel loop gang vector       //Outer Loop Parallel Construct (distribute iterations among blocks and multiple threads in each block)
   #pragma acc cache(ptrrow[0:ncols*nrows])    //Hint for Shared memory usage, not much benefit in case of horizontal convolution
   for (int j = 0; j < nrows; j++) {

    // left border
   #pragma acc loop vector                     //Parallel execute it on threads of a gang
    for (int i = 0; i < radius; i++)
        ptrout[j*ncols + i] = 0.0f;

    // middle
    #pragma acc loop vector
    for (int i = radius; i < ncols - radius; i++) {
        float sum = 0.0f;
        int x = 0;
        for (int k = w-1; k >=0 ; k--) {
            int idx = (j*ncols + i) - radius + x;
            sum += ptrrow[idx] * ker[k];
            x++;
        }
        ptrout[j*ncols + i] = sum;
    }

    // right border
    #pragma acc loop vector
    for (int i = ncols - radius; i < ncols; i++)
        ptrout[j*ncols + i] = 0.0f;
}
}

  gettimeofday(&endCPU, NULL);
  double cpu_time_ms = (endCPU.tv_sec - startCPU.tv_sec) * 1000.0 +
                       (endCPU.tv_usec - startCPU.tv_usec) / 1000.0;

  total_cpu_compute_time += cpu_time_ms;

  //printf("[CPU HORIZ] Time: %.3f ms | Speedup: %.2fx\n", cpu_time_ms, cpu_time_ms / gpuTime);
  // ====================== CPU TIMING END ========================

  // Step 4: Compute absolute difference between CPU and GPU results
    float totalDiff = 0.0f;
    float maxDiff = 0.0f;
    int size = imgin->ncols * imgin->nrows;

    for (int idx = 0; idx < size; idx++) {
      float diff = fabs(cpuOut->data[idx] - imgout->data[idx]);
      totalDiff += diff;
      if (diff > maxDiff)
        maxDiff = diff;
    }

    float meanDiff = totalDiff / size;

    //printf("  Accuracy: Mean=%.2e, Max=%.2e\n", meanDiff, maxDiff);

    if (meanDiff > 1e-3) {
      //printf(" !!!  WARNING: High error detected!\n");
    }

  // Step 5: Free temporary CPU result image
  _KLTFreeFloatImage(cpuOut);
}

/*********************************************************************
 * _convolveImageVert - OpenACC Parallelized Version
 */
 #include <sys/time.h>

static void _convolveImageVert(_KLT_FloatImage imgin, ConvolutionKernel kernel,
                               _KLT_FloatImage imgout) {
    total_convolution_calls++;
    if (!image_size_recorded) {
        first_image_width = imgin->ncols;
        first_image_height = imgin->nrows;
        image_size_recorded = 1;
    }

    // ====================== GPU TIMING START ======================
    //cudaEvent_t startGPU, stopGPU;
    float gpuTime = 0.0f;
    //cudaEventCreate(&startGPU);
    //cudaEventCreate(&stopGPU);
    //cudaEventRecord(startGPU, 0);

    // Step 1: Run GPU convolution
    //_convolveImageVertUsingGPU(imgin, kernel.width, kernel.data, imgout);

    // Stop GPU timer
    //cudaEventRecord(stopGPU, 0);
    //cudaEventSynchronize(stopGPU);
    //cudaEventElapsedTime(&gpuTime, startGPU, stopGPU);

    //printf("\n[GPU VERT] Time: %.3f ms (Image: %dx%d)\n", gpuTime, imgin->ncols, imgin->nrows);

    //cudaEventDestroy(startGPU);
    //cudaEventDestroy(stopGPU);
    // ====================== GPU TIMING END ========================

    // Step 2: Create a temporary CPU output image for OpenACC version
    _KLT_FloatImage openaccOut = _KLTCreateFloatImage(imgin->ncols, imgin->nrows);

    // ====================== OpenACC TIMING START ======================
    struct timeval startACC, endACC;
    gettimeofday(&startACC, NULL);
    
    // ====================== OpenACC VERTICAL CONVOLUTION ======================
    float *ptrcol = imgin->data;
    float *ptrout = openaccOut->data;
    int radius = kernel.width / 2;
    int ncols = imgin->ncols, nrows = imgin->nrows;
    int w = kernel.width;
    float* ker = kernel.data;

    assert(kernel.width % 2 == 1);
    assert(imgin != openaccOut);
    assert(openaccOut->ncols >= imgin->ncols);
    assert(openaccOut->nrows >= imgin->nrows);

    // Copy kernel to GPU once
    #pragma acc data copyin(ker[0:w]) \
                    copyin(ptrcol[0:ncols*nrows]) \
                    copyout(ptrout[0:ncols*nrows])
    {
        // Process each column in parallel - perfect for OpenACC!
        #pragma acc parallel loop gang vector
        for (int i = 0; i < ncols; i++) {
            
            // Zero topmost rows for this column
            #pragma acc loop vector
            for (int j = 0; j < radius; j++) {
                ptrout[j * ncols + i] = 0.0f;
            }

            // Convolve middle rows with kernel for this column
            #pragma acc loop vector
            for (int j = radius; j < nrows - radius; j++) {
                float sum = 0.0f;
                
                // Vertical convolution - access memory with stride ncols
                #pragma acc loop reduction(+:sum)
                for (int k = 0; k < w; k++) {
                    int row_idx = j - radius + k;
                    int idx = row_idx * ncols + i;
                    sum += ptrcol[idx] * ker[k];
                }
                ptrout[j * ncols + i] = sum;
            }

            // Zero bottom most rows for this column
            #pragma acc loop vector
            for (int j = nrows - radius; j < nrows; j++) {
                ptrout[j * ncols + i] = 0.0f;
            }
        }
    }

    gettimeofday(&endACC, NULL);
    double acc_time_ms = (endACC.tv_sec - startACC.tv_sec) * 1000.0 +
                        (endACC.tv_usec - startACC.tv_usec) / 1000.0;

    //printf("[OpenACC VERT] Time: %.3f ms | Speedup vs GPU: %.2fx\n", acc_time_ms, gpuTime / acc_time_ms);
    // ====================== OpenACC TIMING END ========================

    // Step 4: Compute absolute difference between OpenACC and GPU results
    float totalDiff = 0.0f;
    float maxDiff = 0.0f;
    int size = imgin->ncols * imgin->nrows;

    for (int idx = 0; idx < size; idx++) {
        float diff = fabs(openaccOut->data[idx] - imgout->data[idx]);
        totalDiff += diff;
        if (diff > maxDiff) maxDiff = diff;
    }

    float meanDiff = totalDiff / size;
    //printf("  Accuracy: Mean=%.2e, Max=%.2e\n", meanDiff, maxDiff);

    if (meanDiff > 1e-3) {
        //printf("  !!! WARNING: High error detected!\n");
    }

    // Step 5: Free temporary OpenACC result image
    _KLTFreeFloatImage(openaccOut);
}
/*********************************************************************
 * _convolveSeparate
 */

static void _convolveSeparate(_KLT_FloatImage imgin,
                              ConvolutionKernel horiz_kernel,
                              ConvolutionKernel vert_kernel,
                              _KLT_FloatImage imgout) {
  /* Create temporary image */
  _KLT_FloatImage tmpimg;
  tmpimg = _KLTCreateFloatImage(imgin->ncols, imgin->nrows);

  /* Do convolution */
  _convolveImageHoriz(imgin, horiz_kernel, tmpimg);

  _convolveImageVert(tmpimg, vert_kernel, imgout);

  /* Free memory */
  _KLTFreeFloatImage(tmpimg);
}

/*********************************************************************
 * _KLTComputeGradients
 */

void _KLTComputeGradients(_KLT_FloatImage img, float sigma,
                          _KLT_FloatImage gradx, _KLT_FloatImage grady) {

  /* Output images must be large enough to hold result */
  assert(gradx->ncols >= img->ncols);
  assert(gradx->nrows >= img->nrows);
  assert(grady->ncols >= img->ncols);
  assert(grady->nrows >= img->nrows);

  /* Compute kernels, if necessary */
  if (fabs(sigma - sigma_last) > 0.05)
    _computeKernels(sigma, &gauss_kernel, &gaussderiv_kernel);

  _convolveSeparate(img, gaussderiv_kernel, gauss_kernel, gradx);
  _convolveSeparate(img, gauss_kernel, gaussderiv_kernel, grady);
}

/*********************************************************************
 * _KLTComputeSmoothedImage
 */

void _KLTComputeSmoothedImage(_KLT_FloatImage img, float sigma,
                              _KLT_FloatImage smooth) {
  /* Output image must be large enough to hold result */
  assert(smooth->ncols >= img->ncols);
  assert(smooth->nrows >= img->nrows);

  /* Compute kernel, if necessary; gauss_deriv is not used */
  if (fabs(sigma - sigma_last) > 0.05)
    _computeKernels(sigma, &gauss_kernel, &gaussderiv_kernel);

  _convolveSeparate(img, gauss_kernel, gauss_kernel, smooth);
}

/*********************************************************************
 * KLT_PrintPerformanceStats
 *
 * Call this at the end of your program to print cumulative stats
 */

void KLT_PrintPerformanceStats(double cpu_time) {
  printf("\n");
  printf("-----------------------------------------------------------\n");
  printf("              PERFORMANCE SUMMARY (D4 Report)              \n");
  printf("-----------------------------------------------------------\n");
  printf("First Image Size:              %d x %d\n", first_image_width,
         first_image_height);
  printf("Total Convolution Calls:       %d\n", total_convolution_calls);
  printf("-----------------------------------------------------------\n");
  printf("Total GPU Compute Time:        %.2f ms\n", total_gpu_compute_time);
  printf("Total CPU Compute Time:        %.2f ms\n", total_cpu_compute_time);
  printf("-----------------------------------------------------------\n");
  printf("Overall Speedup (GPU vs CPU):  %.2fx\n",
         total_cpu_compute_time / total_gpu_compute_time);
  printf("Time Saved:                    %.2f ms\n",
         total_cpu_compute_time - total_gpu_compute_time);
  printf("Percentage GPU is Faster:      %.1f%%\n",
         ((total_cpu_compute_time - total_gpu_compute_time) /
          total_cpu_compute_time) *
             100.0);
  printf("-----------------------------------------------------------\n");
  printf("\n");
}

/*********************************************************************
 * KLT_ResetPerformanceStats
 *
 * Call this at the start of your program to reset counters
 */

void KLT_ResetPerformanceStats(void) {
  total_gpu_compute_time = 0.0;
  total_cpu_compute_time = 0.0;
  total_memory_transfer_time = 0.0;
  total_convolution_calls = 0;
}

/*
for (j = 0; j < nrows; j++) {
      // Zero leftmost columns
      for (i = 0; i < radius; i++)
        *ptrout++ = 0.0;

      // Convolve middle columns with kernel
      for (; i < ncols - radius; i++) {
        ppp = ptrrow + i - radius;
        sum = 0.0;
        for (k = kernel.width - 1; k >= 0; k--)
          sum += *ppp++ * kernel.data[k];
        *ptrout++ = sum;
      }

      // Zero rightmost columns
      for (; i < ncols; i++)
        *ptrout++ = 0.0;
      ptrrow += ncols;
}
*/
