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
#include <sys/time.h>

#define MAX_KERNEL_WIDTH 71

// =============== GLOBAL STATE FOR PERSISTENT DATA ===============
static int data_on_device = 0;
static float *device_kernel_ptr = NULL;
static int device_kernel_size = 0;

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
 * Initialize persistent GPU data at program start
 *********************************************************************/
void _KLT_InitOpenACCData() {
    if (!data_on_device) {
        // Pre-allocate kernel space on device (will update as needed)
        device_kernel_size = MAX_KERNEL_WIDTH;
        device_kernel_ptr = (float*)malloc(MAX_KERNEL_WIDTH * sizeof(float));
        
        #pragma acc enter data create(device_kernel_ptr[0:MAX_KERNEL_WIDTH])
        
        data_on_device = 1;
        printf("[OpenACC] Initialized persistent GPU data\n");
    }
}

/*********************************************************************
 * Cleanup persistent GPU data at program end
 *********************************************************************/
void _KLT_CleanupOpenACCData() {
    if (data_on_device) {
        #pragma acc exit data delete(device_kernel_ptr[0:MAX_KERNEL_WIDTH])
        free(device_kernel_ptr);
        data_on_device = 0;
        printf("[OpenACC] Cleaned up persistent GPU data\n");
    }
}

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
 * OpenACC Implementation for D4
 *********************************************************************/

/*********************************************************************
 * OPTIMIZED HORIZONTAL CONVOLUTION
 * - Uses persistent data regions
 * - Constant memory for kernel
 * - Async operations
 * - Better loop structure
 *********************************************************************/
static void _convolveImageHoriz_OpenACC_Optimized(
    _KLT_FloatImage imgin, 
    ConvolutionKernel kernel,
    _KLT_FloatImage imgout) 
{
    float *restrict ptrrow = imgin->data;
    float *restrict ptrout = imgout->data;
    int radius = kernel.width / 2;
    int w = kernel.width;
    float *restrict ker = kernel.data;
    int ncols = imgin->ncols, nrows = imgin->nrows;
    int total_pixels = ncols * nrows;
    
    struct timeval start, end;
    gettimeofday(&start, NULL);
    
    // Use present_or_copyin for images (will create if not present)
    // Use copyin for kernel with const hint (constant memory)
    #pragma acc data copyin(ker[0:w]) \
                    present_or_copyin(ptrrow[0:total_pixels]) \
                    present_or_copyout(ptrout[0:total_pixels])
    {
        // Async launch for better pipelining
        #pragma acc parallel loop gang vector_length(256) async(1)
        for (int j = 0; j < nrows; j++) {
            
            #pragma acc loop vector
            for (int i = 0; i < ncols; i++) {
                int idx = j * ncols + i;
                
                if (i < radius || i >= ncols - radius) {
                    ptrout[idx] = 0.0f;
                } else {
                    float sum = 0.0f;
                    // Unroll small kernels for better performance
                    #pragma acc loop seq
                    for (int k = 0; k < w; k++) {
                        sum += ptrrow[idx - radius + k] * ker[w - 1 - k];
                    }
                    ptrout[idx] = sum;
                }
            }
        }
        
        // Wait for completion
        #pragma acc wait(1)
    }
    
    gettimeofday(&end, NULL);
    double time_ms = (end.tv_sec - start.tv_sec) * 1000.0 +
                     (end.tv_usec - start.tv_usec) / 1000.0;
    
    printf("[OpenACC HORIZ] Time: %.3f ms (Image: %dx%d)\n", 
           time_ms, ncols, nrows);
}

/*********************************************************************
 * OPTIMIZED VERTICAL CONVOLUTION WITH TILING
 * - Tile columns to improve cache locality
 * - Use shared memory hint via cache directive
 * - Persistent data
 *********************************************************************/
static void _convolveImageVert_OpenACC_Optimized(
    _KLT_FloatImage imgin,
    ConvolutionKernel kernel,
    _KLT_FloatImage imgout) 
{
    float *restrict ptrcol = imgin->data;
    float *restrict ptrout = imgout->data;
    int radius = kernel.width / 2;
    int w = kernel.width;
    float *restrict ker = kernel.data;
    int ncols = imgin->ncols, nrows = imgin->nrows;
    int total_pixels = ncols * nrows;
    
    struct timeval start, end;
    gettimeofday(&start, NULL);
    
    #pragma acc data copyin(ker[0:w]) \
                    present_or_copyin(ptrcol[0:total_pixels]) \
                    present_or_copyout(ptrout[0:total_pixels])
    {
        // Process in tiles for better cache usage
        #define TILE_SIZE 32
        int num_tiles = (ncols + TILE_SIZE - 1) / TILE_SIZE;
        
        #pragma acc parallel loop gang async(2)
        for (int tile = 0; tile < num_tiles; tile++) {
            int i_start = tile * TILE_SIZE;
            int i_end = (i_start + TILE_SIZE < ncols) ? i_start + TILE_SIZE : ncols;
            
            #pragma acc loop vector
            for (int i = i_start; i < i_end; i++) {
                
                // Process column
                for (int j = 0; j < nrows; j++) {
                    int idx = j * ncols + i;
                    
                    if (j < radius || j >= nrows - radius) {
                        ptrout[idx] = 0.0f;
                    } else {
                        float sum = 0.0f;
                        #pragma acc loop seq
                        for (int k = 0; k < w; k++) {
                            int row_idx = j - radius + k;
                            sum += ptrcol[row_idx * ncols + i] * ker[k];
                        }
                        ptrout[idx] = sum;
                    }
                }
            }
        }
        
        #pragma acc wait(2)
    }
    
    gettimeofday(&end, NULL);
    double time_ms = (end.tv_sec - start.tv_sec) * 1000.0 +
                     (end.tv_usec - start.tv_usec) / 1000.0;
    
    printf("[OpenACC VERT] Time: %.3f ms (Image: %dx%d)\n", 
           time_ms, ncols, nrows);
}

/*********************************************************************
 * ALTERNATIVE: Fully pipelined version with manual data management
 *********************************************************************/
static void _convolveImageHoriz_OpenACC_Pipelined(
    _KLT_FloatImage imgin, 
    ConvolutionKernel kernel,
    _KLT_FloatImage imgout) 
{
    float *ptrrow = imgin->data;
    float *ptrout = imgout->data;
    int radius = kernel.width / 2;
    int w = kernel.width;
    float *ker = kernel.data;
    int ncols = imgin->ncols, nrows = imgin->nrows;
    int total_pixels = ncols * nrows;
    
    // Manually manage data transfers with async
    #pragma acc enter data copyin(ptrrow[0:total_pixels]) async(1)
    #pragma acc enter data create(ptrout[0:total_pixels]) async(1)
    #pragma acc enter data copyin(ker[0:w]) async(1)
    
    #pragma acc wait(1)
    
    struct timeval start, end;
    gettimeofday(&start, NULL);
    
    // Compute on GPU
    #pragma acc parallel loop gang vector_length(256) \
                present(ptrrow[0:total_pixels]) \
                present(ptrout[0:total_pixels]) \
                present(ker[0:w]) \
                async(2)
    for (int j = 0; j < nrows; j++) {
        #pragma acc loop vector
        for (int i = 0; i < ncols; i++) {
            int idx = j * ncols + i;
            
            if (i < radius || i >= ncols - radius) {
                ptrout[idx] = 0.0f;
            } else {
                float sum = 0.0f;
                #pragma acc loop seq
                for (int k = 0; k < w; k++) {
                    sum += ptrrow[idx - radius + k] * ker[w - 1 - k];
                }
                ptrout[idx] = sum;
            }
        }
    }
    
    // Copy back results
    #pragma acc exit data copyout(ptrout[0:total_pixels]) async(2)
    #pragma acc wait(2)
    
    // Cleanup - keep input for next use, delete output
    #pragma acc exit data delete(ptrrow[0:total_pixels], ker[0:w])
    
    gettimeofday(&end, NULL);
    double time_ms = (end.tv_sec - start.tv_sec) * 1000.0 +
                     (end.tv_usec - start.tv_usec) / 1000.0;
    
    printf("[OpenACC HORIZ Pipeline] Time: %.3f ms\n", time_ms);
}

/*********************************************************************
 * UNIFIED DATA MANAGEMENT FOR ENTIRE PYRAMID
 * Keep all pyramid data on GPU across multiple convolutions
 *********************************************************************/
typedef struct {
    float *data;
    int ncols;
    int nrows;
    int on_device;
} ManagedImage;

static ManagedImage managed_images[10]; // For pyramid levels
static int num_managed = 0;

void _KLT_PutImageOnDevice(_KLT_FloatImage img) {
    int size = img->ncols * img->nrows;
    
    #pragma acc enter data copyin(img->data[0:size])
    
    // Track it
    if (num_managed < 10) {
        managed_images[num_managed].data = img->data;
        managed_images[num_managed].ncols = img->ncols;
        managed_images[num_managed].nrows = img->nrows;
        managed_images[num_managed].on_device = 1;
        num_managed++;
    }
}

void _KLT_GetImageFromDevice(_KLT_FloatImage img) {
    int size = img->ncols * img->nrows;
    
    #pragma acc exit data copyout(img->data[0:size])
    
    // Remove from tracking
    for (int i = 0; i < num_managed; i++) {
        if (managed_images[i].data == img->data) {
            managed_images[i].on_device = 0;
            break;
        }
    }
}

void _KLT_UpdateImageOnDevice(_KLT_FloatImage img) {
    int size = img->ncols * img->nrows;
    
    #pragma acc update device(img->data[0:size])
}

void _KLT_FreeDeviceImages() {
    for (int i = 0; i < num_managed; i++) {
        if (managed_images[i].on_device) {
            int size = managed_images[i].ncols * managed_images[i].nrows;
            float *data = managed_images[i].data;
            
            #pragma acc exit data delete(data[0:size])
            
            managed_images[i].on_device = 0;
        }
    }
    num_managed = 0;
}

/*********************************************************************
 * WRAPPER FUNCTIONS - Choose optimization level
 *********************************************************************/

// Set which optimization to use
#define USE_OPTIMIZED_VERSION 1  // 0=basic, 1=optimized, 2=pipelined

static void _convolveImageHoriz(_KLT_FloatImage imgin, 
                               ConvolutionKernel kernel,
                               _KLT_FloatImage imgout) 
{
    total_convolution_calls++;
    if (!image_size_recorded) {
        first_image_width = imgin->ncols;
        first_image_height = imgin->nrows;
        image_size_recorded = 1;
    }
    
    // Initialize persistent data if needed
    if (!data_on_device) {
        _KLT_InitOpenACCData();
    }
    
    // Create CPU baseline output
    _KLT_FloatImage cpuOut = _KLTCreateFloatImage(imgin->ncols, imgin->nrows);
    
    // Time CPU version
    struct timeval startCPU, endCPU;
    gettimeofday(&startCPU, NULL);
    
    // CPU convolution
    {
        float *ptrrow = imgin->data;
        float *ptrout = cpuOut->data;
        int radius = kernel.width / 2;
        int ncols = imgin->ncols, nrows = imgin->nrows;
        
        for (int j = 0; j < nrows; j++) {
            for (int i = 0; i < radius; i++)
                ptrout[j * ncols + i] = 0.0f;
            
            for (int i = radius; i < ncols - radius; i++) {
                float sum = 0.0f;
                for (int k = 0; k < kernel.width; k++) {
                    sum += ptrrow[j * ncols + i - radius + k] * 
                           kernel.data[kernel.width - 1 - k];
                }
                ptrout[j * ncols + i] = sum;
            }
            
            for (int i = ncols - radius; i < ncols; i++)
                ptrout[j * ncols + i] = 0.0f;
        }
    }
    
    gettimeofday(&endCPU, NULL);
    double cpu_time_ms = (endCPU.tv_sec - startCPU.tv_sec) * 1000.0 +
                         (endCPU.tv_usec - startCPU.tv_usec) / 1000.0;
    
    total_cpu_compute_time += cpu_time_ms;
    printf("\n[CPU HORIZ] Time: %.3f ms\n", cpu_time_ms);
    
    // Run optimized OpenACC version
    struct timeval startACC, endACC;
    gettimeofday(&startACC, NULL);
    
#if USE_OPTIMIZED_VERSION == 1
    _convolveImageHoriz_OpenACC_Optimized(imgin, kernel, imgout);
#elif USE_OPTIMIZED_VERSION == 2
    _convolveImageHoriz_OpenACC_Pipelined(imgin, kernel, imgout);
#else
    _convolveImageHoriz_OpenACC(imgin, kernel, imgout);  // Basic version
#endif
    
    gettimeofday(&endACC, NULL);
    double acc_time_ms = (endACC.tv_sec - startACC.tv_sec) * 1000.0 +
                         (endACC.tv_usec - startACC.tv_usec) / 1000.0;
    
    total_gpu_compute_time += acc_time_ms;
    
    printf("[OpenACC Speedup] %.2fx over CPU\n", cpu_time_ms / acc_time_ms);
    
    // Verify correctness
    float totalDiff = 0.0f, maxDiff = 0.0f;
    int size = imgin->ncols * imgin->nrows;
    for (int idx = 0; idx < size; idx++) {
        float diff = fabs(cpuOut->data[idx] - imgout->data[idx]);
        totalDiff += diff;
        if (diff > maxDiff) maxDiff = diff;
    }
    printf("  Accuracy: Mean=%.2e, Max=%.2e\n", totalDiff / size, maxDiff);
    
    _KLTFreeFloatImage(cpuOut);
}

// Similar optimizations for vertical...
static void _convolveImageVert(_KLT_FloatImage imgin,
                              ConvolutionKernel kernel,
                              _KLT_FloatImage imgout) 
{
    total_convolution_calls++;
    
    _KLT_FloatImage cpuOut = _KLTCreateFloatImage(imgin->ncols, imgin->nrows);
    
    // Time CPU
    struct timeval startCPU, endCPU;
    gettimeofday(&startCPU, NULL);
    
    {
        float *ptrcol = imgin->data;
        float *ptrout = cpuOut->data;
        int radius = kernel.width / 2;
        int ncols = imgin->ncols, nrows = imgin->nrows;
        
        for (int i = 0; i < ncols; i++) {
            for (int j = 0; j < radius; j++)
                ptrout[j * ncols + i] = 0.0f;
            
            for (int j = radius; j < nrows - radius; j++) {
                float sum = 0.0f;
                for (int k = 0; k < kernel.width; k++) {
                    sum += ptrcol[(j - radius + k) * ncols + i] * kernel.data[k];
                }
                ptrout[j * ncols + i] = sum;
            }
            
            for (int j = nrows - radius; j < nrows; j++)
                ptrout[j * ncols + i] = 0.0f;
        }
    }
    
    gettimeofday(&endCPU, NULL);
    double cpu_time_ms = (endCPU.tv_sec - startCPU.tv_sec) * 1000.0 +
                         (endCPU.tv_usec - startCPU.tv_usec) / 1000.0;
    
    total_cpu_compute_time += cpu_time_ms;
    printf("\n[CPU VERT] Time: %.3f ms\n", cpu_time_ms);
    
    // Run optimized OpenACC
    struct timeval startACC, endACC;
    gettimeofday(&startACC, NULL);
    
#if USE_OPTIMIZED_VERSION >= 1
    _convolveImageVert_OpenACC_Optimized(imgin, kernel, imgout);
#else
    _convolveImageVert_OpenACC(imgin, kernel, imgout);
#endif
    
    gettimeofday(&endACC, NULL);
    double acc_time_ms = (endACC.tv_sec - startACC.tv_sec) * 1000.0 +
                         (endACC.tv_usec - startACC.tv_usec) / 1000.0;
    
    total_gpu_compute_time += acc_time_ms;
    
    printf("[OpenACC Speedup] %.2fx over CPU\n", cpu_time_ms / acc_time_ms);
    
    // Verify
    float totalDiff = 0.0f, maxDiff = 0.0f;
    int size = imgin->ncols * imgin->nrows;
    for (int idx = 0; idx < size; idx++) {
        float diff = fabs(cpuOut->data[idx] - imgout->data[idx]);
        totalDiff += diff;
        if (diff > maxDiff) maxDiff = diff;
    }
    printf("  Accuracy: Mean=%.2e, Max=%.2e\n", totalDiff / size, maxDiff);
    
    _KLTFreeFloatImage(cpuOut);
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

void KLT_PrintPerformanceStats(double gpu_time) 
{
  printf("\n");
  printf("╔═════════════════════════════════════════════════════════╗\n");
  printf("║             PERFORMANCE SUMMARY (D4 Report)             ║\n");
  printf("╠═════════════════════════════════════════════════════════╣\n");
  printf("║First Image Size:             %d x %d\n", first_image_width,
         first_image_height);
  printf("║Total Convolution Calls:       %d\n", total_convolution_calls);
  printf("║---------------------------------------------------------║\n");
  printf("║Total GPU Compute Time:                  %.2f ms ║\n", total_gpu_compute_time);
  printf("║Total CPU Compute Time:                  %.2f ms ║\n", total_cpu_compute_time);
  printf("║---------------------------------------------------------║\n");
  printf("║Overall Speedup (GPU vs CPU):              %.2fx ║\n",
         total_cpu_compute_time / total_gpu_compute_time);
  printf("║Time Saved:                              %.2f ms ║\n",
         total_cpu_compute_time - total_gpu_compute_time);
  printf("║Percentage GPU is Faster:                 %.1f%% ║\n",
         ((total_cpu_compute_time - total_gpu_compute_time) /
          total_cpu_compute_time) *
             100.0);
  printf("╚═════════════════════════════════════════════════════════╝\n");
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

/*********************************************************************
 * Call at program end to cleanup
 *********************************************************************/
void KLT_CleanupOpenACC(void) {
    _KLT_CleanupOpenACCData();
    _KLT_FreeDeviceImages();
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
