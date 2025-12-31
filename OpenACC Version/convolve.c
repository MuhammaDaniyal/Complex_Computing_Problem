/*********************************************************************
 * convolve.c - OpenACC Optimized Version (V4)
 *********************************************************************/

#include <assert.h>
#include <math.h>
#include <openacc.h>
#include <stdlib.h>
#include <sys/time.h>
#include <time.h>

#include "base.h"
#include "convolve.h"
#include "error.h"
#include "klt_util.h"

#define MAX_KERNEL_WIDTH 71

// =============== GLOBAL TIMING ACCUMULATORS ===============
static double total_gpu_compute_time = 0.0;
static int total_convolution_calls = 0;

typedef struct {
    int width;
    float data[MAX_KERNEL_WIDTH];
} ConvolutionKernel;

static ConvolutionKernel gauss_kernel;
static ConvolutionKernel gaussderiv_kernel;
static float sigma_last = -10.0;

/*********************************************************************
 * _KLTToFloatImage
 *********************************************************************/
void _KLTToFloatImage(KLT_PixelType *img, int ncols, int nrows,
                      _KLT_FloatImage floatimg) {
    KLT_PixelType *ptrend = img + ncols * nrows;
    float *ptrout = floatimg->data;

    assert(floatimg->ncols >= ncols);
    assert(floatimg->nrows >= nrows);

    floatimg->ncols = ncols;
    floatimg->nrows = nrows;

    while (img < ptrend)
        *ptrout++ = (float)*img++;
}

/*********************************************************************
 * _computeKernels
 *********************************************************************/
static void _computeKernels(float sigma, ConvolutionKernel *gauss,
                            ConvolutionKernel *gaussderiv) {
    const float factor = 0.01f;
    int i;

    assert(MAX_KERNEL_WIDTH % 2 == 1);
    assert(sigma >= 0.0);

    const int hw = MAX_KERNEL_WIDTH / 2;
    float max_gauss = 1.0f, max_gaussderiv = (float)(sigma * exp(-0.5f));

    for (i = -hw; i <= hw; i++) {
        gauss->data[i + hw] = (float)exp(-i * i / (2 * sigma * sigma));
        gaussderiv->data[i + hw] = -i * gauss->data[i + hw];
    }

    gauss->width = MAX_KERNEL_WIDTH;
    for (i = -hw; fabs(gauss->data[i + hw] / max_gauss) < factor;
         i++, gauss->width -= 2);
    
    gaussderiv->width = MAX_KERNEL_WIDTH;
    for (i = -hw; fabs(gaussderiv->data[i + hw] / max_gaussderiv) < factor;
         i++, gaussderiv->width -= 2);

    if (gauss->width == MAX_KERNEL_WIDTH ||
        gaussderiv->width == MAX_KERNEL_WIDTH)
        KLTError("(_computeKernels) MAX_KERNEL_WIDTH too small for sigma %f",
                 MAX_KERNEL_WIDTH, sigma);

    for (i = 0; i < gauss->width; i++)
        gauss->data[i] = gauss->data[i + (MAX_KERNEL_WIDTH - gauss->width) / 2];
    
    for (i = 0; i < gaussderiv->width; i++)
        gaussderiv->data[i] =
            gaussderiv->data[i + (MAX_KERNEL_WIDTH - gaussderiv->width) / 2];

    const int hw2 = gaussderiv->width / 2;
    float den = 0.0;
    for (i = 0; i < gauss->width; i++)
        den += gauss->data[i];
    for (i = 0; i < gauss->width; i++)
        gauss->data[i] /= den;

    den = 0.0;
    for (i = -hw2; i <= hw2; i++)
        den -= i * gaussderiv->data[i + hw2];
    for (i = -hw2; i <= hw2; i++)
        gaussderiv->data[i + hw2] /= den;

    sigma_last = sigma;
}

void _KLTGetKernelWidths(float sigma, int *gauss_width, int *gaussderiv_width) {
    _computeKernels(sigma, &gauss_kernel, &gaussderiv_kernel);
    *gauss_width = gauss_kernel.width;
    *gaussderiv_width = gaussderiv_kernel.width;
}

/*********************************************************************
 * _convolveImageHoriz - OpenACC Horizontal Convolution
 *********************************************************************/
static void _convolveImageHoriz(_KLT_FloatImage imgin, ConvolutionKernel kernel,
                                _KLT_FloatImage imgout) {
    total_convolution_calls++;

    struct timeval start, end;
    gettimeofday(&start, NULL);

    float *restrict ptrrow = imgin->data;
    float *restrict ptrout = imgout->data;
    int radius = kernel.width / 2;
    int w = kernel.width;
    float *restrict ker = kernel.data;
    int ncols = imgin->ncols, nrows = imgin->nrows;

    #pragma acc data copyin(ker[0:w], ptrrow[0:ncols*nrows]) \
                     copyout(ptrout[0:ncols*nrows])
    {
        #pragma acc parallel loop gang vector
        for (int j = 0; j < nrows; j++) {
            // Left border
            #pragma acc loop vector
            for (int i = 0; i < radius; i++)
                ptrout[j * ncols + i] = 0.0f;

            // Middle (FIXED: Reverse kernel indexing)
            #pragma acc loop vector
            for (int i = radius; i < ncols - radius; i++) {
                float sum = 0.0f;
                #pragma acc loop reduction(+:sum)
                for (int k = 0; k < w; k++) {
                    int idx = j * ncols + (i - radius + k);
                    sum += ptrrow[idx] * ker[w - 1 - k]; // ? REVERSED
                }
                ptrout[j * ncols + i] = sum;
            }

            // Right border
            #pragma acc loop vector
            for (int i = ncols - radius; i < ncols; i++)
                ptrout[j * ncols + i] = 0.0f;
        }
    }

    gettimeofday(&end, NULL);
    double time_ms = (end.tv_sec - start.tv_sec) * 1000.0 +
                     (end.tv_usec - start.tv_usec) / 1000.0;
    total_gpu_compute_time += time_ms;
}

/*********************************************************************
 * _convolveImageVert - OpenACC Vertical Convolution
 *********************************************************************/
static void _convolveImageVert(_KLT_FloatImage imgin, ConvolutionKernel kernel,
                               _KLT_FloatImage imgout) {
    total_convolution_calls++;

    struct timeval start, end;
    gettimeofday(&start, NULL);

    float *restrict ptrcol = imgin->data;
    float *restrict ptrout = imgout->data;
    int radius = kernel.width / 2;
    int ncols = imgin->ncols, nrows = imgin->nrows;
    int w = kernel.width;
    float *restrict ker = kernel.data;

    assert(kernel.width % 2 == 1);
    assert(imgout->ncols >= imgin->ncols);
    assert(imgout->nrows >= imgin->nrows);

    #pragma acc data copyin(ker[0:w], ptrcol[0:ncols*nrows]) \
                     copyout(ptrout[0:ncols*nrows])
    {
        #pragma acc parallel loop gang vector
        for (int i = 0; i < ncols; i++) {
            // Top border
            #pragma acc loop vector
            for (int j = 0; j < radius; j++) {
                ptrout[j * ncols + i] = 0.0f;
            }

            // Middle (FIXED: Reverse kernel indexing)
            #pragma acc loop vector
            for (int j = radius; j < nrows - radius; j++) {
                float sum = 0.0f;
                #pragma acc loop reduction(+:sum)
                for (int k = 0; k < w; k++) {
                    int row_idx = j - radius + k;
                    int idx = row_idx * ncols + i;
                    sum += ptrcol[idx] * ker[w - 1 - k]; // ? REVERSED
                }
                ptrout[j * ncols + i] = sum;
            }

            // Bottom border
            #pragma acc loop vector
            for (int j = nrows - radius; j < nrows; j++) {
                ptrout[j * ncols + i] = 0.0f;
            }
        }
    }

    gettimeofday(&end, NULL);
    double time_ms = (end.tv_sec - start.tv_sec) * 1000.0 +
                     (end.tv_usec - start.tv_usec) / 1000.0;
    total_gpu_compute_time += time_ms;
}

/*********************************************************************
 * _convolveSeparate
 *********************************************************************/
static void _convolveSeparate(_KLT_FloatImage imgin,
                              ConvolutionKernel horiz_kernel,
                              ConvolutionKernel vert_kernel,
                              _KLT_FloatImage imgout) {
    _KLT_FloatImage tmpimg = _KLTCreateFloatImage(imgin->ncols, imgin->nrows);
    _convolveImageHoriz(imgin, horiz_kernel, tmpimg);
    _convolveImageVert(tmpimg, vert_kernel, imgout);
    _KLTFreeFloatImage(tmpimg);
}

/*********************************************************************
 * _KLTComputeGradients
 *********************************************************************/
void _KLTComputeGradients(_KLT_FloatImage img, float sigma,
                          _KLT_FloatImage gradx, _KLT_FloatImage grady) {
    assert(gradx->ncols >= img->ncols);
    assert(gradx->nrows >= img->nrows);
    assert(grady->ncols >= img->ncols);
    assert(grady->nrows >= img->nrows);

    if (fabs(sigma - sigma_last) > 0.05)
        _computeKernels(sigma, &gauss_kernel, &gaussderiv_kernel);

    _convolveSeparate(img, gaussderiv_kernel, gauss_kernel, gradx);
    _convolveSeparate(img, gauss_kernel, gaussderiv_kernel, grady);
}

/*********************************************************************
 * _KLTComputeSmoothedImage
 *********************************************************************/
void _KLTComputeSmoothedImage(_KLT_FloatImage img, float sigma,
                              _KLT_FloatImage smooth) {
    assert(smooth->ncols >= img->ncols);
    assert(smooth->nrows >= img->nrows);

    if (fabs(sigma - sigma_last) > 0.05)
        _computeKernels(sigma, &gauss_kernel, &gaussderiv_kernel);

    _convolveSeparate(img, gauss_kernel, gauss_kernel, smooth);
}

/*********************************************************************
 * KLT_PrintPerformanceStats
 *********************************************************************/
void KLT_PrintPerformanceStats(double cpu_time) {
    printf("\n");
    printf("-----------------------------------------------------------\n");
    printf("           PERFORMANCE SUMMARY (OpenACC V4)               \n");
    printf("-----------------------------------------------------------\n");
    printf("Total Convolution Calls:       %d\n", total_convolution_calls);
    printf("Total OpenACC Compute Time:    %.2f ms\n", total_gpu_compute_time);
    printf("-----------------------------------------------------------\n\n");
}

void KLT_ResetPerformanceStats(void) {
    total_gpu_compute_time = 0.0;
    total_convolution_calls = 0;
}