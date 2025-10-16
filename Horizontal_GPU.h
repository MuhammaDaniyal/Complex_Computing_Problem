#ifndef HORIZONTAL_GPU_H
#define HORIZONTAL_GPU_H
#include <stdio.h>
#include <assert.h>
#include <math.h>
#include <stdlib.h>   /* malloc(), realloc() */
#include <cuda_runtime.h>

/* Our includes */
#include "base.h"
#include "error.h"
#include "convolve.h"
#include "klt_util.h"

void _convolveImageHorizUsingGPU(_KLT_FloatImage imgin, int width, float* kerneldata, _KLT_FloatImage imgout);

void _convolveImageVertUsingGPU(_KLT_FloatImage imgin, int width, float* kerneldata, _KLT_FloatImage imgout);

#endif
