__global__ static void _convolveImageHorizGPU(float *kernel,float *ptrrow, int radius, int width, float *ptrout, int ncols, int nrows){
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (j < nrows && i < ncols) {
		int idx = j * ncols + i;
		if (i < radius)
			ptrout[idx] = 0.0;
		if (i >= radius && i < ncols - radius){
			float sum = 0.0;
			int x = 0;
			for (int k = width-1; k >= 0; k--) {
				sum += ptrrow[idx - radius + x] * kernel[k];
				x++;
			}
			ptrout[idx] = sum;
		}
		if (i >= ncols - radius)
			ptrout[idx] = 0.0;
	}
}

static void _convolveImageHorizUsingGPU(_KLT_FloatImage imgin, ConvolutionKernel kernel, _KLT_FloatImage imgout) {
        float *ptrrow = imgin->data;
        register float *ptrout = imgout->data;
        register int radius = kernel.width / 2;
        register int ncols = imgin->ncols, nrows = imgin->nrows;
        
        /* Kernel width must be odd */
        assert(kernel.width % 2 == 1);

        /* Must read from and write to different images */
        assert(imgin != imgout);

        /* Output image must be large enough to hold result */
        assert(imgout->ncols >= imgin->ncols);
        assert(imgout->nrows >= imgin->nrows);

	float *ptrrow_device;
        register float *ptrout_device;
	float *kerneldata_device;
	int size_of_imgin = sizeof(float) * ncols * nrows;
	int size_of_imgout = sizeof(float) * imgout->ncols * imgout->nrows;

	cudaMalloc((void**)&ptrrow_device, size_of_imgin);
	cudaMalloc((void**)&ptrout_device, size_of_imgout);
	cudaMalloc((void**)&kerneldata_device, MAX_KERNEL_WIDTH);

	cudaMemcpy(ptrrow_device, ptrrow, size_of_imgin, cudaMemcpyHostToDevice);
	cudaMemcpy(ptrout_device, ptrout, size_of_imgout, cudaMemcpyHostToDevice);
	cudaMemcpy(kerneldata_device, kernel.data, MAX_KERNEL_WIDTH, cudaMemcpyHostToDevice);

	dim3 block(32, 32, 1);
	dim3 grid((ncols + block.x - 1) / block.x, (nrows + block.y - 1) / block.y, 1);
	_convolveImageHorizGPU <<<grid,block>>> (kerneldata_device, ptrrow_device, radius, kernel.width, ptrout_device, ncols, nrows);

	cudaMemcpy(ptrout, ptrout_device, size_of_imgout, cudaMemcpyDeviceToHost);

	cudaFree(ptrrow_device);
	cudaFree(ptrout_device);
	cudaFree(kerneldata_device);
}
