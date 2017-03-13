#include "gpu.h"
#include "common.h"

__constant__ float constConvKernelMem[256];

void setConstantMemory(const void *src, ssize_t count, ssize_t offset) 
{
   cudaMemcpyToSymbol(constConvKernelMem, src, count, offset);
}

// Allocate buffer 
// Return ptr to shared mem
unsigned char* allocateBuffer(unsigned int size, unsigned char **dPtr)
{
   unsigned char *ptr = NULL;
   cudaSetDeviceFlags(cudaDeviceMapHost);
   cudaHostAlloc(&ptr, size, cudaHostAllocMapped);
   cudaHostGetDevicePointer(dPtr, ptr, 0);
   return ptr;
}

void launchGaussian_float(unsigned char *dIn, unsigned char *dOut, cv::Size size, ssize_t offset)
{
   dim3 blocksPerGrid(size.width / 16, size.height / 16);
   dim3 threadsPerBlock(16, 16);

   {
      matrixConvGPU_float << <blocksPerGrid, threadsPerBlock >> >(dIn, size.width, size.height, 0, 0, offset, 5, 5, dOut);
   }
   cudaDeviceSynchronize();
}

void launchGaussian_restrict(unsigned char *dIn, unsigned char *dOut, cv::Size size, ssize_t offset)
{
   dim3 blocksPerGrid(size.width / 16, size.height / 16);
   dim3 threadsPerBlock(16, 16);

   {
      matrixConvGPU_restrict << <blocksPerGrid, threadsPerBlock >> >(dIn, size.width, size.height, 0, 0, offset, 5, 5, dOut);
   }
   cudaDeviceSynchronize();
}

void launchGaussian_constantMemory(unsigned char *dIn, unsigned char *dOut, cv::Size size, ssize_t offset)
{
   dim3 blocksPerGrid(size.width / 16, size.height / 16);
   dim3 threadsPerBlock(16, 16);

   {
      matrixConvGPU_constantMemory << <blocksPerGrid, threadsPerBlock >> >(dIn, size.width, size.height, 0, 0, offset, 5, 5, dOut);
   }
   cudaDeviceSynchronize();
}

void launchGaussian_withoutPadding(unsigned char *dIn, unsigned char *dOut, cv::Size size, const float *kernel)
{
   dim3 blocksPerGrid(size.width / 16, size.height / 16);
   dim3 threadsPerBlock(16, 16);

   {
      matrixConvGPUNaive_withoutPadding << <blocksPerGrid, threadsPerBlock >> >(dIn, size.width, size.height, 5, 5, dOut, kernel);
   }
   cudaDeviceSynchronize();
}

void launchSobel_float(unsigned char *dIn, unsigned char *dOut, unsigned char *dGradX, unsigned char *dGradY, cv::Size size, ssize_t offsetX, ssize_t offsetY)
{
   dim3 blocksPerGrid(size.width / 16, size.height / 16);
   dim3 threadsPerBlock(16, 16);

   // pythagoran kernel launch paramters
   dim3 blocksPerGridP(size.width * size.height / 256);
   dim3 threadsPerBlockP(256, 1);

   {
      matrixConvGPU_float << <blocksPerGrid, threadsPerBlock >> >(dIn, size.width, size.height, 2, 2, offsetX, 3, 3, dGradX);
      matrixConvGPU_float << <blocksPerGrid, threadsPerBlock >> >(dIn, size.width, size.height, 2, 2, offsetY, 3, 3, dGradY);
      sobelGradientKernel_float << <blocksPerGridP, threadsPerBlockP >> >(dGradX, dGradY, dOut);
   }
   cudaDeviceSynchronize();
}

void launchSobel_restrict(unsigned char *dIn, unsigned char *dOut, unsigned char *dGradX, unsigned char *dGradY, cv::Size size, ssize_t offsetX, ssize_t offsetY)
{
   dim3 blocksPerGrid(size.width / 16, size.height / 16);
   dim3 threadsPerBlock(16, 16);

   // pythagoran kernel launch paramters
   dim3 blocksPerGridP(size.width * size.height / 256);
   dim3 threadsPerBlockP(256, 1);

   {
      matrixConvGPU_restrict << <blocksPerGrid, threadsPerBlock >> >(dIn, size.width, size.height, 2, 2, offsetX, 3, 3, dGradX);
      matrixConvGPU_restrict << <blocksPerGrid, threadsPerBlock >> >(dIn, size.width, size.height, 2, 2, offsetY, 3, 3, dGradY);
      sobelGradientKernel_restrict << <blocksPerGridP, threadsPerBlockP >> >(dGradX, dGradY, dOut);
   }
   cudaDeviceSynchronize();
}

void launchSobel_constantMemory(unsigned char *dIn, unsigned char *dOut, unsigned char *dGradX, unsigned char *dGradY, cv::Size size, ssize_t offsetX, ssize_t offsetY)
{
   dim3 blocksPerGrid(size.width / 16, size.height / 16);
   dim3 threadsPerBlock(16, 16);

   // pythagoran kernel launch paramters
   dim3 blocksPerGridP(size.width * size.height / 256);
   dim3 threadsPerBlockP(256, 1);

   {
      matrixConvGPU_constantMemory << <blocksPerGrid, threadsPerBlock >> >(dIn, size.width, size.height, 2, 2, offsetX, 3, 3, dGradX);
      matrixConvGPU_constantMemory << <blocksPerGrid, threadsPerBlock >> >(dIn, size.width, size.height, 2, 2, offsetY, 3, 3, dGradY);
      sobelGradientKernel << <blocksPerGridP, threadsPerBlockP >> >(dGradX, dGradY, dOut);
   }
   cudaDeviceSynchronize();
}

void launchSobelNaive_withoutPadding(unsigned char *dIn, unsigned char *dOut, unsigned char *dGradX, unsigned char *dGradY, cv::Size size, const float *d_X, const float *d_Y)
{
   dim3 blocksPerGrid(size.width / 16, size.height / 16);
   dim3 threadsPerBlock(16, 16);

   // Dimension for Sobel gradient kernel 
   dim3 blocksPerGridP(size.width * size.height / 256);
   dim3 threadsPerBlockP(256, 1);

   {
      matrixConvGPUNaive_withoutPadding << <blocksPerGrid, threadsPerBlock >> >(dIn, size.width, size.height, 3, 3, dGradX, d_X);
      matrixConvGPUNaive_withoutPadding << <blocksPerGrid, threadsPerBlock >> >(dIn, size.width, size.height, 3, 3, dGradY, d_Y);
      sobelGradientKernel << <blocksPerGridP, threadsPerBlockP >> >(dGradX, dGradY, dOut);
   }
   cudaDeviceSynchronize();
}

void launchSobelNaive_withPadding(unsigned char *dIn, unsigned char *dOut, unsigned char *dGradX, unsigned char *dGradY, cv::Size size, const float *d_X, const float *d_Y)
{
   dim3 blocksPerGrid(size.width / 16, size.height / 16);
   dim3 threadsPerBlock(16, 16);

   // Dimension for Sobel gradient kernel 
   dim3 blocksPerGridP(size.width * size.height / 256);
   dim3 threadsPerBlockP(256, 1);

   {
      matrixConvGPUNaive_withPadding << <blocksPerGrid, threadsPerBlock >> >(dIn, size.width, size.height, 2, 2, 3, 3, dGradX, d_X);
      matrixConvGPUNaive_withPadding << <blocksPerGrid, threadsPerBlock >> >(dIn, size.width, size.height, 2, 2, 3, 3, dGradY, d_Y);
      sobelGradientKernel << <blocksPerGridP, threadsPerBlockP >> >(dGradX, dGradY, dOut);
   }
   cudaDeviceSynchronize();
}

/**
Launch separable kernel. Call does both the row and col vector-matrix multiplication.
@param unsigned char *d_input          input array
@param cv::Size size                   size of input array
@param float alpha                     scalar
@param ssize_t kOffset1, kOffset2      offset in constant memory to row and col vectors
@param int kDim                        dimension size of vectors
@param unsigned char *d_buffer         output array
@param int *d_seperableBuffer temp storage for phase one sum with values > 255
**/
void launchSeparableKernel(unsigned char *d_input, cv::Size size, float alpha, ssize_t kOffset1, ssize_t kOffset2, int kDim, unsigned char *d_buffer, float *d_seperableBuffer) {
   dim3 blocks(size.width / 16, size.height / 16);
   dim3 threads(16, 16);

   separableKernel << <blocks, threads >> > (d_input, size.width, size.height, true, alpha, kOffset1, kDim, d_buffer, d_seperableBuffer);
   cudaDeviceSynchronize();
   separableKernel << <blocks, threads >> > (d_input, size.width, size.height, false, alpha, kOffset2, kDim, d_buffer, d_seperableBuffer);
}

void launchSobelGradientKernel(int width, int height, unsigned char *gX, unsigned char *gY, unsigned char *dOut) {
   sobelGradientKernel << <dim3((width * height) / 256), dim3(256) >> >(gX, gY, dOut);
}

// Used for Sobel edge detection
// Calculate gradient value from gradientX and gradientY  
// Calculate G = sqrt(Gx^2 * Gy^2)
__global__ void sobelGradientKernel(unsigned char *gX, unsigned char *gY, unsigned char *dOut)
{
   int idx = (blockIdx.x * blockDim.x) + threadIdx.x;

   float x = float(gX[idx]);
   float y = float(gY[idx]);

   dOut[idx] = (unsigned char)sqrtf(x*x + y*y);
}

__global__ void sobelGradientKernel_float(unsigned char *gX, unsigned char *gY, unsigned char *dOut)
{
   int idx = (int)(((float)blockIdx.x * (float)blockDim.x) + (float)threadIdx.x);

   float x = float(gX[idx]);
   float y = float(gY[idx]);

   dOut[idx] = (unsigned char)sqrtf(x*x + y*y);
}

__global__ void sobelGradientKernel_restrict(unsigned char* __restrict__ gX, unsigned char* __restrict__ gY, unsigned char *dOut)
{
   int idx = (int)(((float)blockIdx.x * (float)blockDim.x) + (float)threadIdx.x);

   float x = float(gX[idx]);
   float y = float(gY[idx]);

   dOut[idx] = (unsigned char)sqrtf(x*x + y*y);
}

//naive without padding
__global__ void matrixConvGPUNaive_withoutPadding(unsigned char *dIn, int width, int height, int kernelW, int kernelH, unsigned char *dOut, const float *kernel)
{
   // Pixel location 
   int x = blockIdx.x * blockDim.x + threadIdx.x;
   int y = blockIdx.y * blockDim.y + threadIdx.y;

   float accum = 0.0;
   // Calculate radius along X and Y axis
   // We can also use one kernel variable instead - kernel radius
   int   kernelRadiusW = kernelW / 2;
   int   kernelRadiusH = kernelH / 2;

   // Determine pixels to operate 
   if (x >= kernelRadiusW && y >= kernelRadiusH &&
      x < (blockDim.x * gridDim.x) - kernelRadiusW &&
      y < (blockDim.y * gridDim.y) - kernelRadiusH)
   {
      for (int i = -kernelRadiusH; i <= kernelRadiusH; i++)  // Along Y axis
      {
         for (int j = -kernelRadiusW; j <= kernelRadiusW; j++) // Along X axis
         {
            // calculate weight 
            int jj = (j + kernelRadiusW);
            int ii = (i + kernelRadiusH);
            float w = kernel[(ii * kernelW) + jj];

            accum += w * float(dIn[((y + i) * width) + (x + j)]);
         }
      }
   }

   dOut[(y * width) + x] = (unsigned char)accum;
}

//Naive with padding
__global__ void matrixConvGPUNaive_withPadding(unsigned char *dIn, int width, int height, int paddingX, int paddingY, int kernelW, int kernelH, unsigned char *dOut, const float *kernel)
{
   // Pixel location 
   int x = blockIdx.x * blockDim.x + threadIdx.x;
   int y = blockIdx.y * blockDim.y + threadIdx.y;

   float accum = 0.0;
   // Calculate radius along X and Y axis
   // We can also use one kernel variable instead - kernel radius
   int   kernelRadiusW = kernelW / 2;
   int   kernelRadiusH = kernelH / 2;

   // Determine pixels to operate 
   if (x >= (kernelRadiusW + paddingX) && y >= (kernelRadiusH + paddingY) &&
      x < ((blockDim.x * gridDim.x) - kernelRadiusW - paddingX) &&
      y < ((blockDim.y * gridDim.y) - kernelRadiusH - paddingY))
   {
      for (int i = -kernelRadiusH; i <= kernelRadiusH; i++)  // Along Y axis
      {
         for (int j = -kernelRadiusW; j <= kernelRadiusW; j++) // Along X axis
         {
            // calculate weight 
            int jj = (j + kernelRadiusW);
            int ii = (i + kernelRadiusH);
            float w = kernel[(ii * kernelW) + jj];

            accum += w * float(dIn[((y + i) * width) + (x + j)]);
         }
      }
   }

   dOut[(y * width) + x] = (unsigned char)accum;
}

//Constant memory
__global__ void matrixConvGPU_constantMemory(unsigned char *dIn, int width, int height, int paddingX, int paddingY, ssize_t kernelOffset, int kernelW, int kernelH, unsigned char *dOut)
{
   // Calculate our pixel's location
   int x = (blockIdx.x * blockDim.x) + threadIdx.x;
   int y = (blockIdx.y * blockDim.y) + threadIdx.y;

   // Calculate radius along X and Y axis
   // We can also use one kernel variable instead - kernel radius
   float accum = 0.0;
   int   kernelRadiusW = kernelW / 2;
   int   kernelRadiusH = kernelH / 2;

   // Determine pixels to operate 
   if (x >= (kernelRadiusW + paddingX) && y >= (kernelRadiusH + paddingY) &&
      x < ((blockDim.x * gridDim.x) - kernelRadiusW - paddingX) &&
      y < ((blockDim.y * gridDim.y) - kernelRadiusH - paddingY))
   {
      for (int i = -kernelRadiusH; i <= kernelRadiusH; i++) // Along Y axis
      {
         for (int j = -kernelRadiusW; j <= kernelRadiusW; j++) //Along X axis
         {
            // Sample the weight for this location
            int jj = (j + kernelRadiusW);
            int ii = (i + kernelRadiusH);
            float w = constConvKernelMem[(ii * kernelW) + jj + kernelOffset]; //kernel from constant memory

            accum += w * float(dIn[((y + i) * width) + (x + j)]);
         }
      }
   }

   dOut[(y * width) + x] = (unsigned char)accum;
}

__global__ void matrixConvGPU_float(unsigned char *dIn, int width, int height, int paddingX, int paddingY, ssize_t kernelOffset, int kernelW, int kernelH, unsigned char *dOut)
{
   // Calculate our pixel's location
   float x = ((float)blockIdx.x * (float)blockDim.x) + (float)threadIdx.x;
   float y = ((float)blockIdx.y * (float)blockDim.y) + (float)threadIdx.y;

   // Calculate radius along X and Y axis
   // We can also use one kernel variable instead - kernel radius
   float accum = 0.0;
   int   kernelRadiusW = kernelW / 2;
   int   kernelRadiusH = kernelH / 2;

   // Determine pixels to operate 
   if (x >= ((float)kernelRadiusW + (float)paddingX) && y >= ((float)kernelRadiusH + (float)paddingY) &&
      x < (((float)blockDim.x * (float)gridDim.x) - (float)kernelRadiusW - (float)paddingX) &&
      y < (((float)blockDim.y * (float)gridDim.y) - (float)kernelRadiusH - (float)paddingY))
   {
      for (int i = -kernelRadiusH; i <= kernelRadiusH; i++) // Along Y axis
      {
         for (int j = -kernelRadiusW; j <= kernelRadiusW; j++) //Along X axis
         {
            // Sample the weight for this location
            float jj = ((float)j + (float)kernelRadiusW);
            float ii = ((float)i + (float)kernelRadiusH);
            float w = constConvKernelMem[(int)((ii * (float)kernelW) + jj + (float)kernelOffset)]; //kernel from constant memory

            accum += w * float(dIn[(int)(((y + (float)i) * (float)width) + (x + (float)j))]);
         }
      }
   }

   dOut[(int)((y * (float)width) + x)] = (unsigned char)accum;
}

__global__ void matrixConvGPU_restrict(unsigned char* __restrict__ dIn, int width, int height, int paddingX, int paddingY, ssize_t kernelOffset, int kernelW, int kernelH, unsigned char* __restrict__ dOut)
{
   // Calculate our pixel's location
   float x = ((float)blockIdx.x * (float)blockDim.x) + (float)threadIdx.x;
   float y = ((float)blockIdx.y * (float)blockDim.y) + (float)threadIdx.y;

   // Calculate radius along X and Y axis
   // We can also use one kernel variable instead - kernel radius
   float accum = 0.0;
   int   kernelRadiusW = kernelW / 2;
   int   kernelRadiusH = kernelH / 2;

   // Determine pixels to operate 
   if (x >= ((float)kernelRadiusW + (float)paddingX) && y >= ((float)kernelRadiusH + (float)paddingY) &&
      x < (((float)blockDim.x * (float)gridDim.x) - (float)kernelRadiusW - (float)paddingX) &&
      y < (((float)blockDim.y * (float)gridDim.y) - (float)kernelRadiusH - (float)paddingY))
   {
      for (int i = -kernelRadiusH; i <= kernelRadiusH; i++) // Along Y axis
      {
         for (int j = -kernelRadiusW; j <= kernelRadiusW; j++) //Along X axis
         {
            // Sample the weight for this location
            float jj = ((float)j + (float)kernelRadiusW);
            float ii = ((float)i + (float)kernelRadiusH);
            float w = constConvKernelMem[(int)((ii * (float)kernelW) + jj + (float)kernelOffset)]; //kernel from constant memory

            accum += w * float(dIn[(int)(((y + (float)i) * (float)width) + (x + (float)j))]);
         }
      }
   }

   dOut[(int)((y * (float)width) + x)] = (unsigned char)accum;
}

/**
Separable Kernel does matrix-vector mutliplication. Is meant to be called twice, once for the row
vector and once for the col vector. First phase needs temp storage for output which has values
greater than 255 - client needs to allocate integer array to store these values. Second phase does
the final sum and then multiples this result with alpha and thresholds the value which is then passed
into the output array.
@param unsigned char *d_input          input array
@param int width, height               width and height of the input array
@param bool phase1                     if true then output stored in d_seperableBuffer
@param float alpha                     scalar
@param ssize_t kOffset                 offset for constant memory where vector stored
@param int kDim                        dimension size of filter to use
@param unsigned char *d_output         output array
@param int *d_separableBuffer temp storage for phase one sum which is > 255
**/
__global__ void separableKernel(unsigned char *d_input, int width, int height, bool phase1, float alpha, ssize_t kOffset, int kDim, unsigned char *d_output, float *d_separableBuffer) {
   int tx = blockIdx.x * blockDim.x + threadIdx.x;
   int ty = blockIdx.y * blockDim.y + threadIdx.y;
   int rad = kDim / 2;
   kOffset += rad;

   //get rid of apron/boundary threads
   if (phase1) {
      if (ty < rad || ty > height - rad)
         return;
   }
   else {
      if (tx < rad || tx > width - rad)
         return;
   }

   //compute values depending on if this is row or col vector
   float accum = 0;
   for (int i = -rad; i <= rad; i++) {
      if (phase1) {
         accum += (float)d_input[tx + (ty + i)*width] * constConvKernelMem[kOffset + i];
      }
      else {
         accum += d_separableBuffer[tx + i + ty*width] * constConvKernelMem[kOffset + i];
      }
   }

   //update output, if phase1 then we need to store values which are >255 in temp storage for next phase
   if (phase1) {
      d_separableBuffer[tx + ty*width] = accum;
   }
   else {
      accum *= alpha;
      accum = accum > 255 ? 255 : accum; //threshold the pixel
      accum = accum < 0 ? 0 : accum;
      d_output[tx + ty*width] = (unsigned char)accum;
   }
}