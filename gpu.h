#ifndef __GPU_H__
#define __GPU_H__

#pragma once
#include <opencv2/core/core.hpp>
#include <cuda_runtime.h>
#include "common.h"

//Launcher
unsigned char* allocateBuffer(unsigned int size, unsigned char **dPtr);
void setConstantMemory(const void *src, ssize_t count, ssize_t offset);

void launchSobel_restrict(unsigned char *dIn, unsigned char *dOut, unsigned char *dGradX, unsigned char *dGradY, cv::Size size,ssize_t offsetX,ssize_t offsetY);
void launchSobel_float(unsigned char *dIn, unsigned char *dOut, unsigned char *dGradX, unsigned char *dGradY, cv::Size size,ssize_t offsetX,ssize_t offsetY);
void launchSobel_constantMemory(unsigned char *dIn, unsigned char *dOut, unsigned char *dGradX, unsigned char *dGradY, cv::Size size,ssize_t offsetX,ssize_t offsetY);
void launchSobelNaive_withoutPadding(unsigned char *dIn, unsigned char *dOut, unsigned char *dGradX, unsigned char *dGradY, cv::Size size,const float *d_X,const float *d_Y);
void launchSobelNaive_withPadding(unsigned char *dIn, unsigned char *dOut, unsigned char *dGradX, unsigned char *dGradY, cv::Size size,const float *d_X,const float *d_Y);
void launchSobelGradientKernel(int width, int height, unsigned char *gX, unsigned char *gY, unsigned char *dOut);

void launchGaussian_restrict(unsigned char *dIn, unsigned char *dOut, cv::Size size,ssize_t offset);
void launchGaussian_float(unsigned char *dIn, unsigned char *dOut, cv::Size size,ssize_t offset);
void launchGaussian_constantMemory(unsigned char *dIn, unsigned char *dOut, cv::Size size,ssize_t offset);
void launchGaussian_withoutPadding(unsigned char *dIn, unsigned char *dOut, cv::Size size,const float* kernel);
void launchSeparableKernel(unsigned char *d_input, cv::Size size, float alpha, ssize_t kOffset1, ssize_t kOffset2, int kDim, unsigned char *d_buffer, float *seperableBuffer);

//Kernel
__global__ void sobelGradientKernel(unsigned char *a, unsigned char *b, unsigned char *c);
__global__ void sobelGradientKernel_float(unsigned char *a, unsigned char *b, unsigned char *c);
__global__ void sobelGradientKernel_restrict(unsigned char *a, unsigned char *b, unsigned char *c);
__global__ void matrixConvGPU_float(unsigned char *dIn, int width, int height, int paddingX, int paddingY, ssize_t kOffset, int kernelW, int kernelH, unsigned char *dOut);
__global__ void matrixConvGPU_restrict(unsigned char *dIn, int width, int height, int paddingX, int paddingY, ssize_t kOffset, int kernelW, int kernelH, unsigned char *dOut);
__global__ void matrixConvGPU_constantMemory(unsigned char *dIn, int width, int height, int paddingX, int paddingY, ssize_t kOffset, int kernelW, int kernelH, unsigned char *dOut);
__global__ void matrixConvGPUNaive_withPadding(unsigned char *dIn, int width, int height, int paddingX, int paddingY, int kernelW, int kernelH, unsigned char *dOut, const float *kernel);
__global__ void matrixConvGPUNaive_withoutPadding(unsigned char *dIn, int width, int height, int kernelW, int kernelH, unsigned char *dOut, const float *kernel);
__global__ void separableKernel(unsigned char *d_input, int width, int height, bool phase1, float alpha, ssize_t kOffset, int kDim, unsigned char *d_output, float *seperableBuffer);

#endif
