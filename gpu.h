#ifndef __GPU_H__
#define __GPU_H__

#pragma once
#include <opencv2/core/core.hpp>
//Launcher
unsigned char* allocateBuffer(unsigned int size, unsigned char **dPtr);
void launchSobel_constantMemory(unsigned char *dIn, unsigned char *dOut, unsigned char *dGradX, unsigned char *dGradY, cv::Size size,ssize_t offsetX,ssize_t offsetY);
void launchSobelNaive_withoutPadding(unsigned char *dIn, unsigned char *dOut, unsigned char *dGradX, unsigned char *dGradY, cv::Size size,const float *d_X,const float *d_Y);
void launchSobelNaive_withPadding(unsigned char *dIn, unsigned char *dOut, unsigned char *dGradX, unsigned char *dGradY, cv::Size size,const float *d_X,const float *d_Y);
void launchGaussian_constantMemory(unsigned char *dIn, unsigned char *dOut, cv::Size size,ssize_t offset);
void launchGaussian_withoutPadding(unsigned char *dIn, unsigned char *dOut, cv::Size size,const float* kernel);

//Kernel
__global__ void sobelGradientKernel(unsigned char *a, unsigned char *b, unsigned char *c);
__global__ void matrixConvGPU_constantMemory(unsigned char *dIn, int width, int height, int paddingX, int paddingY, ssize_t kOffset, int kernelW, int kernelH, unsigned char *dOut);
__global__ void matrixConvGPUNaive_withPadding(unsigned char *dIn, int width, int height, int paddingX, int paddingY, int kernelW, int kernelH, unsigned char *dOut, const float *kernel);
__global__ void matrixConvGPUNaive_withoutPadding(unsigned char *dIn, int width, int height, int kernelW, int kernelH, unsigned char *dOut, const float *kernel);

#endif
