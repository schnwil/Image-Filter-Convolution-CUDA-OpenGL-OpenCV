#ifndef __COMMON_H__
#define __COMMON_H__

#if defined(_MSC_VER)
#include <BaseTsd.h>
typedef SSIZE_T ssize_t;
#endif

#pragma once
#include <opencv2/core/core.hpp>

#define NS_IN_SEC 1000000000L
extern void launchGaussianCPU(unsigned char *dIn, unsigned char *dOut, cv::Size size);
extern void launchSobelCPU(unsigned char *dIn, unsigned char *dOut, unsigned char *dGradX, unsigned char *dGradY,cv::Size size);
// Gaussian kernel 5X5 
// Based on opencv gaussian kernel output for radius 5
const float gaussianKernel5x5[25] = 
{
    2.f/159.f,  4.f/159.f,  5.f/159.f,  4.f/159.f, 2.f/159.f,   
    4.f/159.f,  9.f/159.f, 12.f/159.f,  9.f/159.f, 4.f/159.f,   
    5.f/159.f, 12.f/159.f, 15.f/159.f, 12.f/159.f, 5.f/159.f,   
    4.f/159.f,  9.f/159.f, 12.f/159.f,  9.f/159.f, 4.f/159.f,   
    2.f/159.f,  4.f/159.f,  5.f/159.f,  4.f/159.f, 2.f/159.f,   
};

// Sobel kernel X gradient 
const float sobelGradientX[9] =
{
    1.f, 0.f, -1.f,
    2.f, 0.f, -2.f,
    1.f, 0.f, -1.f,
};

// Sobel kernel Y gradient
const float sobelGradientY[9] =
{
    1.f,  2.f,  1.f,
    0.f,  0.f,  0.f,
   -1.f, -2.f, -1.f,
};

const float gaussianSeparableKernel[5] =
{
   1.f, 4.f, 7.f, 4.f, 1.f,
};

const float sobelSeparable101[3] = {
   1.f, 0.f, -1.f,
};

const float sobelSeparable121[3] = {
   1.f, 2.f, 1.f,
};

#endif
