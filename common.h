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
    -1.f, 0.f, 1.f,
    -2.f, 0.f, 2.f,
    -1.f, 0.f, 1.f,
};

// Sobel kernel Y gradient
const float sobelGradientY[9] =
{
    1.f,  2.f,  1.f,
    0.f,  0.f,  0.f,
   -1.f, -2.f, -1.f,
};

#ifdef _WIN32
#include <Windows.h>
#define BILLION                             (1E9)
#define CLOCK_MONOTONIC 1

static BOOL g_first_time = 1;
static LARGE_INTEGER g_counts_per_sec;

static int clock_gettime(int dummy, struct timespec *ct)
{
   LARGE_INTEGER count;

   if (g_first_time)
   {
      g_first_time = 0;

      if (0 == QueryPerformanceFrequency(&g_counts_per_sec))
      {
         g_counts_per_sec.QuadPart = 0;
      }
   }

   if ((NULL == ct) || (g_counts_per_sec.QuadPart <= 0) ||
      (0 == QueryPerformanceCounter(&count)))
   {
      return -1;
   }

   ct->tv_sec = count.QuadPart / g_counts_per_sec.QuadPart;
   ct->tv_nsec = ((count.QuadPart % g_counts_per_sec.QuadPart) * BILLION) / g_counts_per_sec.QuadPart;

   return 0;
};
#endif

#endif
