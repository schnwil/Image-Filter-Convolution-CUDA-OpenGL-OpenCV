/*****************************************************************
* 	Copyright (C) 2017
* 	Project : Matrix Convolution
* 	Author(s)  : Hemant Nigam, William Schneble
*       Description : Implements CPU kernels 
*****************************************************************/
#include <stdio.h>
#include <time.h>

#include "opencv2/opencv.hpp"
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "common.h"

// Forward declaration
// CPU kernel and launcher
void matrixConvCPUNaive_withoutPadding(unsigned char *dIn, int width, int height,int kernelW, int kernelH, unsigned char *dOut, const float *kernel);
void launchGaussianCPU(unsigned char *dIn, unsigned char *dOut, cv::Size size);
void launchSobelCPU(unsigned char *dIn, unsigned char *dOut, unsigned char *dGradX, unsigned char *dGradY,cv::Size size);
void sobelGradientCPU(unsigned char *gX, unsigned char *gY, int width, int height,unsigned char *dOut);


/// Gaussian CPU
void launchGaussianCPU(unsigned char *dIn, unsigned char *dOut, cv::Size size)
{
#ifdef _debug_
    struct timespec start, end; // variable to record time
    float elapsed;  // variable to record elapsed time`
    clock_gettime(CLOCK_MONOTONIC, &start);  // start time 
#endif
    matrixConvCPUNaive_withoutPadding(dIn, size.width, size.height,5,5, dOut,&gaussianKernel5x5[0]);
#ifdef _debug_
    clock_gettime(CLOCK_MONOTONIC, &end);  // end time 
    elapsed = NS_IN_SEC * (end.tv_sec - start.tv_sec) + end.tv_nsec - start.tv_nsec; 
  
    printf("Gaussian : Throughput in Megapixel per second : %.4f, Size : %d pixels, Elapsed time (in ms): %f\n",1.0e-6* (double)(size.height*size.width)/(elapsed*1.0e-9),size.height*size.width,elapsed*1.0e-6);
#endif
} 

/// Sobel CPU
void launchSobelCPU(unsigned char *dIn, unsigned char *dOut, unsigned char *dGradX, unsigned char *dGradY,cv::Size size)
{
#ifdef _debug_
    struct timespec start, end; // variable to record time
    float elapsed;  // variable to record elapsed time`
    clock_gettime(CLOCK_MONOTONIC, &start);  // start time 
#endif
    matrixConvCPUNaive_withoutPadding(dIn, size.width, size.height,3,3, dGradX, &sobelGradientX[0]);
    matrixConvCPUNaive_withoutPadding(dIn, size.width, size.height,3,3, dGradY, &sobelGradientY[0]);
    sobelGradientCPU(dGradX, dGradY, size.width,size.height,dOut);
#ifdef _debug_
    clock_gettime(CLOCK_MONOTONIC, &end);  // end time 
    elapsed = NS_IN_SEC * (end.tv_sec - start.tv_sec) + end.tv_nsec - start.tv_nsec; 
    
    printf("Sobel : Throughput in Megapixel per second : %.4f, Size : %d pixels, Elapsed time (in ms): %f\n",1.0e-6* (double)(size.height*size.width)/(elapsed*1.0e-9),size.height*size.width,elapsed*1.0e-6);
#endif
} 

/// CPU Naive kernel
void matrixConvCPUNaive_withoutPadding(unsigned char *dIn, int width, int height,int kernelW, int kernelH, unsigned char *dOut, const float *kernel)
{
    // Calculate radius along X and Y axis
    // We can also use one kernel variable instead - kernel radius
    int   kernelRadiusW = kernelW/2;
    int   kernelRadiusH = kernelH/2;
    
    for(int y = kernelRadiusH; y < height - kernelRadiusH; y++)
        for(int x = kernelRadiusW; x < width - kernelRadiusW; x++)
        {
            float accum = 0.0;
            for(int i = -kernelRadiusH; i <= kernelRadiusH; i++)  // Along Y axis
            {
                for(int j = -kernelRadiusW; j <= kernelRadiusW; j++) // Along X axis
                {
                    // calculate weight 
                    int jj = (j+kernelRadiusW);
                    int ii = (i+kernelRadiusH);
                    float w  = kernel[(ii * kernelW) + jj];
        
                    accum += w * float(dIn[((y+i) * width) + (x+j)]);
                }
            }
//            printf("index=%d\n",y*width+x);
            dOut[(y * width) + x] = (unsigned char) accum;
         }
}


/// Sobel gradient kernel CPU
void sobelGradientCPU(unsigned char *gX, unsigned char *gY, int width, int height,unsigned char *dOut)
{
    for(int y = 0; y < height; y++)
        for(int x = 0; x < width; x++)
        {
            int idx = y*width + x;
            float i = float(gX[idx]);
            float j = float(gY[idx]);
            dOut[idx] = (unsigned char) sqrt(i*i + j*j);
        }
}
