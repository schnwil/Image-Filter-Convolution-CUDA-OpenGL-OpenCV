#include <string>
#include <stdio.h>
#include <time.h>

#include "opencv2/opencv.hpp"
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <cuda_runtime.h>

#include "common.h"
#include "gpu.h"
#include "gputimer.h"
#include "key_bindings.h"
#include "helper_funcs.h"

#define MAX_FPS 60.0

// Create the cuda event timers 
gpuTimer timer;

using namespace std;

int main (int argc, char** argv)
{
    FILE *flog = fopen("./log.txt", "w+");
    gpuTimer t1;
    unsigned int frameCounter=0;
    float *d_X,*d_Y,*d_gaussianKernel5x5;

    /// Pass video file as input
    // For e.g. if camera device is at /dev/video1 - pass 1
    // You can pass video file as well instead of webcam stream
    const char *videoFile = "C:/Users/Alex/Videos/The Witcher 3/test.mp4";
    cv::VideoCapture camera(videoFile);
    //cv::VideoCapture camera(1);
    
    cv::Mat frame;
    if(!camera.isOpened()) 
    {
        printf("Error .... campera not opened\n");;
        return -1;
    }
    
    // Open window for each kernel 
    cv::namedWindow("Video Feed");

    // Calculate kernel offset in contant memory
    const ssize_t gaussianKernel5x5Offset = 0;
    const ssize_t sobelKernelGradOffsetX = sizeof(gaussianKernel5x5) / sizeof(float);
    const ssize_t sobelKernelGradOffsetY = sizeof(sobelGradientX) / sizeof(float) + sobelKernelGradOffsetX;
    const ssize_t gaussianSeparableOffset = sizeof(sobelGradientY) / sizeof(float) + sobelKernelGradOffsetY;
    const ssize_t sobel101Offset = sizeof(gaussianSeparableKernel) / sizeof(float) + gaussianSeparableOffset;
    const ssize_t sobel121Offset = sizeof(sobelSeparable101) / sizeof(float) + sobel101Offset;

    //copy to constant memory
    ssize_t offset = 0;
    setConstantMemory(gaussianKernel5x5, sizeof(gaussianKernel5x5), offset); offset += sizeof(gaussianKernel5x5);
    setConstantMemory(sobelGradientX, sizeof(sobelGradientX), offset); offset += sizeof(sobelGradientX);
    setConstantMemory(sobelGradientY, sizeof(sobelGradientY), offset); offset += sizeof(sobelGradientY);
    setConstantMemory(gaussianSeparableKernel, sizeof(gaussianSeparableKernel), offset); offset += sizeof(gaussianSeparableKernel);
    setConstantMemory(sobelSeparable101, sizeof(sobelSeparable101), offset); offset += sizeof(sobelSeparable101);
    setConstantMemory(sobelSeparable121, sizeof(sobelSeparable121), offset); offset += sizeof(sobelSeparable121);

    // Create matrix to hold original and processed image 
    camera >> frame;
    unsigned char *d_pixelDataInput, *d_pixelDataOutput, *d_pixelBuffer;
    float *d_separableBuffer;
    
    cudaMalloc((void **) &d_gaussianKernel5x5, sizeof(gaussianKernel5x5));
    cudaMalloc((void **) &d_X, sizeof(sobelGradientX));
    cudaMalloc((void **) &d_Y, sizeof(sobelGradientY));
    
    cudaMemcpy(d_gaussianKernel5x5, &gaussianKernel5x5[0], sizeof(gaussianKernel5x5), cudaMemcpyHostToDevice);
    cudaMemcpy(d_X, &sobelGradientX[0], sizeof(sobelGradientX), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Y, &sobelGradientY[0], sizeof(sobelGradientY), cudaMemcpyHostToDevice);

    cv::Mat inputMat     (frame.size(), CV_8U, allocateBuffer(frame.size().width * frame.size().height, &d_pixelDataInput));
    cv::Mat outputMat    (frame.size(), CV_8U, allocateBuffer(frame.size().width * frame.size().height, &d_pixelDataOutput));
    cv::Mat bufferMat    (frame.size(), CV_8U, allocateBuffer(frame.size().width * frame.size().height, &d_pixelBuffer));

    cv::Mat inputMatCPU   (frame.size(), CV_8U);
    cv::Mat outputMatCPU  (frame.size(), CV_8U);
    cv::Mat bufferMatCPU  (frame.size(), CV_8U);
    // Create buffer to hold sobel gradients - XandY 
    unsigned char *sobelBufferX, *sobelBufferY;
    cudaMalloc(&sobelBufferX, frame.size().width * frame.size().height);
    cudaMalloc(&sobelBufferY, frame.size().width * frame.size().height);
    cudaMalloc((void**)&d_separableBuffer, frame.size().width * frame.size().height * sizeof(float));

    // Create buffer to hold sobel gradients - XandY 
    unsigned char *sobelBufferXCPU, *sobelBufferYCPU;
    sobelBufferXCPU = (unsigned char*)malloc(frame.size().width * frame.size().height);
    sobelBufferYCPU = (unsigned char*)malloc(frame.size().width * frame.size().height);
    
    //key codes to switch between filters
    unsigned int key_pressed = NO_FILTER;
    double tms = 0.0;
    int prev_key = 0;
    struct timespec start, end; // variable to record cpu time
    
    // Run loop to capture images from camera or loop over single image 
    while(key_pressed != ESCAPE)
    {
        int key = cv::waitKey(1);
        key_pressed = key == -1 ? key_pressed : key;
        string kernel_t = "";

        // special key functions, capture image
        switch(key_pressed) {
        case PAUSE:
           continue;
        case RESUME:
           key_pressed = prev_key;
        }
        prev_key = key_pressed;

        camera >> frame;
        
        // Convert frame to gray scale for further filter operation
	// Remove color channels, simplify convolution operation
        if(key_pressed == SOBEL_NAIVE_CPU || key_pressed == GAUSSIAN_NAIVE_CPU)
            cv::cvtColor(frame, inputMatCPU, CV_BGR2GRAY);
        else
	    cv::cvtColor(frame, inputMat, CV_BGR2GRAY);
       
        switch (key_pressed) {
        case NO_FILTER:
        default:
           outputMat = inputMat;
           kernel_t = "No Filter";
           break;
        case GAUSSIAN_FILTER:
           t1.start(); // timer for overall metrics
           launchGaussian_withoutPadding(d_pixelDataInput, d_pixelBuffer, frame.size(), d_gaussianKernel5x5);
           t1.stop();
           tms = t1.elapsed();
           outputMat = bufferMat;
           kernel_t = "Guassian";
           break;
        case SOBEL_FILTER:
           t1.start(); // timer for overall metrics
           launchGaussian_constantMemory(d_pixelDataInput, d_pixelDataOutput, frame.size(), gaussianKernel5x5Offset);
           launchSobel_constantMemory(d_pixelDataOutput, d_pixelBuffer, sobelBufferX, sobelBufferY, frame.size(), sobelKernelGradOffsetX, sobelKernelGradOffsetY);
           t1.stop();
           tms = t1.elapsed();
           outputMat = bufferMat;
           kernel_t = "Sobel";
           break;
        case SOBEL_NAIVE_FILTER:
           t1.start(); // timer for overall metrics
           launchGaussian_withoutPadding(d_pixelDataInput, d_pixelDataOutput, frame.size(),d_gaussianKernel5x5);
           launchSobelNaive_withoutPadding(d_pixelDataOutput, d_pixelBuffer, sobelBufferX, sobelBufferY, frame.size(), d_X, d_Y);
           t1.stop();
           tms = t1.elapsed();
           outputMat = bufferMat;
           kernel_t = "Sobel Naive";
           break;
        case SOBEL_NAIVE_PADDED_FILTER:
           t1.start(); // timer for overall metrics
           launchGaussian_withoutPadding(d_pixelDataInput, d_pixelDataOutput, frame.size(), d_gaussianKernel5x5);
           launchSobelNaive_withPadding(d_pixelDataOutput, d_pixelBuffer, sobelBufferX, sobelBufferY, frame.size(), d_X, d_Y);
           t1.stop();
           tms = t1.elapsed();
           outputMat = bufferMat;
           kernel_t = "Sobel Naive Pad";
           break;
        case SOBEL_NAIVE_CPU:
           clock_gettime(CLOCK_MONOTONIC, &start);  // start time 
           launchGaussianCPU(inputMatCPU.data, outputMatCPU.data, frame.size());
           launchSobelCPU(outputMatCPU.data, bufferMatCPU.data, sobelBufferXCPU, sobelBufferYCPU, frame.size());
           clock_gettime(CLOCK_MONOTONIC, &end);  // end time 
           tms = (NS_IN_SEC * (end.tv_sec - start.tv_sec) + end.tv_nsec - start.tv_nsec)*1.0e-6; 
           outputMatCPU = bufferMatCPU;
           kernel_t = "Sobel Naive CPU";
           break;
        case GAUSSIAN_NAIVE_CPU:
           clock_gettime(CLOCK_MONOTONIC, &start);  // start time 
           launchGaussianCPU(inputMatCPU.data, outputMatCPU.data, frame.size());
           clock_gettime(CLOCK_MONOTONIC, &end);  // end time 
           tms = (NS_IN_SEC * (end.tv_sec - start.tv_sec) + end.tv_nsec - start.tv_nsec)*1.0e-6; 
           kernel_t = "Gaussian CPU";
           break;
        case SOBEL_FILTER_FLOAT:
           t1.start(); // timer for overall metrics
           launchGaussian_float(d_pixelDataInput, d_pixelDataOutput, frame.size(), gaussianKernel5x5Offset);
           launchSobel_float(d_pixelDataOutput, d_pixelBuffer, sobelBufferX, sobelBufferY, frame.size(), sobelKernelGradOffsetX, sobelKernelGradOffsetY);
           t1.stop();
           tms = t1.elapsed();
           outputMat = bufferMat;
           kernel_t = "Sobel Float";
           break;
        case SOBEL_FILTER_RESTRICT:
           t1.start(); // timer for overall metrics
           launchGaussian_restrict(d_pixelDataInput, d_pixelDataOutput, frame.size(), gaussianKernel5x5Offset);
           launchSobel_restrict(d_pixelDataOutput, d_pixelBuffer, sobelBufferX, sobelBufferY, frame.size(), sobelKernelGradOffsetX, sobelKernelGradOffsetY);
           t1.stop();
           tms = t1.elapsed();
           outputMat = bufferMat;
           kernel_t = "Sobel Restrictive";
           break;
        case SEPARABLE_GAUSSIAN:
           t1.start(); // timer for overall metrics
           launchSeparableKernel(d_pixelDataInput, frame.size(), 1.f / 256.f, gaussianSeparableOffset, gaussianSeparableOffset, 5, d_pixelBuffer, d_separableBuffer);
           t1.stop();
           tms = t1.elapsed();
           outputMat = bufferMat;
           kernel_t = "Gaussian Separable";
           break;
        case SEPARABLE_SOBEL:
           t1.start(); // timer for overall metrics
           launchSeparableKernel(d_pixelDataInput, frame.size(), 1.f / 256.f, gaussianSeparableOffset, gaussianSeparableOffset, 5, d_pixelDataOutput, d_separableBuffer);
           launchSeparableKernel(d_pixelDataOutput, frame.size(), 1.f, sobel101Offset, sobel121Offset, 3, sobelBufferY, d_separableBuffer);
           launchSeparableKernel(d_pixelDataOutput, frame.size(), 1.f, sobel121Offset, sobel101Offset, 3, sobelBufferX, d_separableBuffer);
          
           launchSobelGradientKernel(frame.size().width, frame.size().height, sobelBufferX, sobelBufferY, d_pixelBuffer);
           t1.stop();

           tms = t1.elapsed();
           outputMat = bufferMat;
           kernel_t = "Sobel Separable";
           break;
        }

        /**printf("Overall : Throughput in Megapixel per second : %.4f, Size : %d pixels, Elapsed time (in ms): %f\n",
           1.0e-6* (double)(frame.size().height*frame.size().width)/(tms*0.001),frame.size().height*frame.size().width,tms); **/
        
	     //create metric string
        frameCounter++;
        float fps = 1000.f / tms; //fps = fps > MAX_FPS ? MAX_FPS : fps;
        double mps = 1.0e-6* (double)(frame.size().height*frame.size().width) / (tms*0.001);
        string metricString = getMetricString(frameCounter, fps, mps, kernel_t, tms);
        fprintf(flog, "%s \n", metricString.c_str());

        //update display
        if(key_pressed == SOBEL_NAIVE_CPU || key_pressed == GAUSSIAN_NAIVE_CPU)
        {
            cv::putText(outputMatCPU, metricString, cvPoint(30, 30), CV_FONT_NORMAL, 1, 255, 2, CV_AA, false);
            cv::imshow("Video Feed", outputMatCPU);
        }
        else
        {
            cv::putText(outputMat, metricString, cvPoint(30, 30), CV_FONT_NORMAL, 1, 255, 2, CV_AA, false);
            cv::imshow("Video Feed", outputMat);
        }
    }
    
    // Deallocate memory
    cudaFreeHost(inputMat.data);
    cudaFreeHost(outputMat.data);
    cudaFree(sobelBufferX);
    cudaFree(sobelBufferY);
    cudaFree(d_X);
    cudaFree(d_Y);
    cudaFree(d_gaussianKernel5x5);
        
    // Deallocate host memory
    free(sobelBufferXCPU);
    free(sobelBufferYCPU);

    return 0;
}