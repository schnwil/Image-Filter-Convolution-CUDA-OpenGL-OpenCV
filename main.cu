#include <string>
#include <stdio.h>
#include <time.h>

#include "opencv2/opencv.hpp"
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "common.h"
#include "gpu.h"
#include "gputimer.h"
#include "key_bindings.h"
#include "helper_funcs.h"

#define MAX_FPS 60.0
#define TILE_W 16
#define GAUSSIAN_KERNEL_RADIUS 2
#define SOBEL_KERNEL_RADIUS 1

__constant__ float constConvKernelMem[256];
// Create the cuda event timers 
gpuTimer timer;

using namespace std;

int main (int argc, char** argv)
{
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
    cudaMemcpyToSymbol(constConvKernelMem, gaussianKernel5x5, sizeof(gaussianKernel5x5), offset); offset += sizeof(gaussianKernel5x5);
    cudaMemcpyToSymbol(constConvKernelMem, sobelGradientX, sizeof(sobelGradientX), offset); offset += sizeof(sobelGradientX);
    cudaMemcpyToSymbol(constConvKernelMem, sobelGradientY, sizeof(sobelGradientY), offset); offset += sizeof(sobelGradientY);
    cudaMemcpyToSymbol(constConvKernelMem, gaussianSeparableKernel, sizeof(gaussianSeparableKernel), offset); offset += sizeof(gaussianSeparableKernel);
    cudaMemcpyToSymbol(constConvKernelMem, sobelSeparable101, sizeof(sobelSeparable101), offset); offset += sizeof(sobelSeparable101);
    cudaMemcpyToSymbol(constConvKernelMem, sobelSeparable121, sizeof(sobelSeparable121), offset); offset += sizeof(sobelSeparable121);
 
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
          
           sobelGradientKernel << <dim3(frame.size().width * frame.size().height / 256), dim3(256)>> >(sobelBufferX, sobelBufferY, d_pixelBuffer);
           t1.stop();

           tms = t1.elapsed();
           outputMat = bufferMat;
           kernel_t = "Sobel Separable";
           break;
        case SOBEL_FILTER_SHARED:
           t1.start(); // timer for overall metrics
           launchGaussian_sharedMem(d_pixelDataInput, d_pixelDataOutput, frame.size(), gaussianKernel5x5Offset);
           launchSobel_sharedMem(d_pixelDataOutput, d_pixelBuffer, sobelBufferX, sobelBufferY, frame.size(), sobelKernelGradOffsetX, sobelKernelGradOffsetY);
           t1.stop();
           tms = t1.elapsed();
           outputMat = bufferMat;
           kernel_t = "Sobel Shared Mem";
           break;
        }

        /**printf("Overall : Throughput in Megapixel per second : %.4f, Size : %d pixels, Elapsed time (in ms): %f\n",
           1.0e-6* (double)(frame.size().height*frame.size().width)/(tms*0.001),frame.size().height*frame.size().width,tms); **/
        
	     //create metric string
        frameCounter++;
        float fps = 1000.f / tms; //fps = fps > MAX_FPS ? MAX_FPS : fps;
        double mps = 1.0e-6* (double)(frame.size().height*frame.size().width) / (tms*0.001);
        vector<string> metricString = getMetricString(frameCounter, fps, mps, kernel_t, tms);
        assert(metricString.size() == 3); // Make sure vector consist of 3 metric string
        //printf("Frame #:%d FPS:%2.3f MPS: %.4f Kernel Type %s Kernel Time (ms): %.4f\n", 
        //   frameCounter, fps, mps, kernel_t, tms);

        //update display
        if(key_pressed == SOBEL_NAIVE_CPU || key_pressed == GAUSSIAN_NAIVE_CPU)
        {
            cv::putText(outputMatCPU, metricString[0], cvPoint(30, 30), CV_FONT_NORMAL, 0.5, 255, 1, CV_AA, false);
            cv::putText(outputMatCPU, metricString[1], cvPoint(30, 50), CV_FONT_NORMAL, 0.5, 255, 1, CV_AA, false);
            cv::putText(outputMatCPU, metricString[2], cvPoint(30, 70), CV_FONT_NORMAL, 0.5, 255, 1, CV_AA, false);
            cv::imshow("Video Feed", outputMatCPU);
        }
        else
        {
            cv::putText(outputMat, metricString[0], cvPoint(30, 30), CV_FONT_NORMAL, 0.5, 255, 1, CV_AA, false);
            cv::putText(outputMat, metricString[1], cvPoint(30, 50), CV_FONT_NORMAL, 0.5, 255, 1, CV_AA, false);
            cv::putText(outputMat, metricString[2], cvPoint(30, 70), CV_FONT_NORMAL, 0.5, 255, 1, CV_AA, false);
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

void launchGaussian_float(unsigned char *dIn, unsigned char *dOut, cv::Size size,ssize_t offset)
{
    dim3 blocksPerGrid(size.width / 16, size.height / 16);
    dim3 threadsPerBlock(16, 16);
    
    timer.start();
    {
         matrixConvGPU_float <<<blocksPerGrid,threadsPerBlock>>>(dIn,size.width, size.height, 0, 0, offset, 5, 5, dOut);
    }
    timer.stop();
    cudaThreadSynchronize();
    double tms = timer.elapsed(); 
    //printf("Gaussian : Throughput in Megapixel per second : %.4f, Size : %d pixels, Elapsed time (in ms): %f\n",1.0e-6* (double)(size.height*size.width)/(tms*0.001),size.height*size.width,tms);
}

void launchGaussian_restrict(unsigned char *dIn, unsigned char *dOut, cv::Size size,ssize_t offset)
{
    dim3 blocksPerGrid(size.width / 16, size.height / 16);
    dim3 threadsPerBlock(16, 16);
    
    timer.start();
    {
         matrixConvGPU_restrict <<<blocksPerGrid,threadsPerBlock>>>(dIn,size.width, size.height, 0, 0, offset, 5, 5, dOut);
    }
    timer.stop();
    cudaThreadSynchronize();
    double tms = timer.elapsed(); 
    //printf("Gaussian : Throughput in Megapixel per second : %.4f, Size : %d pixels, Elapsed time (in ms): %f\n",1.0e-6* (double)(size.height*size.width)/(tms*0.001),size.height*size.width,tms);
}

void launchGaussian_sharedMem(unsigned char *dIn, unsigned char *dOut, cv::Size size,ssize_t offset)
{
    dim3 blocksPerGrid(size.width / 16, size.height / 16);
    dim3 threadsPerBlock(16, 16);
    
    timer.start();
    {
         matrixConvGPU_sharedMem <TILE_W,GAUSSIAN_KERNEL_RADIUS,TILE_W+2*GAUSSIAN_KERNEL_RADIUS> <<<blocksPerGrid,threadsPerBlock>>>(dIn,size.width, size.height, offset, 5, 5, dOut);
    }
    timer.stop();
    cudaThreadSynchronize();
    double tms = timer.elapsed(); 
    //printf("Gaussian : Throughput in Megapixel per second : %.4f, Size : %d pixels, Elapsed time (in ms): %f\n",1.0e-6* (double)(size.height*size.width)/(tms*0.001),size.height*size.width,tms);
}

void launchGaussian_constantMemory(unsigned char *dIn, unsigned char *dOut, cv::Size size,ssize_t offset)
{
    dim3 blocksPerGrid(size.width / 16, size.height / 16);
    dim3 threadsPerBlock(16, 16);
    
    timer.start();
    {
         matrixConvGPU_constantMemory <<<blocksPerGrid,threadsPerBlock>>>(dIn,size.width, size.height, 0, 0, offset, 5, 5, dOut);
    }
    timer.stop();
    cudaThreadSynchronize();
    double tms = timer.elapsed(); 
    //printf("Gaussian : Throughput in Megapixel per second : %.4f, Size : %d pixels, Elapsed time (in ms): %f\n",1.0e-6* (double)(size.height*size.width)/(tms*0.001),size.height*size.width,tms);
}

void launchGaussian_withoutPadding(unsigned char *dIn, unsigned char *dOut, cv::Size size, const float *kernel)
{
    dim3 blocksPerGrid(size.width / 16, size.height / 16);
    dim3 threadsPerBlock(16, 16);
    
    timer.start();
    {
         matrixConvGPUNaive_withoutPadding <<<blocksPerGrid,threadsPerBlock>>>(dIn,size.width, size.height, 5, 5, dOut, kernel);
    }
    timer.stop();
    cudaThreadSynchronize();
    double tms = timer.elapsed(); 
    //printf("Gaussian : Throughput in Megapixel per second : %.4f, Size : %d pixels, Elapsed time (in ms): %f\n",1.0e-6* (double)(size.height*size.width)/(tms*0.001),size.height*size.width,tms);
}

void launchSobel_float(unsigned char *dIn, unsigned char *dOut, unsigned char *dGradX, unsigned char *dGradY, cv::Size size,ssize_t offsetX,ssize_t offsetY)
{
    dim3 blocksPerGrid(size.width / 16, size.height / 16);
    dim3 threadsPerBlock(16, 16);
    
    // pythagoran kernel launch paramters
    dim3 blocksPerGridP(size.width * size.height / 256);
    dim3 threadsPerBlockP(256, 1);
     
    timer.start();
    {
        matrixConvGPU_float<<<blocksPerGrid,threadsPerBlock>>>(dIn, size.width, size.height, 2, 2, offsetX, 3, 3, dGradX);
        matrixConvGPU_float<<<blocksPerGrid,threadsPerBlock>>>(dIn, size.width, size.height, 2, 2, offsetY, 3, 3, dGradY);
        sobelGradientKernel_float<<<blocksPerGridP,threadsPerBlockP>>>(dGradX, dGradY, dOut);
    }
    timer.stop();
    cudaThreadSynchronize();
    double tms = timer.elapsed(); 
    //printf("Sobel (using constant memory) : Throughput in Megapixel per second : %.4f, Size : %d pixels, Elapsed time (in ms): %f\n",1.0e-6* (double)(size.height*size.width)/(tms*0.001),size.height*size.width,tms);
}

void launchSobel_restrict(unsigned char *dIn, unsigned char *dOut, unsigned char *dGradX, unsigned char *dGradY, cv::Size size,ssize_t offsetX,ssize_t offsetY)
{
    dim3 blocksPerGrid(size.width / 16, size.height / 16);
    dim3 threadsPerBlock(16, 16);
    
    // pythagoran kernel launch paramters
    dim3 blocksPerGridP(size.width * size.height / 256);
    dim3 threadsPerBlockP(256, 1);
     
    timer.start();
    {
        matrixConvGPU_restrict<<<blocksPerGrid,threadsPerBlock>>>(dIn, size.width, size.height, 2, 2, offsetX, 3, 3, dGradX);
        matrixConvGPU_restrict<<<blocksPerGrid,threadsPerBlock>>>(dIn, size.width, size.height, 2, 2, offsetY, 3, 3, dGradY);
        sobelGradientKernel_restrict<<<blocksPerGridP,threadsPerBlockP>>>(dGradX, dGradY, dOut);
    }
    timer.stop();
    cudaThreadSynchronize();
    double tms = timer.elapsed(); 
    //printf("Sobel (using constant memory) : Throughput in Megapixel per second : %.4f, Size : %d pixels, Elapsed time (in ms): %f\n",1.0e-6* (double)(size.height*size.width)/(tms*0.001),size.height*size.width,tms);
}

void launchSobel_sharedMem(unsigned char *dIn, unsigned char *dOut, unsigned char *dGradX, unsigned char *dGradY, cv::Size size,ssize_t offsetX,ssize_t offsetY)
{
    dim3 blocksPerGrid(size.width / 16, size.height / 16);
    dim3 threadsPerBlock(16, 16);
    
    // pythagoran kernel launch paramters
    dim3 blocksPerGridP(size.width * size.height / 256);
    dim3 threadsPerBlockP(256, 1);
     
    timer.start();
    {
        matrixConvGPU_sharedMem <TILE_W,SOBEL_KERNEL_RADIUS,TILE_W+2*SOBEL_KERNEL_RADIUS> <<<blocksPerGrid,threadsPerBlock>>>(dIn, size.width, size.height, offsetX, 3, 3, dGradX);
        matrixConvGPU_sharedMem <TILE_W,SOBEL_KERNEL_RADIUS,TILE_W+2*SOBEL_KERNEL_RADIUS> <<<blocksPerGrid,threadsPerBlock>>>(dIn, size.width, size.height, offsetY, 3, 3, dGradY);
        sobelGradientKernel_restrict<<<blocksPerGridP,threadsPerBlockP>>>(dGradX, dGradY, dOut);
    }
    timer.stop();
    cudaThreadSynchronize();
    double tms = timer.elapsed(); 
    //printf("Sobel (using constant memory) : Throughput in Megapixel per second : %.4f, Size : %d pixels, Elapsed time (in ms): %f\n",1.0e-6* (double)(size.height*size.width)/(tms*0.001),size.height*size.width,tms);
}

void launchSobel_constantMemory(unsigned char *dIn, unsigned char *dOut, unsigned char *dGradX, unsigned char *dGradY, cv::Size size,ssize_t offsetX,ssize_t offsetY)
{
    dim3 blocksPerGrid(size.width / 16, size.height / 16);
    dim3 threadsPerBlock(16, 16);
    
    // pythagoran kernel launch paramters
    dim3 blocksPerGridP(size.width * size.height / 256);
    dim3 threadsPerBlockP(256, 1);
     
    timer.start();
    {
        matrixConvGPU_constantMemory<<<blocksPerGrid,threadsPerBlock>>>(dIn, size.width, size.height, 2, 2, offsetX, 3, 3, dGradX);
        matrixConvGPU_constantMemory<<<blocksPerGrid,threadsPerBlock>>>(dIn, size.width, size.height, 2, 2, offsetY, 3, 3, dGradY);
        sobelGradientKernel<<<blocksPerGridP,threadsPerBlockP>>>(dGradX, dGradY, dOut);
    }
    timer.stop();
    cudaThreadSynchronize();
    double tms = timer.elapsed(); 
    //printf("Sobel (using constant memory) : Throughput in Megapixel per second : %.4f, Size : %d pixels, Elapsed time (in ms): %f\n",1.0e-6* (double)(size.height*size.width)/(tms*0.001),size.height*size.width,tms);
}

void launchSobelNaive_withoutPadding(unsigned char *dIn, unsigned char *dOut, unsigned char *dGradX, unsigned char *dGradY, cv::Size size, const float *d_X,const float *d_Y)
{
    dim3 blocksPerGrid(size.width / 16, size.height / 16);
    dim3 threadsPerBlock(16, 16);
    
    // Dimension for Sobel gradient kernel 
    dim3 blocksPerGridP(size.width * size.height / 256);
    dim3 threadsPerBlockP(256, 1);
     
    timer.start();
    {
        matrixConvGPUNaive_withoutPadding<<<blocksPerGrid,threadsPerBlock>>>(dIn, size.width, size.height, 3, 3, dGradX,d_X);
        matrixConvGPUNaive_withoutPadding<<<blocksPerGrid,threadsPerBlock>>>(dIn, size.width, size.height, 3, 3, dGradY,d_Y);
        sobelGradientKernel<<<blocksPerGridP,threadsPerBlockP>>>(dGradX, dGradY, dOut);
    }
    timer.stop();
    cudaThreadSynchronize();
    double tms = timer.elapsed(); 
    //printf("Sobel Naive (without padding): Throughput in Megapixel per second : %.4f, Size : %d pixels, Elapsed time (in ms): %f\n",1.0e-6* (double)(size.height*size.width)/(tms*0.001),size.height*size.width,tms);
}

void launchSobelNaive_withPadding(unsigned char *dIn, unsigned char *dOut, unsigned char *dGradX, unsigned char *dGradY, cv::Size size, const float *d_X,const float *d_Y)
{
    dim3 blocksPerGrid(size.width / 16, size.height / 16);
    dim3 threadsPerBlock(16, 16);
    
    // Dimension for Sobel gradient kernel 
    dim3 blocksPerGridP(size.width * size.height / 256);
    dim3 threadsPerBlockP(256, 1);
     
    timer.start();
    {
        matrixConvGPUNaive_withPadding<<<blocksPerGrid,threadsPerBlock>>>(dIn, size.width, size.height, 2, 2, 3, 3, dGradX,d_X);
        matrixConvGPUNaive_withPadding<<<blocksPerGrid,threadsPerBlock>>>(dIn, size.width, size.height, 2, 2, 3, 3, dGradY,d_Y);
        sobelGradientKernel<<<blocksPerGridP,threadsPerBlockP>>>(dGradX, dGradY, dOut);
    }
    timer.stop();
    cudaThreadSynchronize();
    double tms = timer.elapsed(); 
    //printf("Sobel Naive (with padding): Throughput in Megapixel per second : %.4f, Size : %d pixels, Elapsed time (in ms): %f\n",1.0e-6* (double)(size.height*size.width)/(tms*0.001),size.height*size.width,tms);
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

// Used for Sobel edge detection
// Calculate gradient value from gradientX and gradientY  
// Calculate G = sqrt(Gx^2 * Gy^2)
__global__ void sobelGradientKernel(unsigned char *gX, unsigned char *gY, unsigned char *dOut)
{
    int idx = (blockIdx.x * blockDim.x) + threadIdx.x;

    float x = float(gX[idx]);
    float y = float(gY[idx]);

    dOut[idx] = (unsigned char) sqrtf(x*x + y*y);
}

__global__ void sobelGradientKernel_float(unsigned char *gX, unsigned char *gY, unsigned char *dOut)
{
    int idx = (int)(((float)blockIdx.x * (float)blockDim.x) + (float)threadIdx.x);

    float x = float(gX[idx]);
    float y = float(gY[idx]);

    dOut[idx] = (unsigned char) sqrtf(x*x + y*y);
}

__global__ void sobelGradientKernel_restrict(unsigned char* __restrict__ gX, unsigned char* __restrict__ gY, unsigned char *dOut)
{
    int idx = (int)(((float)blockIdx.x * (float)blockDim.x) + (float)threadIdx.x);

    float x = float(gX[idx]);
    float y = float(gY[idx]);

    dOut[idx] = (unsigned char) sqrtf(x*x + y*y);
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
    int   kernelRadiusW = kernelW/2;
    int   kernelRadiusH = kernelH/2;

    // Determine pixels to operate 
    if(x >= kernelRadiusW && y >= kernelRadiusH &&
       x < (blockDim.x * gridDim.x) - kernelRadiusW &&
       y < (blockDim.y * gridDim.y)-kernelRadiusH)
    {
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
    int   kernelRadiusW = kernelW/2;
    int   kernelRadiusH = kernelH/2;

    // Determine pixels to operate 
    if(x >= (kernelRadiusW + paddingX) && y >= (kernelRadiusH + paddingY) &&
       x < ((blockDim.x * gridDim.x) - kernelRadiusW - paddingX) &&
       y < ((blockDim.y * gridDim.y) - kernelRadiusH - paddingY))
    {
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
    int   kernelRadiusW = kernelW/2;
    int   kernelRadiusH = kernelH/2;

    // Determine pixels to operate 
    if(x >= (kernelRadiusW + paddingX) && y >= (kernelRadiusH + paddingY) &&
       x < ((blockDim.x * gridDim.x) - kernelRadiusW - paddingX) &&
       y < ((blockDim.y * gridDim.y) - kernelRadiusH - paddingY))
    {
        for(int i = -kernelRadiusH; i <= kernelRadiusH; i++) // Along Y axis
        {
            for(int j = -kernelRadiusW; j <= kernelRadiusW; j++) //Along X axis
            {
                // Sample the weight for this location
                int jj = (j+kernelRadiusW);
                int ii = (i+kernelRadiusH);
                float w  = constConvKernelMem[(ii * kernelW) + jj + kernelOffset]; //kernel from constant memory
                 
                accum += w * float(dIn[((y+i) * width) + (x+j)]);
            }
        }
    }
    
    dOut[(y * width) + x] = (unsigned char) accum;
}

__global__ void matrixConvGPU_float(unsigned char *dIn, int width, int height, int paddingX, int paddingY, ssize_t kernelOffset, int kernelW, int kernelH, unsigned char *dOut)
{
    // Calculate our pixel's location
    float x = ((float)blockIdx.x * (float)blockDim.x) + (float)threadIdx.x;
    float y = ((float)blockIdx.y * (float)blockDim.y) + (float)threadIdx.y;

    // Calculate radius along X and Y axis
    // We can also use one kernel variable instead - kernel radius
    float accum = 0.0;
    int   kernelRadiusW = kernelW/2;
    int   kernelRadiusH = kernelH/2;

    // Determine pixels to operate 
    if(x >= ((float)kernelRadiusW + (float)paddingX) && y >= ((float)kernelRadiusH + (float)paddingY) &&
       x < (((float)blockDim.x * (float)gridDim.x) - (float)kernelRadiusW - (float)paddingX) &&
       y < (((float)blockDim.y * (float)gridDim.y) - (float)kernelRadiusH - (float)paddingY))
    {
        for(int i = -kernelRadiusH; i <= kernelRadiusH; i++) // Along Y axis
        {
            for(int j = -kernelRadiusW; j <= kernelRadiusW; j++) //Along X axis
            {
                // Sample the weight for this location
                float jj = ((float)j+(float)kernelRadiusW);
                float ii = ((float)i+(float)kernelRadiusH);
                float w  = constConvKernelMem[(int)((ii * (float)kernelW) + jj + (float)kernelOffset)]; //kernel from constant memory
                 
                accum += w * float(dIn[(int)(((y+(float)i) * (float)width) + (x+(float)j))]);
            }
        }
    }
    
    dOut[(int)((y * (float)width) + x)] = (unsigned char) accum;
}

__global__ void matrixConvGPU_restrict(unsigned char* __restrict__ dIn, int width, int height, int paddingX, int paddingY, ssize_t kernelOffset, int kernelW, int kernelH, unsigned char* __restrict__ dOut)
{
    // Calculate our pixel's location
    float x = ((float)blockIdx.x * (float)blockDim.x) + (float)threadIdx.x;
    float y = ((float)blockIdx.y * (float)blockDim.y) + (float)threadIdx.y;

    // Calculate radius along X and Y axis
    // We can also use one kernel variable instead - kernel radius
    float accum = 0.0;
    int   kernelRadiusW = kernelW/2;
    int   kernelRadiusH = kernelH/2;

    // Determine pixels to operate 
    if(x >= ((float)kernelRadiusW + (float)paddingX) && y >= ((float)kernelRadiusH + (float)paddingY) &&
       x < (((float)blockDim.x * (float)gridDim.x) - (float)kernelRadiusW - (float)paddingX) &&
       y < (((float)blockDim.y * (float)gridDim.y) - (float)kernelRadiusH - (float)paddingY))
    {
        for(int i = -kernelRadiusH; i <= kernelRadiusH; i++) // Along Y axis
        {
            for(int j = -kernelRadiusW; j <= kernelRadiusW; j++) //Along X axis
            {
                // Sample the weight for this location
                float jj = ((float)j+(float)kernelRadiusW);
                float ii = ((float)i+(float)kernelRadiusH);
                float w  = constConvKernelMem[(int)((ii * (float)kernelW) + jj + (float)kernelOffset)]; //kernel from constant memory
                 
                accum += w * float(dIn[(int)(((y+(float)i) * (float)width) + (x+(float)j))]);
            }
        }
    }
    
    dOut[(int)((y * (float)width) + x)] = (unsigned char) accum;
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
      if ( ty < rad || ty > height - rad)
         return;
   } else {
      if (tx < rad || tx > width - rad)
         return;
   }

   //compute values depending on if this is row or col vector
   float accum = 0;
   for (int i = -rad; i <= rad; i++) {
      if (phase1) {
         accum += (float)d_input[tx + (ty + i)*width] * constConvKernelMem[kOffset + i];
      } else {
         accum += d_separableBuffer[tx + i + ty*width] * constConvKernelMem[kOffset + i];
      }
   }

   //update output, if phase1 then we need to store values which are >255 in temp storage for next phase
   if (phase1) {
      d_separableBuffer[tx + ty*width] = accum;
   } else {
      accum *= alpha;
      accum = accum > 255 ? 255 : accum; //threshold the pixel
      accum = accum < 0 ? 0 : accum;
      d_output[tx + ty*width] = (unsigned char)accum;
   }
}

template<const int TILE_WIDTH, const int KERNEL_RADIUS, const int SMEM_WIDTH>
__global__ void matrixConvGPU_sharedMem(unsigned char* dIn, int width, int height, ssize_t kernelOffset, int kernelW, int kernelH, unsigned char* dOut)
{
    __shared__ char s_data[SMEM_WIDTH*SMEM_WIDTH];

    // Calculate output pixel's location
    int row = blockIdx.y * TILE_WIDTH + threadIdx.y;
    int col = blockIdx.x * TILE_WIDTH + threadIdx.x;
    
    // Calculate index of thread in thread block based on TILE_WIDTH
    int smem_index = threadIdx.x + threadIdx.y * TILE_WIDTH;
    // Calculate 2-D coordinate based on shared_memory size which is greater than TILE_WIDTH
    int dy = smem_index / SMEM_WIDTH;
    int dx = smem_index % SMEM_WIDTH;
    
    // 
    float accum = 0.0;
    
    //
    int shifted_row =  dy + (blockIdx.y * TILE_WIDTH) - KERNEL_RADIUS;
    int shifted_col =  dx + (blockIdx.x * TILE_WIDTH) - KERNEL_RADIUS;
    int gmem_index = shifted_col + shifted_row * width;
    
    // Load TILE_WIDTH x TILE_WIDTH data from global memory
    if(shifted_row >= 0 && ((shifted_row) < height) &&
       shifted_col >= 0 && (shifted_col < width))
        s_data[dy * SMEM_WIDTH + dx] = dIn[gmem_index];
    else
        s_data[dy * SMEM_WIDTH + dx] = 0; 
    __syncthreads();  // make sure all thread has finished execution and shared memory is populated with required matrix tile
  
    // Calculate index and 2-D coordinates for loading apron data which
    // does not gets fetched during previous load of TILE_WIDTHxTILE_WIDTH
    // Note the offset below - TILE_WIDTH*TILE_WIDTH
    smem_index  = threadIdx.x + (threadIdx.y * TILE_WIDTH) + (TILE_WIDTH * TILE_WIDTH);  
    dy = smem_index / SMEM_WIDTH;
    dx = smem_index % SMEM_WIDTH; 
    
    shifted_row =  dy +  (blockIdx.y * TILE_WIDTH) - KERNEL_RADIUS;
    shifted_col =  dx +  (blockIdx.x * TILE_WIDTH) - KERNEL_RADIUS;
    gmem_index = shifted_col + shifted_row * width;

    if(dy<SMEM_WIDTH) // ignore threads outside apron 
    {
        if(shifted_row >= 0 && ((shifted_row) < height) &&
           shifted_col >= 0 && (shifted_col < width))
           s_data[dy * SMEM_WIDTH + dx] = dIn[gmem_index];
        else
            s_data[dy * SMEM_WIDTH + dx] = 0; 
    }
    __syncthreads();  // make sure all thread has finished execution and shared memory is populated with required matrix tile
    
    // Perform Convolution
    /*for(int i = -kernelRadiusH; i <= kernelRadiusH; i++) // Along Y axis
     // {}
        for(int j = -kernelRadiusW; j <= kernelRadiusW; j++) //Along X axis
    */ 
     //Simplify above operation
     for(int i = 0; i < kernelH; i++) // Along Y axis
     {
        for(int j = 0; j < kernelW; j++) //Along X axis
        {
                // Sample the weight for this location
                /*float jj = ((float)j+(float)kernelRadiusW);
                  float ii = ((float)i+(float)kernelRadiusH);
                  float w  = constConvKernelMem[(int)((ii * (float)kernelW) + jj + (float)kernelOffset)]; //kernel from constant memory
                */
                float w  = constConvKernelMem[i * kernelW + (j + kernelOffset)];
                accum += w * (float)s_data[(threadIdx.y + i) * SMEM_WIDTH + (threadIdx.x + j)];
        }
     }
   
   // All threads does not write to output buffer
   if(row < height && col < width) 
       dOut[row * width + col] = (unsigned char) accum;
   __syncthreads();  // make sure all thread has finished execution and shared memory is populated with required matrix tile
}

