#ifndef __HELPER_FUNCS_H__
#define __HELPER_FUNCS_H__

/**
Helper functions for the convolution program.
**/

#include <string>
#include <cuda_runtime.h>

/**
Returns the formatted string with the metrics passed in.
@param int frameCounter       the frame #
@param float fps              frames per second
@param float mps              Megapixels per second
@param string kernel_t        string of the kernel type used
@param double tms             time elapsed
@return vector<string>        formatted string vector with metric info
**/
static std::vector<std::string> getMetricString(int frameCounter, float fps, float mps, std::string kernel_t, double tms) {
   char charOutputBuf[256];
   std::vector<std::string> metricString;
   std::string s1,s2,s3;
   s1 = "[Kernel Type : "; s1 +=kernel_t; s1 += "]";
   sprintf(charOutputBuf, "[Frame #:%d] [Kernel Time(ms):%.4f]", frameCounter, tms);
   s2 = charOutputBuf;
   sprintf(charOutputBuf, "[FPS:%2.3f] [MPS:%.4f]", fps, mps);
   s3 = charOutputBuf;
   metricString.push_back(s1);
   metricString.push_back(s2);
   metricString.push_back(s3);
   return metricString;
};

/**
Print a matrix to file from device memory (VERY SLOW).
@param unsigned char *d_mat      device array to print to file
@param int width, height         the width and height of the device array
@param string filename           name of file to print to (w+)
@precondition                    filename < 100 chars
**/
static void printDeviceMatrix(unsigned char *d_mat, int width, int height, std::string filename) {
   char buff[100];
   std::string path = "./" + filename + ".txt";
   sprintf(buff, "%s", path);
   FILE *file = fopen(buff, "w+");
   
   size_t size = width * height;
   unsigned char *tmp = new unsigned char[size];
   cudaMemcpy(tmp, d_mat, size, cudaMemcpyDeviceToHost);

   for (int i = 0; i < height; i++) {
      for (int j = 0; j < width; j++) {
         fprintf(file, "%3d ", tmp[i*width + j]);
      }
      fprintf(file, "\n");
   }

   free(tmp);
};

/**
If Windows then define clock_gettime
**/
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
