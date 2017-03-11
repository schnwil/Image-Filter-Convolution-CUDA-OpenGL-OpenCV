#ifndef __HELPER_FUNCS_H__
#define __HELPER_FUNCS_H__

/**
Helper functions for the convolution program.
**/

#include <string>

/**
Returns the formatted string with the metrics passed in.
@param int frameCounter       the frame #
@param float fps              frames per second
@param float mps              Megapixels per second
@param string kernel_t        string of the kernel type used
@param double tms             time elapsed
@return string                formatted string with metric info
**/
static std::string getMetricString(int frameCounter, float fps, float mps, std::string kernel_t, double tms) {
   char charOutputBuf[256];
   sprintf(charOutputBuf, "[Frame #:%d] [FPS:%2.3f] [MPS: %.4f] [Kernel Type ", frameCounter, fps, mps);
   std::string metricString = charOutputBuf; metricString += kernel_t;
   sprintf(charOutputBuf, "][Kernel Time(ms):%.4f]", tms);
   metricString += charOutputBuf;
   return metricString;
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