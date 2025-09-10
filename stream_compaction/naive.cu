#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "naive.h"

namespace StreamCompaction {
    namespace Naive {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }
        
        __global__ void naiveScan(int n, int* odata, const int* idata, int offset) {
            int index = threadIdx.x + (blockIdx.x * blockDim.x);

            // Check if idx is out of bounds. If yes, return.
            if (index >= N)
                return;

            // need to add 0 to beginning somewhere?? 

            if (index >= offset) {
                odata[n] = idata[n - offset] + idata[n];
            }
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            timer().startGpuTimer();
            // TODO
            
            // make 2 buffers a & b
            // write to one
            // read from other



            // d = 1 to ln(n) 
            // loop through above && create threads for scan
            
            // swap buffers

            timer().endGpuTimer();
        }
    }
}
