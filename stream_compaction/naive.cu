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
        
        // computes inclusive sum!
        __global__ void kernNaiveScan(int n, int* odata, const int* idata, int offset) {
            int index = threadIdx.x + (blockIdx.x * blockDim.x);

            // Check if idx is out of bounds. If yes, return.
            if (index >= n)
                return;

            // bit shift equivalent to power of 2
            int pow = 1 << (offset - 1);

            // this ver is not in-place (which reads from both in & out arrs)
            if (index >= pow) {
                odata[index] = idata[index - pow] + idata[index];
            }
            else {
                // takes care of setting elems that are already added in prev iter
                odata[index] = idata[index];
            }
        }

        // to make sure it's exclusive!!
        // idata = odata here (using my buffers)
        __global__ void kernShiftArray(int n, int* odata, int*idata) {
            int index = threadIdx.x + (blockIdx.x * blockDim.x);

            // don't want the last value here
            if (index >= n - 1) {
                return;
            }

            odata[index+1] = idata[index];
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            // adding padding for non power of 2 len array
            int padding = 1 << ilog2ceil(n);

            int blockSize = 128;
            dim3 fullBlocksPerGrid((padding + blockSize - 1)/blockSize);

            // read buffer
            int* dev_dataA;
            // write buffer
            int* dev_dataB; 

            // CUDA memory management and error checking.
            cudaMalloc((void**)&dev_dataA, padding * sizeof(int));
            checkCUDAError("cudaMalloc dataA failed!");

            cudaMalloc((void**)&dev_dataB, padding * sizeof(int));
            checkCUDAError("cudaMalloc dataB failed!");

            // setting array to 0
            cudaMemset(dev_dataA, 0, padding * sizeof(int));

            // copying idata into buffer
            cudaMemcpy(dev_dataA, idata, sizeof(int) * n, cudaMemcpyHostToDevice);
            
            timer().startGpuTimer();

            for (int d = 1; d <= ilog2ceil(padding); d++) {
                // typical CUDA kernel invocation.
                kernNaiveScan <<< fullBlocksPerGrid, blockSize >>> (padding, dev_dataB, dev_dataA, d);
                checkCUDAError("NaiveScan failed!");

                // synchronize
                cudaDeviceSynchronize();

                // swap buffers
                int* tempPtr = dev_dataB;
                dev_dataB = dev_dataA;
                dev_dataA = tempPtr;
            }
            
            // exclusive process b/c above is inclusive
            // shift array to right by 1
            kernShiftArray << < fullBlocksPerGrid, blockSize >> > (padding, dev_dataB, dev_dataA);
            checkCUDAError("ShiftArray failed!");

            timer().endGpuTimer();

            // putting data into odata
            cudaMemcpy(odata, dev_dataB, sizeof(int) * n, cudaMemcpyDeviceToHost);

            // setting the first elem to identity
            // doing here to prevent branch (does this matter?)
            odata[0] = 0;

            cudaFree(dev_dataA);
            cudaFree(dev_dataB);
        }
    }
}
