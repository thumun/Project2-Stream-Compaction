#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"

namespace StreamCompaction {
    namespace Efficient {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        __global__ void kernUpSweep(int n, int* data, int offset) {
            int index = threadIdx.x + (blockIdx.x * blockDim.x);

            // don't want the last value here
            if (index >= n) {
                return;
            }

            // need to do this in parallel hmm
            // for all k = 0 to n – 1 by 2d+1 in parallel

            // bit shift equivalent to power of 2
            int base = 1 << offset;
            int multiple = 1 << (offset + 1);

            // going left to right now
            data[index + multiple - 1] += data[index + base - 1];
        }

        __global__ void kernDownSweep(int n, int* data, int offset) {
            int index = threadIdx.x + (blockIdx.x * blockDim.x);

            // don't want the last value here
            if (index >= n) {
                return;
            }
            
            // need to do this!!
            // for all k = 0 to n – 1 by 2d+1 in parallel

            int base = 1 << offset;
            int multiple = 1 << (offset + 1);

            int leftChild = data[index + base - 1];
            // setting left as right child's val
            data[index + base - 1] = data[index + multiple - 1];
            // adding left child to right
            data[index + multiple - 1] += leftChild;
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            timer().startGpuTimer();
            // TODO

            int blockSize = 128;
            dim3 fullBlocksPerGrid((n + blockSize - 1) / blockSize);

            // read & write buffer b/c no overlap now
            int* dev_data;

            // CUDA memory management and error checking.
            cudaMalloc((void**)&dev_data, n * sizeof(int));
            checkCUDAError("cudaMalloc data failed!");

            // copying idata into buffer
            cudaMemcpy(dev_data, idata, sizeof(int) * n, cudaMemcpyHostToDevice);

            for (int d = 0; d <= ilog2ceil(n) - 1; d++) {
                // typical CUDA kernel invocation.
                kernUpSweep << < fullBlocksPerGrid, blockSize >> > (n, dev_data, d);
                checkCUDAError("UpSweep failed!");

                // synchronize
                cudaDeviceSynchronize();
            }

            // put zero in root (last elem of dev_data)

            for (int d = 0; d <= ilog2ceil(n) - 1; d++) {
                kernDownSweep << < fullBlocksPerGrid, blockSize>> > (n, dev_data, d);
            
                cudaDeviceSynchronize();
            }

            timer().endGpuTimer();
        }

        /**
         * Performs stream compaction on idata, storing the result into odata.
         * All zeroes are discarded.
         *
         * @param n      The number of elements in idata.
         * @param odata  The array into which to store elements.
         * @param idata  The array of elements to compact.
         * @returns      The number of elements remaining after compaction.
         */
        int compact(int n, int *odata, const int *idata) {
            timer().startGpuTimer();
            // TODO
            timer().endGpuTimer();
            return -1;
        }
    }
}
