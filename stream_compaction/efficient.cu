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

        // based on the logic in the slides
        // in-place sum
        __global__ void kernUpSweep(int n, int* data, int offset) {
            int index = threadIdx.x + (blockIdx.x * blockDim.x);

            // bit-wise 2^(d+1)
            int multiple = 1 << (offset + 1);

            // mapping threads to be every 2^(d+1) index in array
            int newIndex = index * multiple;

            // only want values up until n-1
            if (newIndex > n - 1) {
                return;
            }

            // bit shift equivalent to power of 2
            int base = multiple >> 1;

            // going left to right now
            data[newIndex + multiple - 1] += data[newIndex + base - 1];            
        }

        // based on logic in slides
        __global__ void kernDownSweep(int n, int* data, int offset) {
            int index = threadIdx.x + (blockIdx.x * blockDim.x);
            
            // bit-wise 2^(d+1)
            int multiple = 1 << (offset + 1);

            // mapping threads to be every 2^(d+1) index in array
            int newIndex = index * multiple;

            // only want values up until n-1
            if (newIndex > n - 1) {
                return;
            }

            // bit shift equivalent to power of 2
            int base = multiple >> 1;

            int leftChild = data[newIndex + base - 1];
            // setting left as right child's val
            data[newIndex + base - 1] = data[newIndex + multiple - 1];
            // adding left child to right
            data[newIndex + multiple - 1] += leftChild;
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            // adding padding for non power of 2 len array
            int padding = 1 << ilog2ceil(n);

            int blockSize = 128;
            dim3 fullBlocksPerGrid((padding + blockSize - 1) / blockSize);

            // read & write buffer b/c no overlap now
            int* dev_data;

            // CUDA memory management and error checking.
            cudaMalloc((void**)&dev_data, padding * sizeof(int));
            checkCUDAError("cudaMalloc data failed!");

            // setting array to 0
            cudaMemset(dev_data, 0, padding * sizeof(int));

            // copying idata into buffer
            cudaMemcpy(dev_data, idata, sizeof(int) * n, cudaMemcpyHostToDevice);

            timer().startGpuTimer();

            for (int d = 0; d <= ilog2ceil(padding) - 1; d++) {
                kernUpSweep <<< fullBlocksPerGrid, blockSize >>> (padding, dev_data, d);
                checkCUDAError("UpSweep failed!");

                // synchronize
                cudaDeviceSynchronize();
            }

            // put zero in root (last elem of dev_data)
            int zerotest[1] = { 0 };
            cudaMemcpy(dev_data + padding - 1, zerotest, sizeof(int), cudaMemcpyHostToDevice);

            for (int d = ilog2ceil(padding) - 1; d >= 0; d--) {
                kernDownSweep <<< fullBlocksPerGrid, blockSize>>> (padding, dev_data, d);
                checkCUDAError("DownSweep failed!");

                cudaDeviceSynchronize();
            }

            timer().endGpuTimer();
        
            // copying back into output
            cudaMemcpy(odata, dev_data, sizeof(int) * n, cudaMemcpyDeviceToHost);

            cudaFree(dev_data);
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
            // adding padding for non power of 2 len array
            int padding = 1 << ilog2ceil(n);

            int blockSize = 128;
            dim3 fullBlocksPerGrid((padding + blockSize - 1) / blockSize);

            int* dev_indices;
            int* dev_bools;
            int* dev_out;
            int* dev_in;

            // CUDA memory management and error checking.
            cudaMalloc((void**)&dev_indices, padding * sizeof(int));
            checkCUDAError("cudaMalloc indices failed!");
            
            cudaMalloc((void**)&dev_bools, padding * sizeof(int));
            checkCUDAError("cudaMalloc bools failed!");

            cudaMalloc((void**)&dev_in, padding * sizeof(int));
            checkCUDAError("cudaMalloc in failed!");

            cudaMalloc((void**)&dev_out, padding * sizeof(int));
            checkCUDAError("cudaMalloc out failed!");

            // setting array to 0
            cudaMemset(dev_in, 0, padding * sizeof(int));
            // copying idata into buffer
            cudaMemcpy(dev_in, idata, sizeof(int) * n, cudaMemcpyHostToDevice);

            timer().startGpuTimer();

            // creating array of bool: 1 = needed, 0 = not needed
            Common::kernMapToBoolean << < fullBlocksPerGrid, blockSize >> > (padding, dev_bools, dev_in);
            checkCUDAError("MapToBool failed!");

            // need to do scan on temp array
            cudaMemcpy(dev_indices, dev_bools, sizeof(int) * padding, cudaMemcpyDeviceToDevice);

            for (int d = 0; d <= ilog2ceil(padding) - 1; d++) {
                // typical CUDA kernel invocation.
                kernUpSweep << < fullBlocksPerGrid, blockSize >> > (padding, dev_indices, d);
                checkCUDAError("UpSweep failed!");

                // synchronize
                cudaDeviceSynchronize();
            }

            // put zero in root (last elem of dev_data)
            int zerotest[1] = { 0 };
            cudaMemcpy(dev_indices + padding - 1, zerotest, sizeof(int), cudaMemcpyHostToDevice);

            for (int d = ilog2ceil(padding) - 1; d >= 0; d--) {
                kernDownSweep << < fullBlocksPerGrid, blockSize >> > (padding, dev_indices, d);
                checkCUDAError("DownSweep failed!");

                cudaDeviceSynchronize();
            }

            // get needed values in output (marked as 1 in bools)
            Common::kernScatter << < fullBlocksPerGrid, blockSize >> > (n, dev_out,
                dev_in, dev_bools, dev_indices);
            
            timer().endGpuTimer();

            // size is in last elem of indices and bools!
            int indxSize[1];
            int boolSize[1];

            cudaMemcpy(indxSize, dev_indices + n - 1, sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemcpy(boolSize, dev_bools + n - 1, sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemcpy(odata, dev_out, sizeof(int) * n, cudaMemcpyDeviceToHost);

            // deallocating
            cudaFree(dev_in);
            cudaFree(dev_bools);
            cudaFree(dev_indices);
            cudaFree(dev_out);

            int returnval = indxSize[0] + boolSize[0];

            return returnval;
        }
    }
}
