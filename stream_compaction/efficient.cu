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

            // d = 1
            // 4
            int multiple = 1 << (offset + 1);

            int newIndex = index * multiple;

            if (newIndex > n - 1) {
                return;
            }

            // bit shift equivalent to power of 2
            int base = multiple >> 1;

            // going left to right now
            data[newIndex + multiple - 1] += data[newIndex + base - 1];            
        }

        __global__ void kernDownSweep(int n, int* data, int offset) {
            int index = threadIdx.x + (blockIdx.x * blockDim.x);

            int multiple = 1 << (offset + 1);

            int newIndex = index * multiple;

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
            timer().startGpuTimer();
            // TODO

            int padding = 1 << ilog2ceil(n);

            int blockSize = 128;
            dim3 fullBlocksPerGrid((padding + blockSize - 1) / blockSize);

            // read & write buffer b/c no overlap now
            int* dev_data;

            // CUDA memory management and error checking.
            cudaMalloc((void**)&dev_data, padding * sizeof(int));
            checkCUDAError("cudaMalloc data failed!");

            cudaMemset(dev_data, 0, padding * sizeof(int));

            // copying idata into buffer
            cudaMemcpy(dev_data + padding - n, idata, sizeof(int) * n, cudaMemcpyHostToDevice);

            for (int d = 0; d <= ilog2ceil(padding) - 1; d++) {
                // typical CUDA kernel invocation.
                kernUpSweep <<< fullBlocksPerGrid, blockSize >>> (padding, dev_data, d);
                checkCUDAError("UpSweep failed!");

                // synchronize
                cudaDeviceSynchronize();
            }

            // put zero in root (last elem of dev_data)
            int zerotest[1] = { 0 }; // check if this or int?
            cudaMemcpy(dev_data + padding - 1, zerotest, sizeof(int), cudaMemcpyHostToDevice);

            for (int d = ilog2ceil(padding) - 1; d >= 0; d--) {
                kernDownSweep <<< fullBlocksPerGrid, blockSize>>> (padding, dev_data, d);
                checkCUDAError("DownSweep failed!");

                cudaDeviceSynchronize();
            }

            timer().endGpuTimer();
        
            cudaMemcpy(odata, dev_data + padding - n, sizeof(int) * n, cudaMemcpyDeviceToHost);

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

            cudaMemset(dev_in, 0, padding * sizeof(int));
            cudaMemcpy(dev_in + padding - n, idata, sizeof(int) * n, cudaMemcpyHostToDevice);

            timer().startGpuTimer();
            // TODO
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
            int zerotest[1] = { 0 }; // check if this or int?
            cudaMemcpy(dev_indices + padding - 1, zerotest, sizeof(int), cudaMemcpyHostToDevice);

            for (int d = ilog2ceil(padding) - 1; d >= 0; d--) {
                kernDownSweep << < fullBlocksPerGrid, blockSize >> > (padding, dev_indices, d);
                checkCUDAError("DownSweep failed!");

                cudaDeviceSynchronize();
            }

            Common::kernScatter << < fullBlocksPerGrid, blockSize >> > (n, dev_out,
                dev_in, dev_bools, dev_indices);
            
            timer().endGpuTimer();
            // size is in last elem of indices
            int size[1];
            int sizeTest[1];

            cudaMemcpy(size, dev_indices + padding - 1, sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemcpy(sizeTest, dev_bools + padding - 1, sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemcpy(odata, dev_out + padding - n, sizeof(int) * n, cudaMemcpyDeviceToHost);

            cudaFree(dev_in);
            cudaFree(dev_bools);
            cudaFree(dev_indices);
            cudaFree(dev_out);

            int returnval = size[0] + sizeTest[0];

            return returnval;
        }
    }
}
