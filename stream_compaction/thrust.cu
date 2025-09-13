#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/scan.h>
#include "common.h"
#include "thrust.h"

namespace StreamCompaction {
    namespace Thrust {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }
        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            thrust::device_vector<int> dev_thrust_input(idata, idata + n);
            thrust::device_vector<int> dev_thrust_output(n);

            timer().startGpuTimer();
            thrust::exclusive_scan(dev_thrust_input.begin(), dev_thrust_input.end(), dev_thrust_output.begin());
            timer().endGpuTimer();

            thrust::copy(dev_thrust_output.begin(), dev_thrust_output.end(), odata);
        }
    }
}
