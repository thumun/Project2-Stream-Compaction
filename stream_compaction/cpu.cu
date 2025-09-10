#include <cstdio>
#include "cpu.h"

#include "common.h"

namespace StreamCompaction {
    namespace CPU {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        /**
         * CPU scan (exclusive prefix sum).
         * For performance analysis, this is supposed to be a simple for loop.
         * (Optional) For better understanding before starting moving to GPU, you can simulate your GPU scan in this function first.
         */
        void scan(int n, int *odata, const int *idata) {
            timer().startCpuTimer();
            // want to do this outside if b/c no branching
            // also set to 0 b/c it's exclusive
            odata[0] = 0;

            for (int i = 1; i < n; i++) {
                odata[i] = idata[i - 1] + odata[i - 1];
            }

            timer().endCpuTimer();
        }

        /**
         * CPU stream compaction without using the scan function.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithoutScan(int n, int *odata, const int *idata) {
            timer().startCpuTimer();
            // stream compaction = only want to return arr of vals that meet crit
            int updateIndex = 0;

            for (int i = 0; i < n; i++) {
                if (idata[i] != 0) {
                    odata[updateIndex] = idata[i];
                    updateIndex++;
                }
            }

            timer().endCpuTimer();
            return updateIndex;
        }

        /**
         * CPU stream compaction using scan and scatter, like the parallel version.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithScan(int n, int *odata, const int *idata) {
            timer().startCpuTimer();
            // Step 1: temp array with 1's (meet crit) and 0's (not meet crit)
            int* tempArr = new int[n];

            // better on gpu b/c wld be parallel 
            for (int i = 0; i < n; i++) {
                if (idata[i] == 0) {
                    tempArr[i] = 0;
                }
                else {
                    tempArr[i] = 1;
                }
            }

            // Step 2: Run exclusive scan on temp arr
            // Result of scan is index into final array
            int* exclusiveArr = new int[n];
            exclusiveArr[0] = 0;

            for (int i = 1; i < n; i++) {
                exclusiveArr[i] = tempArr[i - 1] + exclusiveArr[i - 1];
            }

            // Step 3: Scatter -> create final arr
            int arrLen = 0; 

            for (int i = 0; i < n; i++) {
                if (tempArr[i] == 1) {
                    odata[exclusiveArr[i]] = idata[i];
                    arrLen++;
                }
            }

            timer().endCpuTimer();
            return arrLen;
        }
    }
}
