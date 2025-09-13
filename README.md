CUDA Stream Compaction
======================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 2**

* Neha Thumu
  * [LinkedIn](https://www.linkedin.com/in/neha-thumu/)
* Tested on: Windows 11 Pro, i9-13900H @ 2.60GHz 32GB, Nvidia GeForce RTX 4070

## Project 2: Details 

### Implementation
This project focused on the implementation of two algorithms: scan (prefix sum) and stream compaction. In this implementation of the scan algorithm, an exclusive prefix sum is being computed. This involves prepending an identity (in our case, 0) to the front of the array while the final element contains the sum. The stream compaction algorithm utilizes the scan algorithm in its process but the main idea is to remove elements that we are not using from our data: this takes the form of first mapping the data to 0's and 1's and then discarding those marked as 0. 

#### Scan Methods
**CPU**: This is the baseline method that is a point of comparison to the GPU methods. As it is on the CPU, loops are being utilized.

**Naive**: This method creates threads for pairs of values in the input array and computes the sum for each of the pairs until we have the cumulative sum. This method computes an inclusive sum that needs two arrays (input and output) rather than being an in-place computation in order to prevent reading/writing from the same value for multiple threads. In order to ensure the result is an exclusive prefix sum, the resultant array is shifted to the right by 1 and the identity (in this case, 0) is prepended.

**Work-Efficient**: This method builds off the naive method but is able to do an in-place calculation without shifting the array by using two key methods: UpSweep and DownSweep. UpSweep is similar to the naive scan but in order to calculate in-place, we need to ensure that threads do not read/write to the same index. This involves mapping the threads to the indices they need to read from and utlizing an offset to find which indices are crucial in our current iteration. The DownSweep method uses a tree-like structure to go through our array and add the left and right children together in order to eventually get our sum. 

**Thrust**: This method uses the CUDA Thrust library which can be found [here](https://nvidia.github.io/cccl/thrust/). This uses the exclusive_scan function to do to the scan method. 

#### Stream Compaction
**CPU**: This is the baseline method that is a point of comparison to the GPU methods. As it is on the CPU, loops are being utilized.

**GPU**: This method uses the scan and scatter methods in order to perform stream compaction. First, a temporary array of bools is created where the values that we want to discard are marked as 0 and others are marked as 1. We then use the scan method from work efficient (UpStream and DownStream) on the temporary array. After that, the scatter method is employed to only write values from the input array that are marked as 1 in our temporary array to the output array.

### Analysis
#### Optimal block size for each method
These charts show data where different blocksizes were tested for the Naive and Work Efficient scan methods to see how the runtime changes based on these sizes. The first chart shows how the runtime looks for an array size where the length is a power of two while the second shows how it looks for an array length that is not a power of two.

![blocksizevsruntime](https://github.com/thumun/Project2-Stream-Compaction/blob/main/img/blocksize_runtime_powtwo.png)

![blocksizevsruntime](https://github.com/thumun/Project2-Stream-Compaction/blob/main/img/blocksize_runtime_nonpowtwo.png)

**Naive**: This method

**Work-Efficient**: This method  

#### Comparison of methods
These charts compare the GPU and CPU methods for Scan by seeing how the runtime changes depending on different array sizes. The first chart shows how the runtime looks for an array size where the length are powers of two while the second shows how it looks for array lengths that are not powers of two. The block size is fixed as 128 in both of these charts.

![arraysizevsruntime](https://github.com/thumun/Project2-Stream-Compaction/blob/main/img/arraysize_runtime_powtwo.png)

![arraysizevsruntime](https://github.com/thumun/Project2-Stream-Compaction/blob/main/img/arraysize_runtime_nonpowtwo.png)

**CPU**: /

**Naive**: This method

**Work-Efficient**: This method  

**Thrust**: /

#### Phenomena Thoughts
do this!!!

#### Output of test program (for block size of 128)
```
****************
** SCAN TESTS **
****************
    [  38  49  43  31   9  18  18  30  34  21  18  49  44 ...  45   0 ]
==== cpu scan, power-of-two ====
   elapsed time: 0.002ms    (std::chrono Measured)
    [   0  38  87 130 161 170 188 206 236 270 291 309 358 ... 6643 6688 ]
==== cpu scan, non-power-of-two ====
   elapsed time: 0.0013ms    (std::chrono Measured)
    [   0  38  87 130 161 170 188 206 236 270 291 309 358 ... 6524 6557 ]
    passed
==== naive scan, power-of-two ====
   elapsed time: 1.152ms    (CUDA Measured)
    passed
==== naive scan, non-power-of-two ====
   elapsed time: 0.815104ms    (CUDA Measured)
    passed
==== work-efficient scan, power-of-two ====
   elapsed time: 0.968704ms    (CUDA Measured)
    passed
==== work-efficient scan, non-power-of-two ====
   elapsed time: 0.523264ms    (CUDA Measured)
    passed
==== thrust scan, power-of-two ====
   elapsed time: 0.456704ms    (CUDA Measured)
    passed
==== thrust scan, non-power-of-two ====
   elapsed time: 0.101632ms    (CUDA Measured)
    passed

*****************************
** STREAM COMPACTION TESTS **
*****************************
    [   0   1   3   3   3   0   0   2   2   1   0   3   0 ...   3   0 ]
==== cpu compact without scan, power-of-two ====
   elapsed time: 0.0021ms    (std::chrono Measured)
    [   1   3   3   3   2   2   1   3   3   2   1   3   3 ...   3   3 ]
    passed
==== cpu compact without scan, non-power-of-two ====
   elapsed time: 0.0012ms    (std::chrono Measured)
    [   1   3   3   3   2   2   1   3   3   2   1   3   3 ...   1   3 ]
    passed
==== cpu compact with scan ====
   elapsed time: 0.0237ms    (std::chrono Measured)
    [   1   3   3   3   2   2   1   3   3   2   1   3   3 ...   3   3 ]
    passed
==== work-efficient compact, power-of-two ====
   elapsed time: 1.35984ms    (CUDA Measured)
    passed
==== work-efficient compact, non-power-of-two ====
   elapsed time: 0.607232ms    (CUDA Measured)
    passed
Press any key to continue . . .
```

### Thoughts regarding GPU approach being slower than CPU
I believe that the GPU approach ends up being slower is due two main reasons. One reason is if modulus is being used in order to properly traverse through our array. Or, in other words, to maintain our in-place calculations by not having any conflicts for indicies that are being read/write to by different threads. The modulus logic can be removed and instead a thread can be mapped to our desired index by multiplying the thread index by 2^(offset + 1). The time can be cut down a bit more by using bit shifting rather than a power function. However, I think the biggest reason for the difference in timings is due to the number of threads that are being created but are not being used. Mapping the initial threads to desired indices, rather than it being one-to-one, causes threads that go beyond the number of values to in our array to be not utilized. This will definitely have an impact for larger arrays and higher levels of our tree as there are less values that we care about at that point. This can be optimized by calculating the number of threads needed and only launchiing those threads at each iteration.
