CUDA Stream Compaction
======================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 2**

* Neha Thumu
  * [LinkedIn](https://www.linkedin.com/in/neha-thumu/)
* Tested on: Windows 11 Pro, i9-13900H @ 2.60GHz 32GB, Nvidia GeForce RTX 4070

## Project 2: Details 

### Implementation
CPU (baseline): //

Naive: //

Work-Efficient: //

Thrust: //

### Analysis
#### Optimal block size for each method

#### Comparison of methods

#### Phenomena Thoughts

#### Output of test program (for block size of 128)
```
****************
** SCAN TESTS **
****************
    [  23   2   0  42   0   4  35  21  25  10  13  35  29 ...  34   0 ]
==== cpu scan, power-of-two ====
   elapsed time: 0.0024ms    (std::chrono Measured)
    [   0  23  25  25  67  67  71 106 127 152 162 175 210 ... 5813 5847 ]
==== cpu scan, non-power-of-two ====
   elapsed time: 0.0022ms    (std::chrono Measured)
    [   0  23  25  25  67  67  71 106 127 152 162 175 210 ... 5717 5764 ]
    passed
==== naive scan, power-of-two ====
   elapsed time: 1.77357ms    (CUDA Measured)
    passed
==== naive scan, non-power-of-two ====
   elapsed time: 0.256832ms    (CUDA Measured)
    passed
==== work-efficient scan, power-of-two ====
   elapsed time: 3.00138ms    (CUDA Measured)
    passed
==== work-efficient scan, non-power-of-two ====
   elapsed time: 1.33837ms    (CUDA Measured)
    passed
==== thrust scan, power-of-two ====
   elapsed time: 65.5063ms    (CUDA Measured)
    passed
==== thrust scan, non-power-of-two ====
   elapsed time: 0.985312ms    (CUDA Measured)
    passed

*****************************
** STREAM COMPACTION TESTS **
*****************************
    [   1   2   0   2   2   2   3   3   3   2   1   1   1 ...   0   0 ]
==== cpu compact without scan, power-of-two ====
   elapsed time: 0.0054ms    (std::chrono Measured)
    [   1   2   2   2   2   3   3   3   2   1   1   1   1 ...   1   1 ]
    passed
==== cpu compact without scan, non-power-of-two ====
   elapsed time: 0.0128ms    (std::chrono Measured)
    [   1   2   2   2   2   3   3   3   2   1   1   1   1 ...   1   1 ]
    passed
==== cpu compact with scan ====
   elapsed time: 0.0357ms    (std::chrono Measured)
    [   1   2   2   2   2   3   3   3   2   1   1   1   1 ...   1   1 ]
    passed
==== work-efficient compact, power-of-two ====
   elapsed time: 3.27565ms    (CUDA Measured)
    passed
==== work-efficient compact, non-power-of-two ====
   elapsed time: 0.958464ms    (CUDA Measured)
    passed
Press any key to continue . . .
```

### Thoughts regarding GPU approach being slower than CPU
I believe that the GPU approach ends up being slower is due two main reasons. One reason is if modulus is being used in order to properly traverse through our array. Or, in other words, to maintain our in-place calculations by not having any conflicts for indicies that are being read/write to by different threads. The modulus logic can be removed and instead a thread can be mapped to our desired index by multiplying the thread index by 2^(offset + 1). The time can be cut down a bit more by using bit shifting rather than a power function. However, I think the biggest reason for the difference in timings is due to the number of threads that are being created but are not being used. Mapping the initial threads to desired indices, rather than it being one-to-one, causes threads that go beyond the number of values to in our array to be not utilized. This will definitely have an impact for larger arrays and higher levels of our tree as there are less values that we care about at that point. This can be optimized by calculating the number of threads needed and only launchiing those threads at each iteration.
