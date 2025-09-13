CUDA Stream Compaction
======================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 2**

* Neha Thumu
  * [LinkedIn](https://www.linkedin.com/in/neha-thumu/)
* Tested on: Windows 11 Pro, i9-13900H @ 2.60GHz 32GB, Nvidia GeForce RTX 4070

### Project 2: Details 

## Implementation
insert here

## Analysis
insert here

## Thoughts regarding GPU approach being slower than CPU
I believe that the GPU approach ends up being slower is due two main reasons. One reason is if modulus is being used in order to properly traverse through our array. Or, in other words, to maintain our in-place calculations by not having any conflicts for indicies that are being read/write to by different threads. The modulus logic can be removed and instead a thread can be mapped to our desired index by multiplying the thread index by 2^(offset + 1). The time can be cut down a bit more by using bit shifting rather than a power function. However, I think the biggest reason for the difference in timings is due to the number of threads that are being created but are not being used. Mapping the initial threads to desired indices, rather than it being one-to-one, causes threads that go beyond the number of values to in our array to be not utilized. This will definitely have an impact for larger arrays and higher levels of our tree as there are less values that we care about at that point. This can be optimized by calculating the number of threads needed and only launchiing those threads at each iteration.
