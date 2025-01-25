## change log
### 01. EigenConvolutionKernel3D
```
a new shared memory area is added to store the weights of the convolution kernel. each thread is responsible for loading a portion of the weights from global memory to shared memory.

<font color="blue">#pragma unroll</font> directive is used to unroll the innermost loop of the convolution calculation, reducing the loop control overhead.
```