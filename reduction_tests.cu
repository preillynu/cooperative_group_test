#include <cooperative_groups.h>
#include <stdio.h>
#include "cuda.h"
#include "cuda_runtime.h"
#include <iostream>
#define ARRAYSIZE 10000 
#define BLOCKSIZE 256

using namespace cooperative_groups;

// Basic reduction code found in the presentation; going to test on a variety of
// thread groups 

__device__ float threadSum(float *x, int elements)
{
        float thread_sum = 0.0;

        int id = blockIdx.x*blockDim.x + threadIdx.x;
        int step = blockDim.x*gridDim.x;

        for (int i = id; i < elements / step; i += step)
        {
                 thread_sum += x[i];
        }

        return thread_sum;

}

template <unsigned size>
__device__ float reduce(thread_block_tile<size> g, float val)
{
        for (int i = g.size() / 2; i > 0; i /= 2) {
                val += g.shfl_down(val, i);
        }

        return val;
}

// use this kernel to get sum
__global__ void sum_kernel(float *x, int elements, float *val)
{
        float thread_sum = threadSum(x, elements);

        thread_block_tile<32> g = tiled_partition<32>(this_thread_block());
        float tile_sum = reduce<32>(g, thread_sum);

        // first block and first warp in block 
        if (g.thread_rank() == 0)
        {
                atomicAdd(val, tile_sum);
        }
}

int  main(void)
{

        float *x, *devX, *devSum; // pointers for local and device arrays

        x = new float[ARRAYSIZE]; // make x the specified size

        float local_sum = 0.0;
        int grid_size = (ARRAYSIZE + BLOCKSIZE - 1)/BLOCKSIZE;

        // compute sum of all array elements and fill x with the data where the ith
        // element is i
        for (int i = 0; i < ARRAYSIZE; i++)
        {
                x[i] = i/1000;
                local_sum += i/1000;
        }

        // create an array on the device that contains X and copy over the data
        cudaMalloc((void**)&devX, ARRAYSIZE*sizeof(float));
        cudaMalloc((void**)&devSum, sizeof(float));
        cudaMemcpy(devX, x, ARRAYSIZE*sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(devSum, &local_sum, sizeof(float), cudaMemcpyHostToDevice);



        // Setup timers to test different configurations
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        cudaEventRecord(start); // start timer

        // run the kernel
        sum_kernel<<<grid_size, BLOCKSIZE>>>(devX, ARRAYSIZE, devSum);

        cudaEventRecord(stop); // stop timing

        // get the runtime 
        cudaEventSynchronize(stop);
        float milliseconds = 0.0;
        cudaEventElapsedTime(&milliseconds, start, stop);

        float device_sum = 0.0;
        cudaMemcpy(&device_sum, devSum, sizeof(float), cudaMemcpyDeviceToHost);        // print the runtime
        std::cout << milliseconds << " milliseconds for parallel run" << std::endl;
        std::cout << "Host sum: " << local_sum << std::endl;
        std::cout << "Device sum: " << device_sum << std::endl;
        // free memory
        cudaFree(devX);
        cudaFree(devSum);
        delete[] x;

        return 0;
}
