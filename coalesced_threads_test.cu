#include <cooperative_groups.h>
#include <stdio.h>
#include "cuda.h"
#include "cuda_runtime.h"
#include <iostream>

using namespace cooperative_groups;

// Basic reduction code found in the presentation; going to test on a variety of
// thread groups 

__device__ void coalesced_thread_ids(coalesced_group threads)
{
        printf("Thread ID: %d Block ID: %d Coalesced Thread Rank: %d \n", threadIdx.x, blockIdx.x, threads.thread_rank());
}

// use this kernel to get appropriate threads ??
__global__ void descriminator_kernel()
{
        // first block and first warp in block 
        if (threadIdx.x % 5 == 0)
        {
                coalesced_group active_threads = coalesced_threads();
                coalesced_thread_ids(active_threads);
        }

}

int  main(void)
{
        // Setup timers to test different configurations
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        cudaEventRecord(start); // start timer

        // run the kernel
        descriminator_kernel<<<2, 64>>>();

        cudaEventRecord(stop); // stop timing

        // get the runtime 
        cudaEventSynchronize(stop);
        float milliseconds = 0.0;
        cudaEventElapsedTime(&milliseconds, start, stop);

        // print the runtime
        std::cout << milliseconds << " milliseconds for parallel run" << std::endl;

        return 0;
}
