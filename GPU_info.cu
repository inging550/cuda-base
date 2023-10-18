#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>

__global__ void hanshu()
{
    printf("%d %d", threadIdx.x, blockIdx.x);
}

int main()
{
    cudaDeviceProp dp;
    cudaGetDeviceProperties(&dp, 0);
    printf("\nDevice %d: \"%s\"\n", 0, dp.name);
    printf("Total amount of global memory: %.0f MBytes\n", (float)dp.totalGlobalMem);
    printf("(%d) Multiprocessors\n", dp.multiProcessorCount);
    printf("GPU Max Clock rate: %.0f MHz (%0.2f GHz)\n", dp.clockRate*1e-3f, dp.clockRate*1e-6f);
    printf("显存频率：%.0f  MHz 显存位宽：%d bit \n", dp.memoryClockRate*1e-3f, dp.memoryBusWidth);
    if(dp.l2CacheSize)
    {
        printf("L2 Cache Size : %d bytes\n", dp.l2CacheSize);
    }
    printf("Constant memory : %lu bytes\n", dp.totalConstMem);
    printf("shared memory per block:%lu bytes\n",dp.sharedMemPerBlock);
    printf("number of registers(寄存器) available per block: %d\n", dp.regsPerBlock);
    printf("Maximum number of threads per multiprocessor: %d\n", dp.maxThreadsPerMultiProcessor);
    printf("Maximum number of threads per block: %d\n", dp.maxThreadsPerBlock);
    printf("单个 block 中 Thread 的最大维度(x,y,z): (%d, %d, %d)\n", dp.maxThreadsDim[0],
                                                    dp.maxThreadsDim[1],
                                                    dp.maxThreadsDim[2]);
    printf("单个 grid 中 block 的最大维度(x,y,z),(%d, %d, %d)\n",dp.maxGridSize[0],
                                                dp.maxGridSize[1],
                                                dp.maxGridSize[2]);
    return 0;
}