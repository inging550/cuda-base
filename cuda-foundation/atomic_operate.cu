#include <iostream>
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define NUM_THREADS 1000
#define SIZE 10

#define BLOCK_WIDTH 100

__global__ void gpu_increment_atomic(int *d_a)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	tid = tid % 10;
	atomicAdd(&d_a[tid], 1); // 原子加
	// d_a[tid] += 1;
}

int main()
{
	int h_a[SIZE];
	const int ARRAY_BYTES = SIZE * sizeof(int);
	int *d_a;
	cudaMalloc((void**)&d_a, ARRAY_BYTES);
	cudaMemset((void*)d_a, 0, ARRAY_BYTES);
	// 原子操作
	gpu_increment_atomic << <1000, BLOCK_WIDTH >> > (d_a);
	cudaMemcpy(h_a, d_a, ARRAY_BYTES, cudaMemcpyDeviceToHost);
	for (int i = 0; i < SIZE; i++)
	{
		printf("index : %d ---> %d times \n", i, h_a[i]);
	}
	cudaFree(d_a);
	return 0;
}