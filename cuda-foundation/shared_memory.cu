#include <iostream>
#include <stdio.h>
#include <stdio.h>
#include <cuda.h>
#include<cuda_runtime.h>

__global__ void gpu_share_memory(float *d_a)
{
	int index = threadIdx.x;
	float average, sum = 0.0f;
	__shared__ float sh_arr[10];
	sh_arr[index] = d_a[index];
	__syncthreads();  // 确保赋值完成再进行计算
	for (int i = 0; i <= index; i++)
	{
		sum += sh_arr[i];
	}
	average = sum / (index + 1.0f);
	d_a[index] = average;
}


int main()
{
	float h_a[10];
	float *d_a;
	for (int i = 0; i < 10; i++)
	{
		h_a[i] = i;
	}
	cudaMalloc((void**)&d_a, sizeof(float) * 10);
	cudaMemcpy((void*)d_a, (void*)h_a, sizeof(float) * 10, cudaMemcpyHostToDevice);
	// 开始计算
	cudaEvent_t e_start, e_end; // 建立事件对象
	cudaEventCreate(&e_start); // 建立事件
	cudaEventCreate(&e_end);
	cudaEventRecord(e_start, 0);  // 记录时间戳

	gpu_share_memory << <1, 10 >> > (d_a);

	cudaDeviceSynchronize();  // 等待核函数执行完毕
	cudaEventRecord(e_end, 0);  // 等待记录命令执行完毕
	cudaEventSynchronize(e_end);  // 记录时间戳
	float Time;
	cudaEventElapsedTime(&Time, e_start, e_end);  // 计算时间
	printf("Time:%f ms\n", Time);
	cudaMemcpy((void*)h_a, (void*)d_a, sizeof(float)*10, cudaMemcpyDeviceToHost);
	for (int i = 0; i < 10; i++)
	{
		printf("%d--->%f\n", i, h_a[i]);
	}
	return 0;
}
