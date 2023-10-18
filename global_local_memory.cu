#include <iostream>
#include <stdio.h>
#include <stdio.h>
#include <cuda.h>
#include<cuda_runtime.h>

#define N 5
// cudamalloc分配的为全局内存
//__global__ void gpu_global_memory(int* d_a)
//{
//	d_a[threadIdx.x] = threadIdx.x;
//}

// 本地内存 or 寄存器堆
__global__ void gpu_local_memory(int d_in)
{
	int t_local;  // 定义本地内存/寄存器堆
	t_local = d_in * threadIdx.x;
	printf("%d\n", t_local);
}

int main()
{
	//int h_a[N];
	//int *d_a;
	//cudaMalloc((void**)&d_a, sizeof(int)*N);
	//cudaMemcpy(d_a, h_a, sizeof(int)*N, cudaMemcpyHostToDevice);
	//std::cout << typeid(d_a).name() << std::endl;
	//gpu_global_memory << <1, N >> > (d_a);
	//cudaMemcpy(h_a, d_a, sizeof(int)*N, cudaMemcpyDeviceToHost);
	//for (int i = 0; i < N; i++)
	//{
	//	printf("%d -->%d\n", i, h_a[i]);
	//}

	gpu_local_memory << <1, N >> > (5);
	cudaDeviceSynchronize();
	return 0;
}