#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
// 实现 y = a * x + b

// 场外定义常量内存
__constant__ int const_a=5;
__constant__ int const_b=2;

__global__ void hanshu(int* d_a)
{
	int tid = threadIdx.x;
	d_a[tid] = d_a[tid] * const_a + const_b;
}
int main()
{
	// 常量赋值 或者直接在定义的时候赋值
	// int hc_a=5,hc_b=2;
	// cudaMemcpyToSymbol(const_a,&hc_a,sizeof(int));
	// cudaMemcpyToSymbol(const_b,&hc_b,sizeof(int),0,cudaMemcpyHostToDevice);
  // 数组赋值
	int h_a[5];
	int* d_a;
	for(int i=0;i<5;i++)
	{
		h_a[i]=i;
		// printf("h_a[%d]:%d\n",i,i);
	}
	cudaMalloc((void**)&d_a,sizeof(int)*5);
	cudaMemcpy(d_a,h_a,sizeof(int)*5,cudaMemcpyHostToDevice);
	hanshu<<<1,5>>>(d_a);
	cudaMemcpy(h_a,d_a,sizeof(int)*5,cudaMemcpyDeviceToHost);
	// 打印计算结果并释放内存
	for(int i=0;i<5;i++)
	{
		printf("new[%d]=%d\n",i,h_a[i]);
	}
	cudaFree(d_a);
	return 0;
}