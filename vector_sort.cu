// 使用共享内存实现数组的排序
/*
任务一：当线程数够用的情况，即d_a可全部放入共享内存
任务二：线程数不够用的情况，即d_a需要分批次放入共享内存
*/
#include<stdio.h>
#include<cuda.h>
#include<cuda_runtime.h>

// 任务一
__global__ void sort(int* d_a, int* d_out)
{
  int tid = threadIdx.x;
  __shared__ int sh_a[5];
  sh_a[tid] = d_a[tid];
  __syncthreads();
  int count = 0;
  for(int i=0;i<5;i++)
  {
    if(sh_a[tid]>sh_a[i]) ++count;
  }
  d_out[count] = sh_a[tid];
}
// 任务二
__global__ void sort2(int* d_a, int* d_out)
{
  int tid = threadIdx.x + blockDim.x * blockIdx.x;  // 线程的全局唯一ID
  int index = threadIdx.x; // 线程在本块中的ID
  int val = d_a[tid], count = 0;
  __shared__ int sh_a[5];
  for(int i=index;i<5;i+=3)
  {
    sh_a[index] = d_a[i];
    __syncthreads();
    for(int j=0;j<blockDim.x;j++)
    {
      if(val>sh_a[j]) ++count;
      __syncthreads();
    }
  }
  d_out[count] = val;
}
int main()
{
  // 定义数据
  int h_a[5] = {5,2,7,3,0};
  int h_out[5];
  int *d_a, *d_out;
  // 分配内存
  cudaMalloc((void**)&d_a, sizeof(int)*5);
  cudaMalloc((void**)&d_out, sizeof(int)*5);
  cudaMemcpy(d_a, h_a, sizeof(int)*5, cudaMemcpyHostToDevice);
  // 开始计算
  // sort<<<1,5>>>(d_a, d_out);
  sort2<<<2,3>>>(d_a, d_out);
  // 返回计算结果
  cudaMemcpy(h_out, d_out, sizeof(int)*5, cudaMemcpyDeviceToHost);
  for(int i=0; i<5; i++)
  {
    printf("%d  ", h_out[i]);
  }
  printf("\n");
  // 释放内存
  cudaFree(d_a);
  cudaFree(d_out);
  return 0;
}