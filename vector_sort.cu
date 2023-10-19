// 使用共享内存实现数组的排序
/*
任务一：当线程数够用的情况，即d_a可全部放入共享内存
任务二：线程数不够用的情况，即d_a需要分批次放入共享内存
子任务一：实现数组长度为偶数时的排序（特殊情况）
子任务二：数组长度为奇数时的排序（一般情况）
*/
#include<stdio.h>
#include<cuda.h>
#include<cuda_runtime.h>

#define N 5  // 数组长度
#define threadNum 3  // 每个块的线程数

// 任务一（代码适用与子任务一）
__global__ void sort(int* d_a, int* d_out)
{
  int tid = threadIdx.x;
  __shared__ int sh_a[N];
  sh_a[tid] = d_a[tid];
  __syncthreads();
  int count = 0;
  for(int i=0;i<N;i++)
  {
    if(sh_a[tid]>sh_a[i]) ++count;
  }
  d_out[count] = sh_a[tid];
}
// 任务二（代码适用与子任务二，即一般情况），稍复杂
__global__ void sort2(int* d_a, int* d_out)
{
  int tid = threadIdx.x + blockDim.x * blockIdx.x;  // 线程的全局唯一ID
  int index = threadIdx.x; // 线程在本块中的ID
  int val = d_a[tid], count = 0, res=N;
  __shared__ int sh_a[threadNum];
  for(int i=0;i<(N/threadNum+1);i++)
  {
    // 将数组分段，然后逐段计算大于val的个数
    sh_a[index] = d_a[index + i*threadNum];
    __syncthreads();
    int time = (res>=threadNum) ? threadNum : res;
    for(int j=0;j<time;j++)
    {
      if(val>sh_a[j]) ++count;
      __syncthreads();
    }
    res -= threadNum;
  }
  d_out[count] = val;
}
int main()
{
  // 定义数据
  int h_a[N] = {2,9,7,4,8};
  int h_out[N];
  int *d_a, *d_out;
  // 分配内存
  cudaMalloc((void**)&d_a, sizeof(int)*N);
  cudaMalloc((void**)&d_out, sizeof(int)*N);
  cudaMemcpy(d_a, h_a, sizeof(int)*N, cudaMemcpyHostToDevice);
  // 开始计算
  // sort<<<1,N>>>(d_a, d_out);
  sort2<<<(N/threadNum+1),threadNum>>>(d_a, d_out);
  // 返回计算结果
  cudaMemcpy(h_out, d_out, sizeof(int)*N, cudaMemcpyDeviceToHost);
  for(int i=0; i<N; i++)
  {
    printf("%d  ", h_out[i]);
  }
  printf("\n");
  // 释放内存
  cudaFree(d_a);
  cudaFree(d_out);
  return 0;
}