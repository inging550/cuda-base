#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define N 5

__global__ void vector_add(int* d_a, int* d_b, int* d_c)
{
  int thread_id = blockIdx.x;
  d_c[thread_id] = d_a[thread_id] + d_b[thread_id];
}

int main()
{
  // 定义并赋初值
  int h_a[N], h_b[N], h_c[N];
  int *d_a, *d_b, *d_c;
  for(int i=0;i<N;i++)
  {
    h_a[i] = i;
    h_b[i] = i*2;
    printf("h_a[%d]:%d-------h_b[%d]:%d \n", i, i, i, i*2);
  }
  // GPU内存配置
  cudaMalloc((void**)&d_a, sizeof(int)*N);
  cudaMalloc((void**)&d_b, sizeof(int)*N);
  cudaMalloc((void**)&d_c, sizeof(int)*N);
  cudaMemcpy(d_a, &h_a, sizeof(int)*N, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, &h_b, sizeof(int)*N, cudaMemcpyHostToDevice);
  // 开始计算
  clock_t start_time = clock();
  vector_add<<<N, 1>>>(d_a, d_b, d_c);
  cudaDeviceSynchronize();
  clock_t end_time = clock();
  double time_d = (double)(end_time-start_time)/CLOCKS_PER_SEC;
  printf("run time is %f seconds \n", time_d);
  // 返回计算结果
  cudaMemcpy(&h_c, d_c, sizeof(int)*N, cudaMemcpyDeviceToHost);
  // 打印计算结果
  for(int i=0;i<N;i++)
  {
    printf("h_c[%d]:%d  \n", i, h_c[i]);
  }
  // 释放内存
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);
  return 0;
}