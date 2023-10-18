// 使用CUDA流实现矩阵加法
/*
任务：
使用两个CUDA流实现矩阵加法，每个流负责一半的计算
stream1负责前半段计算
stream2负责后半段计算
*/
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>

#define N 512

__global__ void gpuadd(int* d_a, int* d_b, int* d_c)
{
  int tid = threadIdx.x + blockDim.x * blockIdx.x;
  d_c[tid] = d_a[tid] + d_b[tid];
}


int main()
{
  // 定义数据
  int *h_a, *h_b, *h_c;
  int *d_a1, *d_b1, *d_c1;
  int *d_a2, *d_b2, *d_c2;
  // 创建CUDA流
  cudaStream_t stream2, stream1;  // 创建流对象
  cudaStreamCreate(&stream2);  // 创建流
  cudaStreamCreate(&stream1);
  // CUDA流作数据传输时必须用固定内存
  cudaHostAlloc((void**)&h_a, 2*N*sizeof(int), cudaHostAllocDefault);
  cudaHostAlloc((void**)&h_b, 2*N*sizeof(int), cudaHostAllocDefault);
  cudaHostAlloc((void**)&h_c, 2*N*sizeof(int), cudaHostAllocDefault);
  // 再设备上分配内存
  cudaMalloc((void**)&d_a1, N*sizeof(int));
  cudaMalloc((void**)&d_a2, N*sizeof(int));
  cudaMalloc((void**)&d_b1, N*sizeof(int));
  cudaMalloc((void**)&d_b2, N*sizeof(int));
  cudaMalloc((void**)&d_c1, N*sizeof(int));
  cudaMalloc((void**)&d_c2, N*sizeof(int));
  // 主机内存赋值
  for(int i=0; i<N*2; i++)
  {
    h_a[i] = 2*i*i;
    h_b[i] = i;
  }
  // 计算时间(包括数据传输事件)
  cudaEvent_t start, end;
  cudaEventCreate(&start);
  cudaEventCreate(&end);
  cudaEventRecord(start, 0);
  // 将数据传输操作和内核执行操作加入对列
  cudaMemcpyAsync(d_a1, h_a, N*sizeof(int), cudaMemcpyHostToDevice, stream1);
  cudaMemcpyAsync(d_a2, h_a+N, N*sizeof(int), cudaMemcpyHostToDevice, stream2);
  cudaMemcpyAsync(d_b1, h_b, N*sizeof(int), cudaMemcpyHostToDevice, stream1);
  cudaMemcpyAsync(d_b2, h_b+N, N*sizeof(int), cudaMemcpyHostToDevice, stream2);
  // 核函数执行
  gpuadd<<<1,512,0,stream1>>>(d_a1, d_b1, d_c1);
  gpuadd<<<1,512,0,stream2>>>(d_a2, d_b2, d_c2);
  // 返回计算结果
  cudaMemcpyAsync(h_c, d_c1, N*sizeof(int), cudaMemcpyDeviceToHost, stream1);
  cudaMemcpyAsync(h_c+N, d_c2, N*sizeof(int), cudaMemcpyDeviceToHost, stream2);
  cudaStreamSynchronize(stream1);
  cudaStreamSynchronize(stream2);
  // 计算时间
  cudaEventRecord(end, 0);
  cudaEventSynchronize(end);
  float Time;
  cudaEventElapsedTime(&Time, start, end);
  printf("计算+传输耗时 %f ms\n", Time);
  // 等待流同步后打印结果
  int correct = 1;
  for(int i=0; i<2*N; i++)
  {
    if(h_a[i]+h_b[i]!=h_c[i]) correct=0;
  }
  if(correct==1) printf("GPU compute is correct\n");
  else printf("Error\n");
  // 释放内存
  cudaStreamDestroy(stream1);
  cudaStreamDestroy(stream2);
  cudaFree(d_a1);
  cudaFree(d_b1);
  cudaFree(d_c1);
  cudaFree(d_a1);
  cudaFree(d_b1);
  cudaFree(d_c1);
  cudaFreeHost(h_a);
  cudaFreeHost(h_b);
  cudaFreeHost(h_c);
  return 0;
}
