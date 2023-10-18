// 矩阵乘积
/*
任务一：实现两个4*4的方阵相乘
任务二：实现4*6与6*4的矩阵相乘
任务三：实现任意维度矩阵相乘

子任务一：使用全局内存
子任务二：使用共享内存
*/

#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>

// 分配了 4*4 的线程
__global__ void mission1_global(int* d_a, int* d_b, int* d_result)
{
  int row = threadIdx.y;  // 第几行
  int col = threadIdx.x;  // 第几列
  /*
  若是分配的为2*2的block以及2*2的thread，则
  int row = blockIdx.y*2 + threadIdx.y;
  int col = blockIdx.x*2 + threadIdx.x;
  其余不变
  */
  for(int k=0; k<4; k++)
  {
    // d_a第row行所有元素与d_b第col列所有元素作点乘
    d_result[row*4+col] += d_a[row*4+k] * d_b[k*4+col];
  }
}

// 分配4*4的线程
__global__ void mission1_shared(int* d_a, int* d_b, int* d_result)
{
  int row = threadIdx.y;
  int col = threadIdx.x;
  __shared__ int sh_a[4][4];  // 定义共享内存
  __shared__ int sh_b[4][4];
  for(int i=0; i<4; i++)
  {
    for(int j=0; j<4; j++)
    {
      sh_a[i][j] = d_a[i*4+j];  // 将数据放进共享内存中
      sh_b[i][j] = d_b[i*4+j];
      __syncthreads();
    }
  }
  for(int i=0; i<4; i++)
  {
    // sh_a第row行所有元素与sh_b第col列所有元素作点乘
    d_result[row*4+col] += sh_a[row][i] * sh_b[i][col];
  }
}

// 分配 2*2->block  2*2->thread
__global__ void mission1_shared1(int* d_a, int* d_b, int* d_result)
{
  int row = blockIdx.y*2 + threadIdx.y;
  int col = blockIdx.x*2 + threadIdx.x;
  __shared__ int sh_a[2][2], sh_b[2][2];
  for(int i=0; i<2; i++)
  {
    // 先计算四个block中的加号前的内容（i=0时），然后计算加号后的内容(i=1时)
    sh_a[threadIdx.y][threadIdx.x] = d_a[row*4 + i*2 + threadIdx.x];
    sh_b[threadIdx.y][threadIdx.x] = d_b[(threadIdx.y + i*2)*4 + col];
    __syncthreads();
    for(int j=0; j<2; j++)
    {
      // 子矩阵是一个2*2矩阵之间的计算
      d_result[row*4 + col] += sh_a[threadIdx.y][j] * sh_b[j][threadIdx.x];
    }
  }
}

// 分配线程4*4
__global__ void mission2_global(int* d_a, int* d_b, int* d_result)
{
  int row = threadIdx.y;
  int col = threadIdx.x;
  for(int j=0; j<6; j++)
  {
    // d_a第row行所有元素与d_b第col列所有元素作点乘
    d_result[row*4 + col] += d_a[row*6 + j] * d_b[j*4 + col]; 
  }
}

// 分配内存  block->2*2  thread->3*3
// 版本一
__global__ void mission2_shared(int* d_a, int* d_b, int* d_result)
{
  __shared__ int sh_a[3][3], sh_b[3][3];
  int row = blockIdx.y*2 + threadIdx.y;
  int col = blockIdx.x*2 + threadIdx.x;
  for(int i=0; i<2; i++)
  {
    // 先计算四个block中的加号前的内容（i=0时），然后计算加号后的内容(i=1时)
    sh_a[threadIdx.y][threadIdx.x] = d_a[row*6 + i*3 + threadIdx.x];
    sh_b[threadIdx.y][threadIdx.x] = d_b[col + (threadIdx.y + i*3)*4]; 
    __syncthreads();
    for(int j=0; j<3; j++)
    {
      // 向量点乘
      d_result[row*4 + col] += sh_a[threadIdx.y][j] * sh_b[j][threadIdx.x];
    }
  }
}

//  block->2*2  thread->3*3
// 版本二运行速度可能稍慢，但便于理解
__global__ void mission2_shared1(int* d_a, int* d_b, int* d_result)
{
  __shared__ int sh_a[2][3], sh_b[3][2];
  int row = blockIdx.y*2 + threadIdx.y;
  int col = blockIdx.x*2 + threadIdx.x;
  for(int i=0; i<2; i++)
  {
    // 线程多余的部分就不管他了
    if(threadIdx.y<2) sh_a[threadIdx.y][threadIdx.x] = d_a[row*6 + i*3 + threadIdx.x];
    if(threadIdx.x<2) sh_b[threadIdx.y][threadIdx.x] = d_b[col + (threadIdx.y + i*3)*4]; 
    __syncthreads();
    for(int j=0; j<3; j++)
    { 
      if(threadIdx.x<2 && threadIdx.y<2) d_result[row*4 + col] += sh_a[threadIdx.y][j] * sh_b[j][threadIdx.x];
    }
  }
}

// 任务三 [2,6]*[6,3]->[2,3]     block->1*1  thread->3*2
__global__ void mission3_global(int* d_a, int* d_b, int* d_result)
{
  int row = threadIdx.y;
  int col = threadIdx.x;
  for(int i=0; i<6; i++)
  {
    d_result[row*3 + col] += d_a[row*6 + i] * d_b[i*3 + col];
  }
}

// 任务三 [2,6]*[6,3]->[2,3]   block->1*1  thread->6*6
__global__ void mission3_shared(int* d_a, int* d_b, int* d_result)
{
  int row = threadIdx.y;
  int col = threadIdx.x;
  __shared__ int sh_a[6][6], sh_b[6][6];
  sh_a[row][col] = d_a[row*6 + col];
  sh_b[row][col] = d_b[row*3 + col];
  __syncthreads();
  for(int i=0; i<6; i++)
  {
    // 只在规定范围内（result矩阵的最大尺寸下）计算
    if(row<2 && col<3) d_result[row*3 + col] += sh_a[row][i] * sh_b[i][col];
  }
}

int main()
{
  /*
  任务一
  */
  printf("----------------\n");
  printf("任务一\n");
	int *d_a, *d_b, *d_result;
	int h_a[4][4], h_b[4][4], h_result[4][4];
	//初始化两个矩阵
	for (int i = 0; i < 4; i++)
	{
		for (int j = 0; j < 4; j++)
		{
			h_a[i][j] = i;
			h_b[i][j] = j;
		}
	}
  cudaMalloc((void**)&d_a, sizeof(int)*4*4);
  cudaMalloc((void**)&d_b, sizeof(int)*4*4);
  cudaMalloc((void**)&d_result, sizeof(int)*4*4);
  cudaMemcpy(d_a, h_a, sizeof(int)*4*4, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, h_b, sizeof(int)*4*4, cudaMemcpyHostToDevice);
  // 开始运算
  // mission1_global<<<1,dim3(4,4,1)>>>(d_a, d_b, d_result);
  // mission1_shared<<<1,dim3(4,4,1)>>>(d_a, d_b, d_result);
  mission1_shared1<<<dim3(2,2,1),dim3(2,3,1)>>>(d_a, d_b, d_result);
  // 返回计算结果
  cudaMemcpy(h_result, d_result, sizeof(int)*4*4, cudaMemcpyDeviceToHost);
  for(int i=0; i<4; i++)
  {
    for(int k=0; k<4; k++)
    {
      printf("h_result[%d][%d]:%d ", i, k, h_result[i][k]);
    }
    printf("\n");
  }
  /*
  任务二
  */
  printf("----------------\n");
  printf("任务二 \n");
  //  定义需要用到的数据
  int h_a1[4][6], h_b1[6][4], h_result1[4][4];
  int *d_a1, *d_b1, *d_result1;
  for(int i=0; i<4; i++)
  {
    for(int j=0; j<6; j++)
    {
      h_a1[i][j] = i;
    }
  }
  for(int i=0; i<6; i++)
  {
    for(int j=0; j<4; j++)
    {
      h_b1[i][j] = j;
    }
  }
  cudaMalloc((void**)&d_a1, sizeof(int)*4*6);
  cudaMalloc((void**)&d_b1, sizeof(int)*6*4);
  cudaMalloc((void**)&d_result1, sizeof(int)*4*4);
  cudaMemcpy(d_a1, h_a1, sizeof(int)*4*6, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b1, h_b1, sizeof(int)*6*4, cudaMemcpyHostToDevice);
  // 开始计算
  // mission2_global<<<1,dim3(4,4,1)>>>(d_a1, d_b1, d_result1);
  mission2_shared<<<dim3(2,2,1),dim3(3,3,1)>>>(d_a1, d_b1, d_result1);
  // mission2_shared1<<<dim3(2,2,1),dim3(3,3,1)>>>(d_a1, d_b1, d_result1);
  // 返回计算结果
  cudaMemcpy(h_result1, d_result1, sizeof(int)*4*4, cudaMemcpyDeviceToHost);
  for(int i=0; i<4; i++)
  {
    for(int j=0; j<4; j++)
    {
      printf("h_result1[%d][%d]:%d ", i, j, h_result1[i][j]);
    }
    printf("\n");
  }

  /*
  任务三
  */
  printf("----------------\n");
  printf("任务三 \n");
  // 定义要用到的数据
  int *d_a2, *d_b2, *d_result2;
  int h_a2[2][6], h_b2[6][3], h_result2[2][3];
  for(int i=0; i<2; i++)
  {
    for(int j=0; j<6; j++)
    {
      // printf("h_a2[%d][%d]:%d ", i, j, i);
      h_a2[i][j] = i;
    }
    // printf("\n");
  }
  for(int i=0; i<6; i++)
  {
    for(int j=0; j<3; j++)
    {
      // printf("h_b2[%d][%d]:%d ", i, j, j);
      h_b2[i][j] = j;
    }
    // printf("\n");
  }
  cudaMalloc((void**)&d_a2, sizeof(int)*2*6);
  cudaMalloc((void**)&d_b2, sizeof(int)*6*3);
  cudaMalloc((void**)&d_result2, sizeof(int)*2*3);
  cudaMemcpy(d_a2, h_a2, sizeof(int)*2*6, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b2, h_b2, sizeof(int)*6*3, cudaMemcpyHostToDevice);
  // 开始计算
  // mission3_global<<<1,dim3(3,2,1)>>>(d_a2, d_b2, d_result2);
  mission3_shared<<<1,dim3(6,6,1)>>>(d_a2, d_b2, d_result2);
  // 返回计算结果
  cudaMemcpy(h_result2, d_result2, sizeof(int)*2*3, cudaMemcpyDeviceToHost);
  for(int i=0; i<2; i++)
  {
    for(int j=0; j<3; j++)
    {
      printf("h_result2[%d][%d]:%d ", i, j, h_result2[i][j]);
    }
    printf("\n");
  }
  // 程序结束 释放内存
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_result);
  cudaFree(d_a1);
  cudaFree(d_b1);
  cudaFree(d_result1);
  cudaFree(d_a2);
  cudaFree(d_b2);
  cudaFree(d_result2);
  return 0;
}

