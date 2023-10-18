// 归约运算：原始输入是两个数组，但是输出却缩减为单一值的运算
/*
任务：计算1024维向量的内积
方法1：分配1024个线程
方法2：分配512个线程
方法3：使用纹理内存分配1024个线程
*/
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>

// 方法1：1024个线程
__global__ void function1(float* d_a, float* d_b, float* d_c)
{
	int tid = threadIdx.x;
	__shared__ float sh_sum[1024];
	sh_sum[tid] = d_a[tid] * d_b[tid];
	__syncthreads();
	int i = blockDim.x / 2; // 512
	while(i)
	{
		if(tid<i) sh_sum[tid] += sh_sum[tid+i];
			__syncthreads();
			i/=2;
	}
	if(tid == 0)
	{
		*d_c = sh_sum[0];
	}
}

// 方法2：512个线程
__global__ void function2(float* d_a, float* d_b, float* d_c)
{
	int index = threadIdx.x;
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	__shared__ float sh_sum[512];
	sh_sum[index] = d_a[tid] * d_b[tid];
	__syncthreads();

	int i = blockDim.x / 2; // 256
	while(i)
	{
		if(index<i) sh_sum[index] += sh_sum[i+index];
		__syncthreads();
		i /= 2;
	}
	if(index==0) d_c[blockIdx.x] = sh_sum[0];
}

// 方法3：纹理内存
texture<float, 1> textureRef1;
texture<float, 1> textureRef2;
__global__ void function3(float* d_e)
{
	int tid = threadIdx.x;
	__shared__ float sh_sum[1024];
	sh_sum[tid] = tex1D(textureRef1, float(tid)) * tex1D(textureRef2, float(tid));
	__syncthreads();

	int i = blockDim.x / 2;
	while(i)
	{
		if(tid < i) sh_sum[tid] += sh_sum[tid+i];
		__syncthreads();
		i /= 2;
	}
	if(tid==0) *d_e = sh_sum[0];
}

int main()
{
	// 初始化参数
	float *h_a, *h_b, *h_c, *h_e;
	float h_d[2];
	h_a = (float*)malloc(sizeof(float)*1024);
	h_b = (float*)malloc(sizeof(float)*1024);
	h_c = (float*)malloc(sizeof(float));
	float *d_a, *d_b, *d_c, *d_d, *d_e;
	for(int i=0; i<1024; i++)
	{ 
		h_a[i] = i;
		h_b[i] = 2;
	}
	// 方法3，建立CUDA数组并绑定
	cudaArray *cu_a, *cu_b;
	cudaMallocArray(&cu_a, &textureRef1.channelDesc, 1024, 1);
	cudaMallocArray(&cu_b, &textureRef2.channelDesc, 1024, 1);
	cudaMemcpyToArray(cu_a, 0, 0, h_a, sizeof(float)*1024, cudaMemcpyHostToDevice);
	cudaMemcpyToArray(cu_b, 0, 0, h_b, sizeof(float)*1024, cudaMemcpyHostToDevice);
	cudaBindTextureToArray(textureRef1, cu_a);
	cudaBindTextureToArray(textureRef2, cu_b);
	cudaMalloc((void**)&d_e, sizeof(float));
	clock_t start = clock();
	function3<<<1, 1024>>>(d_e);
	clock_t end = clock();
	double time = (double)(end-start) / CLOCKS_PER_SEC;
	cudaMemcpy(h_e, d_e, sizeof(float), cudaMemcpyDeviceToHost);
	printf("the function3's sum is %f --- use time %f s \n", *h_e, time);
	cudaUnbindTexture(textureRef1);
	cudaUnbindTexture(textureRef2);
	cudaFreeArray(cu_a);
	cudaFreeArray(cu_b);
	// 给方法1，2分配内存
	cudaMalloc((void**)&d_a, sizeof(float)*1024);
	cudaMalloc((void**)&d_b, sizeof(float)*1024);
	cudaMalloc((void**)&d_c, sizeof(float));
	cudaMalloc((void**)&d_d, sizeof(float)*2);
	cudaMemcpy(d_a, h_a, sizeof(float)*1024, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, h_b, sizeof(float)*1024, cudaMemcpyHostToDevice);
	// 计算方法1
	start = clock();
	function1<<<1, 1024>>>(d_a, d_b, d_c);
	end = clock();
	time = (double)(end-start) / CLOCKS_PER_SEC;
	cudaMemcpy(h_c, d_c, sizeof(float), cudaMemcpyDeviceToHost);
	printf("the function1's sum is %f ---- use time %f s \n", *h_c, time);
	// 计算方法2
	start = clock();
	function2<<<2, 512>>>(d_a, d_b, d_d);
	end = clock();
	time = (double)(end-start) / CLOCKS_PER_SEC;
	cudaMemcpy(h_d, d_d, sizeof(float)*2, cudaMemcpyDeviceToHost);
	printf("the function2's sum is %f ---- use time %f s\n", (h_d[0]+h_d[1]), time);
	// 后处理
	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);
	cudaFree(d_d);
	cudaFree(d_e);
	free(h_c);
	return 0;
}
/*
总结 ：
方法3：使用的纹理内存最满
方法2稍快于方法1，因为方法2单个块只计算了512维但是加上CPU端的整合两者速度应该相差无几
*/