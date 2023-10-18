// 线性内存绑定纹理内存
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>

// 纹理引用
texture<int, 1> textureRef;

__global__ void hanshu(int* d_out)
{
	int tid = threadIdx.x;
	d_out[tid] = tex1Dfetch(textureRef, int(tid));
}

int main()
{
	// 定义数据
	int h_in[10],h_out[10];
	int* d_out;
	int* d_in;
	for(int i=0;i<10;i++)
	{
		h_in[i] = i*2;
		h_out[i] = 1;
		printf("h_in[%d]:%d\n",i,i*2);
	}
	cudaMalloc((void**)&d_out, sizeof(int)*10);
	cudaMalloc((void**)&d_in, sizeof(int)*10);
	cudaMemcpy(d_in, h_in, sizeof(int)*10, cudaMemcpyHostToDevice);
	// 绑定纹理内存
	cudaBindTexture(NULL, textureRef, d_in, sizeof(int)*10);
	// 开始计算
	hanshu<<<1,10>>>(d_out);
	// 返回计算结果
	cudaMemcpy(h_out, d_out, sizeof(int)*10, cudaMemcpyDeviceToHost);
	for(int i=0;i<10;i++)
	{
		printf("d_out[%d]:%d\n",i, h_out[i]);
	}
	// 后处理
	cudaFree(d_out);
	cudaUnbindTexture(textureRef);
	return 0;
}


/*
// CUDA数组绑定纹理内存
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>

//纹理引用
texture<int, 1> textureRef;

//核函数
__global__ void hanshu(int* d_out)
{
	int tid = threadIdx.x;
	d_out[tid] = tex1D(textureRef, int(tid));
}

int main()
{
	// 定义数据并分配内存
	int h_in[10], h_out[10];
	int* d_out;
	for(int i=0;i<10;i++)
	{
		h_in[i] = int(i)+1;
		printf("h_in[%d]:%d\n",i,int(i));
	}
	cudaMalloc((void**)&d_out, sizeof(int)*10);

	// 定义CUDA数组
	cudaChannelFormatDesc desc = cudaCreateChannelDesc<int>();
	cudaArray *cu_Array;
	// cudaMallocArray(&cu_Array, &textureRef.channelDesc, 10, 1);
	cudaMallocArray(&cu_Array, &desc, 10, 1);
	cudaMemcpyToArray(cu_Array, 0, 0, h_in, sizeof(int)*10, cudaMemcpyHostToDevice);

	// 绑定纹理内存
	cudaBindTextureToArray(textureRef, cu_Array);

	// 开始计算
	hanshu<<<1,10>>>(d_out);

	// 返回计算结果
	cudaMemcpy(h_out, d_out, sizeof(int)*10, cudaMemcpyDeviceToHost);
	for(int i=0;i<10;i++)
	{
		printf("d_out[%d]:%d\n",i,h_out[i]);
	}

	// 后处理
	cudaUnbindTexture(textureRef);
	cudaFreeArray(cu_Array);
	cudaFree(d_out);
	return 0;
}
*/