// 版本最新用法
#include <cuda.h>
#include <cuda_runtime.h>
#include<thrust/device_vector.h>
#include <stdio.h>
// #include <iostream>

__global__ void hanshu(float* output,cudaTextureObject_t texObj,int width,int height, float theta)
{
  unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int xx = x / 1024;
  // unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
  // float u = x / (float)width;
  // float v = y / (float)height;
  // u -= 0.5f;
  // v -= 0.5f;
  // float tu = u * cosf(theta) - v * sinf(theta) + 0.5f;
  // float tv = v * cosf(theta) + u * sinf(theta) + 0.5f;
  // output[y*width + x] = tex2D<float>(texObj, x, y);
	// x += 0.5;
	output[x] = tex1D<float>(texObj, xx);
}

int main()
{
  const int height = 1;
  const int width = 1024;
  float angle = 0.5;
  float *h_data = (float*)std::malloc(sizeof(float)*width*height);
  for(int i=0;i<height*width;i++) h_data[i] = i;
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
  cudaArray_t cuArray;
  cudaMallocArray(&cuArray, &channelDesc, width, height);
  // const size_t spitch = width * sizeof(float);
  // cudaMemcpy2DToArray(cuArray, 0, 0, h_data, spitch, width*sizeof(float), height, cudaMemcpyHostToDevice);
	cudaMemcpyToArray(cuArray, 0, 0, h_data, sizeof(float)*width, cudaMemcpyHostToDevice);
  
	// 设置数据的形式
	struct cudaResourceDesc resDesc;
  memset(&resDesc, 0, sizeof(resDesc));
  resDesc.resType = cudaResourceTypeArray;  // 使用CUDA数组
  resDesc.res.array.array = cuArray;  // 设置为有效的CUDA数组
	// resDesc.resType = cudaResourceTypeLinear; // 使用线性内存
	// resDesc.res.linear.devPtr = // 指向设备上的内存
	// resDesc.res.linear.desc = //描述类型和位长
	// resDesc.res.linear.sizeInBytes = // 占用的字节空间

	// 纹理引用
  struct cudaTextureDesc texDesc;
  memset(&texDesc, 0, sizeof(texDesc));  // 若有些值没有设置就使用默认值
  texDesc.addressMode[0] = cudaAddressModeWrap;  // x坐标处理模式
  // texDesc.addressMode[1] = cudaAddressModeClamp;  // y坐标处理模式
  texDesc.filterMode = cudaFilterModeLinear;  
  texDesc.readMode = cudaReadModeElementType;
  // texDesc.normalizedCoords = 1;  // 是否对坐标进行标准化

	// 定义纹理对象并绑定纹理内存
  cudaTextureObject_t texObj = 0;
  cudaCreateTextureObject(&texObj, &resDesc, &texDesc, NULL);

  float *output;
  cudaMalloc((void**)&output, width*height*sizeof(float));

  dim3 threadsperBlock(16,16);
  dim3 numBlocks((width+threadsperBlock.x-1)/threadsperBlock.x, (height+threadsperBlock.y-1)/threadsperBlock.y);
  hanshu<<<numBlocks, threadsperBlock>>>(output, texObj, width, height, angle);

  cudaMemcpy(h_data, output, width*height*sizeof(float), cudaMemcpyDeviceToHost);
	printf("out[1022]=%f\n", h_data[1]);
	printf("out[1023]=%f\n", h_data[1023]);
	printf("out[555]=%f\n", h_data[555]);
  cudaDestroyTextureObject(texObj);

  cudaFreeArray(cuArray);
  cudaFree(output);
  free(h_data);
  return 0;
}

// // 线性内存绑定纹理内存
// #include <stdio.h>
// #include <cuda.h>
// #include <cuda_runtime.h>

// // 纹理引用
// texture<int, 1> textureRef;

// __global__ void hanshu(int* d_out)
// {
// 	int tid = threadIdx.x;
// 	d_out[tid] = tex1Dfetch(textureRef, int(tid));
// }

// int main()
// {
// 	// 定义数据
// 	int h_in[10],h_out[10];
// 	int* d_out;
// 	int* d_in;
// 	for(int i=0;i<10;i++)
// 	{
// 		h_in[i] = i*2;
// 		h_out[i] = 1;
// 		printf("h_in[%d]:%d\n",i,i*2);
// 	}
// 	cudaMalloc((void**)&d_out, sizeof(int)*10);
// 	cudaMalloc((void**)&d_in, sizeof(int)*10);
// 	cudaMemcpy(d_in, h_in, sizeof(int)*10, cudaMemcpyHostToDevice);
// 	// 绑定纹理内存
// 	cudaBindTexture(NULL, textureRef, d_in, sizeof(int)*10);
// 	// 开始计算
// 	hanshu<<<1,10>>>(d_out);
// 	// 返回计算结果
// 	cudaMemcpy(h_out, d_out, sizeof(int)*10, cudaMemcpyDeviceToHost);
// 	for(int i=0;i<10;i++)
// 	{
// 		printf("d_out[%d]:%d\n",i, h_out[i]);
// 	}
// 	// 后处理
// 	cudaFree(d_out);
// 	cudaUnbindTexture(textureRef);
// 	return 0;
// }


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