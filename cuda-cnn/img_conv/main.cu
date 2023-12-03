#include <opencv2/opencv.hpp>
#include <cuda.h>
#include <cuda_runtime.h>
#include <opencv2/cudaarithm.hpp>
#include <iostream>
#include <vector>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <malloc.h>
#include <math.h>

// 使用单个线程模拟CPU端的卷积计算
__global__ void cpu_conv(float* in_img, float* out_img, float* kernel_info, int kernel_size, int stride, int padding, int in_channel,
                          int out_channel, int in_wh)
{
  // 先确定输出图像的尺寸,(默认img为方形图片)
  printf("开始计算\n");
  int out_wh = (in_wh + 2*padding - kernel_size) / stride + 1;
  printf("%d\n", out_wh);
  // 开始卷积运算
  for(int C=0; C<out_channel; C++)
    for(int ROW=0; ROW<out_wh; ROW++)
      for(int COL=0; COL<out_wh; COL++)
        for(int c=0; c<in_channel; c++)
            for(int i=0; i<kernel_size; i++)
              for(int j=0; j<kernel_size; j++)
              {
                int row = ROW*stride+i;
                int col = COL*stride+j;
                if(row>=padding && col>=padding && row<(in_wh+padding) && col<(in_wh+padding))
                  out_img[C*(out_wh*out_wh)+ROW*out_wh+COL] += in_img[c*(in_wh*in_wh)+(row-padding)*in_wh+(col-padding)] * kernel_info[C*(kernel_size*kernel_size)+i*kernel_size+j];
              }
}
                
// naive卷积，每个线程负责outimg的一个位置的计算
__global__ void naive_conv(float* in_img, float* out_img, float* kernel_info, int kernel_size, int stride, int padding, int in_channel, int in_wh)
{
  int out_wh = (in_wh + 2*padding - kernel_size) / stride + 1;
  int gridnum = ceilf(float(out_wh)/32); // 一个方向几个小块
  if(gridnum==0) gridnum=1;
  int h = (blockIdx.y / gridnum)*32 + threadIdx.y; // 第几行
  int w = (blockIdx.y % gridnum)*32 + threadIdx.x;  // 第几列
  float result = 0;
  for(int c=0; c<in_channel; c++)
    for(int i=0; i<kernel_size; i++)
      for(int j=0; j<kernel_size; j++)
      {
        if(h<out_wh && w<out_wh)
        {
          int row = h*stride + i;  // 对应目前in_img的哪一行
          int col = w*stride + j;  // 哪一列
          if(row>=padding && col>=padding && row<(in_wh+padding) && col<(in_wh+padding))
          result += in_img[c*in_wh*in_wh+(row-padding)*in_wh+col-padding] * 
          kernel_info[blockIdx.x*kernel_size*kernel_size+i*kernel_size+j];
        } 
      }
  if(w<out_wh && h<out_wh)
  {
    out_img[blockIdx.x*out_wh*out_wh+h*out_wh+w] = result;
  }
    
}

// 与之前类似但是加入共享内存
extern __shared__ float total_info[];
__global__ void tile_conv(float* in_img, float* out_img, float* kernel_info, int kernel_size, int stride, int padding, int in_channel, int in_wh)
{
  int out_wh = (in_wh + 2*padding - kernel_size) / stride + 1;
  // 因为一个块管理32*32个像素，所以此范围映射的输入大小为：
  int scope = 32*stride + kernel_size - 1;  
  // 此处存放范围内的in_img 以及 kernel_info
  // __shared__ float total_info[66*66+3*3]; // scope*scope + kernel_size*kernel_size
  float* share_scope_img = (float*)&total_info[0];
  float* share_kernel_info = (float*)&total_info[scope*scope];

  int gridnum = ceilf(float(out_wh) / 32); // 一个方向分几块
  if(gridnum==0) gridnum=1;
  int y_grid = blockIdx.z / gridnum; // y方向第几个块
  int x_grid = blockIdx.z % gridnum; // x方向第几个块
  int y = y_grid*32 + threadIdx.y;  // 对应out_img的第几行
  int x = x_grid*32 + threadIdx.x;  // 第几列
  float result = 0;
  // 开始计算
  for(int c=0; c<in_channel; c++)
  {
    // 对卷积核赋值(卷积核尺寸小于32)
    if(threadIdx.y<kernel_size && threadIdx.x<kernel_size)
      share_kernel_info[threadIdx.y*kernel_size + threadIdx.x] = kernel_info[blockIdx.x*kernel_size*kernel_size + threadIdx.y*kernel_size + threadIdx.x];
    __syncthreads();
    // 对感受野内的in_img赋值，这里可以通过并行进行优化，(暂不考虑padding)
    for(int p=0; p<kernel_size; p++)
      for(int q=0; q<kernel_size; q++)
      {
        // 每个线程赋值参与计算in_img的部分，可能会重复赋值同一区域
        int row = y*stride + p - padding; // 对应in_img的第几行
        int col = x*stride + q - padding;
        if(row>=0 && col>=0 && row<in_wh && col<in_wh)
          share_scope_img[(threadIdx.y*stride+p)*scope+threadIdx.x*stride+q] = in_img[c*in_wh*in_wh+row*in_wh+col]; 
      }
      __syncthreads();
    // 开始卷积
    for(int i=0; i<kernel_size; i++)
    {
      for(int j=0; j<kernel_size; j++)
      {
        int row = threadIdx.y*stride + i;  // 当前块的(x,y)对应sh_scope_img的row行col列
        int col = threadIdx.x*stride + j;
        result += share_scope_img[row*scope+col] * share_kernel_info[i*kernel_size+j];
      }
    }
  }
  if(x<out_wh && y<out_wh)
    out_img[blockIdx.x*out_wh*out_wh+y*out_wh+x] = result;
}

// 此函数不涉及计算
__global__ void img2col()
{

}

// __global__ void fft_conv()
// {

// }

// __global__ void winograd_conv()
// {

// }

int main()
{
  // 先读取待处理的图片，并进行处理
  cv::Mat img = cv::imread("../../cat.jpg", 1);
  cv::resize(img, img, cv::Size(64, 64), cv::INTER_CUBIC);
  // std::cout<<"【C风格】\n"<<format(img,cv::Formatter::FMT_PYTHON)<<std::endl;
  for(int i=0;i<2;i++)
  {
    for(int j=1;j<4;j++)
    {
      std::cout << img.at<cv::Vec3b>(i, j) << std::endl;
    }
  }

  std::vector<cv::Mat> c_vec;
  cv::split(img, c_vec);
  std::vector<float> vec;
  vec.reserve(64*64*3);
  thrust::device_vector<float> in_vecimg, out_vecimg(32*32*4, 0);
  in_vecimg.reserve(64*64*3);
  // out_vecimg.reserve(22*22*4);
  // 将图片的格式转为NCWH格式，并存入device_vector  (再核函数中作隐式转换)
  for(int i=0; i<3; i++)
  {
    std::vector<float> temp(c_vec[i].reshape(0, 1));
    in_vecimg.insert(in_vecimg.end(), temp.begin(), temp.end());
  }
  // 定义卷积核，假设为size=3*3，输出通道为4
  thrust::device_vector<float> kernel(36, 1);
  // 参数设定
  float* in_img = thrust::raw_pointer_cast(in_vecimg.data());
  float* out_img = thrust::raw_pointer_cast(out_vecimg.data());
  float* kernel_info = thrust::raw_pointer_cast(kernel.data());
  int kernel_size=3, stride=2, padding=1, in_channel=3, out_channel=4, in_wh=64;
  int out_wh = (in_wh + 2*padding - kernel_size) / stride + 1;  // 输出img的尺寸
  int bz = pow(ceilf(out_wh/32), 2);  // 图像一共分为几块
  if(bz==0) bz=1;
  // 开始计算
  //// cpu
  // cpu_conv<<<1,1>>>(in_img, out_img, kernel_info, kernel_size, stride, padding, in_channel, out_channel, in_wh);
  // naive_conv
  // naive_conv<<<dim3(out_channel,bz,1),dim3(32,32,1)>>>(in_img, out_img, kernel_info, kernel_size, stride, padding, in_channel, in_wh);
  // tile_conv
  int scope = 32*stride + kernel_size - 1;
  int sh_contain = scope * scope + kernel_size * kernel_size; 
  tile_conv<<<dim3(out_channel,bz,1),dim3(32,32,1),sizeof(float)*sh_contain>>>(in_img, out_img, kernel_info, kernel_size, stride, padding, in_channel, in_wh);
  // img2col + GEMM
  float *d_convert_img, *d_kernel_info;
  cudaMalloc((void**)&d_convert_img, sizeof(float)*out_wh*out_wh*kernel_size*kernel_size);
  cudaMalloc((void**)&d_kernel_info, sizeof(float)*kernel_size*kernel_size);
  // 返回计算结果
  float h_out[32*32*4];
  cudaMemcpy(h_out, out_img, sizeof(float)*32*32*4, cudaMemcpyDeviceToHost);
  printf("计算结束->%f %f\n", h_out[14], h_out[15]);
  return 0;
}