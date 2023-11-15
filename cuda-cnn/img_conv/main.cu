#include <opencv2/opencv.hpp>
#include <cuda.h>
#include <cuda_runtime.h>
#include <opencv2/cudaarithm.hpp>
#include <iostream>
#include <vector>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <malloc.h>
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
                if(row>=padding && col>=padding && (row<in_wh+padding) && (col<in_wh+padding))
                  out_img[C*(out_wh*out_wh)+ROW*out_wh+COL] += in_img[c*(in_wh*in_wh)+(row-padding)*in_wh+(col-padding)] * kernel_info[C*(kernel_size*kernel_size)+i*kernel_size+j];
              }
  printf("计算结束->%f %f\n", out_img[575], out_img[0]);
}
                

// __global__ void naive_conv()
// {

// }

// __global__ void tile_conv()
// {

// }

// __global__ void gemm_conv()
// {

// }

// __global__ void img2col_conv()
// {

// }

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
  cv::resize(img, img, cv::Size(24, 24), cv::INTER_CUBIC);
  // std::cout<<"【C风格】\n"<<format(img,cv::Formatter::FMT_PYTHON)<<std::endl;
  for(int i=0;i<2;i++)
  {
    for(int j=0;j<2;j++)
    {
      std::cout << img.at<cv::Vec3b>(i, j) << std::endl;
    }
  }

  std::vector<cv::Mat> c_vec;
  cv::split(img, c_vec);
  std::vector<float> vec;
  vec.reserve(24*24*3);
  thrust::device_vector<float> in_vecimg, out_vecimg(24*24*4, 0);
  in_vecimg.reserve(24*24*3);
  // out_vecimg.reserve(22*22*4);
  // 将图片的格式转为NCWH格式，并存入device_vector  (再核函数中作隐式转换)
  for(int i=0; i<3; i++)
  {
    std::vector<float> temp(c_vec[i].reshape(0, 1));
    in_vecimg.insert(in_vecimg.end(), temp.begin(), temp.end());
  }
  // 定义卷积核，假设为size=3*3，输出通道为4
  thrust::device_vector<float> kernel(36, 1);
  // 开始计算
  float* in_img = thrust::raw_pointer_cast(in_vecimg.data());
  float* out_img = thrust::raw_pointer_cast(out_vecimg.data());
  float* kernel_info = thrust::raw_pointer_cast(kernel.data());
  cpu_conv<<<1,1>>>(in_img, out_img, kernel_info, 3, 1, 1, 3, 4, 24);
  // 返回计算结果
  float h_out[24*24*4];
  cudaMemcpy(h_out, out_img, sizeof(float)*24*24*4, cudaMemcpyDeviceToHost);
  printf("计算结束->%f %f\n", h_out[575], h_out[0]);
  return 0;
}