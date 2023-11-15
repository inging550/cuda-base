#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudawarping.hpp>

int main()
{
  cv::Mat h_img = cv::imread("../../transformer.png", 1);
  cv::Mat h_result1, h_result2;
  cv::cuda::GpuMat d_img, d_result1, d_result2;
  // 图像平移
  d_img.upload(h_img);
  cv::Mat trans_mat = (cv::Mat_<double>(2,3)<<1,0,70,0,1,50);
  cv::cuda::warpAffine(d_img, d_result1, trans_mat, d_img.size());
  // 图像旋转
  cv::Point2f pt(d_img.cols/2, d_img.rows/2);
  cv::Mat rot_mat = cv::getRotationMatrix2D(pt, 45, 0.5);
  cv::cuda::warpAffine(d_img, d_result2, rot_mat, cv::Size(d_img.cols, d_img.rows));
  // 返回计算结果
  d_result1.download(h_result1);
  d_result2.download(h_result2);
  cv::imshow("平移变换结果", h_result1);
  cv::imshow("旋转变换结果", h_result2);
  cv::waitKey(0);
  return 0;
}
