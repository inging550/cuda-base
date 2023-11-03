#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/cudafilters.hpp>

int main()
{
  // 读取原始数据
  cv::Mat h_img = cv::imread("../../transformer.png", 0);
  cv::Mat h_result3x3, h_result5x5, h_result7x7;
  cv::cuda::GpuMat d_img,d_result3x3,d_result5x5,d_result7x7;
  d_img.upload(h_img);
  // 开始计算
  cv::Ptr<cv::cuda::Filter> filter3x3,filter5x5,filter7x7;
  filter3x3 = cv::cuda::createBoxFilter(CV_8UC1, CV_8UC1, cv::Size(3, 3));
  filter3x3->apply(d_img, d_result3x3);
  filter5x5 = cv::cuda::createBoxFilter(CV_8UC1, CV_8UC1, cv::Size(5, 5));
  filter5x5->apply(d_img, d_result5x5);
  filter7x7 = cv::cuda::createBoxFilter(CV_8UC1, CV_8UC1, cv::Size(7, 7));
  filter7x7->apply(d_img, d_result7x7);
  // 返回计算结果
  d_result3x3.download(h_result3x3);
  d_result5x5.download(h_result5x5);
  d_result7x7.download(h_result7x7);
  // 结果展示
  cv::imshow("h_img", h_img);
  cv::imshow("3x3", h_result3x3);
  cv::imshow("5x5", h_result5x5);
  cv::imshow("7x7", h_result7x7);
  cv::waitKey();
  return 0;
}
