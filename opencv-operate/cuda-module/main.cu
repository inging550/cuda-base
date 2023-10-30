/*
任务一：两张图的像素值相加
*/
#include <stdio.h>
#include <iostream>
#include <opencv2/opencv.hpp>
// #include <opencv2/cudacodec.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudawarping.hpp>

int main()
{
  // 定义图像
  cv::Mat h_img1 = cv::imread("../../transformer.png", 1);
  cv::Mat h_img2(h_img1.rows, h_img1.cols, CV_8UC3, cv::Scalar(100,100,100));
  cv::Mat h_result;
  // 在设备（显卡）上创建内存
  cv::cuda::GpuMat d_img1, d_img2, d_result;
  // 上传数据到设备
  d_img1.upload(h_img1);
  d_img2.upload(h_img2);
  // 开始计算
  // cv::cuda::add(d_img1, d_img2, d_result);  // 像素加
  // cv::cuda::subtract(d_img1, d_img2, d_result);  // 像素减
  // cv::cuda::bitwise_not(d_img1, d_result);  // 像素值反转  255-val
  // cv::cuda::cvtColor(d_img1, d_result, cv::COLOR_BGR2HSV);  //  改变颜色空间
  // cv::cuda::threshold(d_img1, d_result, 100, 200, cv::THRESH_BINARY); // 阈值操作
  // cv::cuda::equalizeHist(d_img1, d_result); // 直方图均衡 针对灰度图

  // // 直方图均衡，彩色三通道图
  // std::vector<cv::cuda::GpuMat> vec1;
  // cv::cuda::split(d_result, vec1);
  // cv::cuda::equalizeHist(vec1[0], vec1[0]);
  // cv::cuda::equalizeHist(vec1[1], vec1[1]);
  // cv::cuda::equalizeHist(vec1[2], vec1[2]);
  // cv::cuda::merge(vec1, d_result);

  cv::cuda::resize(d_img1, d_result, cv::Size(200,200), cv::INTER_CUBIC);
  // 返回计算结果
  d_result.download(h_result);
  cv::Vec3b intense = h_result.at<cv::Vec3b>(cv::Point(10, 10));
  std::cout << intense << std::endl;
  cv::imshow("img1", h_img1);
  cv::imshow("img2", h_img2);
  cv::imshow("result1", h_result);
  cv::waitKey(0);
  return 0;
}