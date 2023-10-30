/*
任务：创建一张图并编辑它
子任务：画线，画框，画圆，写字
*/
#include <cuda.h>
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>
#include <iostream>
using namespace std;


int main()
{
  // Create
  cv::Mat img(255, 255, CV_8UC3, cv::Scalar(255,0,0)); // 高度 宽度 3通道8位无符号整数图像 像素初始化为255,0,0

  // Edit
  cv::line(img, cv::Point(0,0), cv::Point(511,511),cv::Scalar(0,255,0), 2);  // 画线
  cv::rectangle(img, cv::Point(20,20), cv::Point(200,200), cv::Scalar(255,255,0), 5);  // 画框
  cv::circle(img, cv::Point(100,100), 100, cv::Scalar(255,255,0), 3); //画圆
  cv::putText(img, "Opencv", cv::Point(10,100), cv::FONT_HERSHEY_SCRIPT_SIMPLEX, 3, cv::Scalar(255,255,255), 5, 8);  // 写字

  // show
  cv::String win_name = "blank img";
  cv::namedWindow(win_name);
  cv::imshow(win_name, img);
  cv::waitKey(0);
  cv::destroyWindow(win_name);
  return 0;
}