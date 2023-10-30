/*
任务打开一张图片
*/
#include <opencv2/opencv.hpp>
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
using namespace cv;
using namespace std;

int main()
{
  Mat img;
  img = imread("../../transformer.png", 1);
  if(img.empty())
  {
    cout << "Coule not open an image" << endl;
    return -1;
  }
  String win_name = "My First Opencv Program";
  namedWindow(win_name);
  imshow(win_name, img);
  waitKey(0);  // 程序暂停并等待用户输入
  destroyWindow(win_name);
  return 0;
}