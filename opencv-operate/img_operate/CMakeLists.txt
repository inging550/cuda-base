cmake_minimum_required(VERSION 3.9)
project(img_operate)
enable_language(CUDA)

set(CMAKE_CXX_STANDART 11)
SET(CMAKE_CUDA_STANDARD 11)

find_package(OpenCV 4 REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
link_libraries(${OpenCV_LIBS})

set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/build)

add_executable(${PROJECT_NAME} trans_rot.cu)
