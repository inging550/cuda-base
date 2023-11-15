cmake_minimum_required(VERSION 3.9)
project(conv)
enable_language(CUDA)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CUDA_STANDARD 14)

find_package(OpenCV 4 REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
link_libraries(${OpenCV_LIBS})


set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/build)
add_compile_options(-g -G)
add_executable(${PROJECT_NAME} main.cu)