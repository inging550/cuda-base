cmake_minimum_required(VERSION 3.9)  # 版本要求
project(create_img)  # 设置项目名
enable_language(CUDA)  # 支持CUDA编程
# 设置C++以及CUDA语言标准
set(CMAKE_CXX_STANDARD 11)  
set(CMAKE_CUDA_STANDARD 11)

# 添加opencv库
find_package(OpenCV 4 REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
message(STATUS "include:${OpenCV_INCLUDE_DIRS}")
link_libraries(${OpenCV_LIBS})

# 可执行文件输出路径
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/build)

# 支持 cuda-gdb 以及 gdb 调试
# add_compile_options(-g -G) 

# 获取路径下所有的文件，并赋值到变量
aux_source_directory(. STC_LIST)

# 添加执行依赖文件
add_executable(${PROJECT_NAME} img_create.cu)
# add_executable(${PROJECT_NAME} img_read.cu)