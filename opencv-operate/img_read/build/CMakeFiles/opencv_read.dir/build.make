# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.16

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/zjl/桌面/project/CUDA/opencv-operate/img_read

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/zjl/桌面/project/CUDA/opencv-operate/img_read/build

# Include any dependencies generated for this target.
include CMakeFiles/opencv_read.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/opencv_read.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/opencv_read.dir/flags.make

CMakeFiles/opencv_read.dir/img_create.cu.o: CMakeFiles/opencv_read.dir/flags.make
CMakeFiles/opencv_read.dir/img_create.cu.o: ../img_create.cu
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/zjl/桌面/project/CUDA/opencv-operate/img_read/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CUDA object CMakeFiles/opencv_read.dir/img_create.cu.o"
	/usr/local/cuda-11.4/bin/nvcc  $(CUDA_DEFINES) $(CUDA_INCLUDES) $(CUDA_FLAGS) -x cu -c /home/zjl/桌面/project/CUDA/opencv-operate/img_read/img_create.cu -o CMakeFiles/opencv_read.dir/img_create.cu.o

CMakeFiles/opencv_read.dir/img_create.cu.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CUDA source to CMakeFiles/opencv_read.dir/img_create.cu.i"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_PREPROCESSED_SOURCE

CMakeFiles/opencv_read.dir/img_create.cu.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CUDA source to assembly CMakeFiles/opencv_read.dir/img_create.cu.s"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_ASSEMBLY_SOURCE

# Object files for target opencv_read
opencv_read_OBJECTS = \
"CMakeFiles/opencv_read.dir/img_create.cu.o"

# External object files for target opencv_read
opencv_read_EXTERNAL_OBJECTS =

opencv_read: CMakeFiles/opencv_read.dir/img_create.cu.o
opencv_read: CMakeFiles/opencv_read.dir/build.make
opencv_read: /usr/local/lib/libopencv_gapi.so.4.5.5
opencv_read: /usr/local/lib/libopencv_highgui.so.4.5.5
opencv_read: /usr/local/lib/libopencv_ml.so.4.5.5
opencv_read: /usr/local/lib/libopencv_objdetect.so.4.5.5
opencv_read: /usr/local/lib/libopencv_photo.so.4.5.5
opencv_read: /usr/local/lib/libopencv_stitching.so.4.5.5
opencv_read: /usr/local/lib/libopencv_video.so.4.5.5
opencv_read: /usr/local/lib/libopencv_videoio.so.4.5.5
opencv_read: /usr/local/lib/libopencv_imgcodecs.so.4.5.5
opencv_read: /usr/local/lib/libopencv_dnn.so.4.5.5
opencv_read: /usr/local/lib/libopencv_calib3d.so.4.5.5
opencv_read: /usr/local/lib/libopencv_features2d.so.4.5.5
opencv_read: /usr/local/lib/libopencv_flann.so.4.5.5
opencv_read: /usr/local/lib/libopencv_imgproc.so.4.5.5
opencv_read: /usr/local/lib/libopencv_core.so.4.5.5
opencv_read: CMakeFiles/opencv_read.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/zjl/桌面/project/CUDA/opencv-operate/img_read/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CUDA executable opencv_read"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/opencv_read.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/opencv_read.dir/build: opencv_read

.PHONY : CMakeFiles/opencv_read.dir/build

CMakeFiles/opencv_read.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/opencv_read.dir/cmake_clean.cmake
.PHONY : CMakeFiles/opencv_read.dir/clean

CMakeFiles/opencv_read.dir/depend:
	cd /home/zjl/桌面/project/CUDA/opencv-operate/img_read/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/zjl/桌面/project/CUDA/opencv-operate/img_read /home/zjl/桌面/project/CUDA/opencv-operate/img_read /home/zjl/桌面/project/CUDA/opencv-operate/img_read/build /home/zjl/桌面/project/CUDA/opencv-operate/img_read/build /home/zjl/桌面/project/CUDA/opencv-operate/img_read/build/CMakeFiles/opencv_read.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/opencv_read.dir/depend

