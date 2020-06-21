cmake_minimum_required(VERSION 3.0 FATAL_ERROR)

project(KinectFusionLib LANGUAGES CXX)

# Use modern C++, set compiler flags
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Weffc++")

# Setting output paths
set(LIBRARY_OUTPUT_PATH ${CMAKE_BINARY_DIR})

# ------------------------------------------------
#                 Dependencies
# ------------------------------------------------
find_package(CUDA 8.0 REQUIRED REQUIRED)
set(CUDA_PROPAGATE_HOST_FLAGS OFF)
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS};-O3;-std=c++11 -gencode arch=compute_52,code=sm_52")

find_package(OpenCV 3.0 REQUIRED COMPONENTS opencv_core
                                            opencv_cudaimgproc
                                            opencv_cudawarping)

find_package(Eigen3 3.3 REQUIRED)
# ------------------------------------------------

set(_header_dir ${CMAKE_SOURCE_DIR}/include)
set(_sources_dir ${CMAKE_SOURCE_DIR}/src)

file(GLOB _sources "${_sources_dir}/cuda/*.cu" "${_sources_dir}/*.cpp")

cuda_add_library(KinectFusion STATIC "${_sources}")

target_include_directories(KinectFusion PUBLIC ${_header_dir}
                                               ${_sources_dir}/cuda/include
                                               ${CUDA_INCLUDE_DIRS})

target_link_libraries(KinectFusion opencv_core
                                   opencv_cudaimgproc
                                   opencv_cudawarping
                                   Eigen3::Eigen)
