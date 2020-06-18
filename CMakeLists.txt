cmake_minimum_required(VERSION 3.10 FATAL_ERROR)

project(KinectFusionLib LANGUAGES CXX CUDA)

# Setting output paths
set(LIBRARY_OUTPUT_PATH ${CMAKE_BINARY_DIR})

find_package(CUDA 8.0 REQUIRED)

# Optional: Specify the arch of your CUDA hardware here
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-O3;-std=c++11 -gencode arch=compute_52,code=sm_52)

find_package(OpenCV 3.0 REQUIRED COMPONENTS opencv_core
                                            opencv_cudaimgproc
                                            opencv_cudawarping)

find_package(Eigen3 3.3 REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Weffc++")

add_library(KinectFusion STATIC)

target_sources(KinectFusion PRIVATE ${CMAKE_SOURCE_DIR}/include/kinectfusion.h
                                    ${CMAKE_SOURCE_DIR}/include/data_types.h
                                    ${CMAKE_SOURCE_DIR}/src/kinectfusion.cpp
                                    ${CMAKE_SOURCE_DIR}/src/pose_estimation.cpp
                                    ${CMAKE_SOURCE_DIR}/src/surface_measurement.cpp
                                    ${CMAKE_SOURCE_DIR}/src/cuda/include/common.h
                                    ${CMAKE_SOURCE_DIR}/src/cuda/include/mc_tables.h
                                    ${CMAKE_SOURCE_DIR}/src/cuda/extract_pointcloud.cu
                                    ${CMAKE_SOURCE_DIR}/src/cuda/marching_cubes.cu
                                    ${CMAKE_SOURCE_DIR}/src/cuda/pose_estimation.cu
                                    ${CMAKE_SOURCE_DIR}/src/cuda/surface_measurement.cu
                                    ${CMAKE_SOURCE_DIR}/src/cuda/surface_prediction.cu
                                    ${CMAKE_SOURCE_DIR}/src/cuda/surface_reconstruction.cu)

target_include_directories(KinectFusion PUBLIC ${CMAKE_SOURCE_DIR}/include
                                               ${CMAKE_SOURCE_DIR}/src/cuda/include
                                               ${CUDA_INCLUDE_DIRS})

target_link_libraries(KinectFusion opencv_core
                                   opencv_cudaimgproc
                                   opencv_cudawarping
                                   Eigen3::Eigen)

target_compile_features(KinectFusion PUBLIC cxx_std_14)
