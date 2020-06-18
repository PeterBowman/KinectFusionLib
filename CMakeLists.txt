cmake_minimum_required(VERSION 3.10 FATAL_ERROR)

project(KinectFusionLib LANGUAGES CXX CUDA)

# Setting output paths
set(LIBRARY_OUTPUT_PATH ${CMAKE_BINARY_DIR})

# ------------------------------------------------
#                 Dependencies
# ------------------------------------------------
## CUDA
find_package(CUDA 8.0 REQUIRED)

# Optional: Specify the arch of your CUDA hardware here
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-O3;-std=c++11 -gencode arch=compute_52,code=sm_52)

## OpenCV
find_package(OpenCV 3.0 REQUIRED COMPONENTS opencv_core
                                            opencv_cudaimgproc
                                            opencv_cudawarping)

## Eigen3
find_package(Eigen3 3.3 REQUIRED)
# ------------------------------------------------

set(CUDA_PROPAGATE_HOST_FLAGS OFF)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Weffc++")

# Targets
set(PROJECT_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/include)
set(PROJECT_SOURCE_DIR ${PROJECT_SOURCE_DIR}/src)

include_directories("${PROJECT_INCLUDE_DIR}" "${PROJECT_SOURCE_DIR}/cuda/include")
file(GLOB KinectFusion_SRCS "${PROJECT_SOURCE_DIR}/cuda/*.cu" "${PROJECT_SOURCE_DIR}/*.cpp")

add_library(KinectFusion STATIC "${KinectFusion_SRCS}")

target_include_directories(KinectFusion PUBLIC ${CUDA_INCLUDE_DIRS})

target_link_libraries(KinectFusion opencv_core
                                   opencv_cudaimgproc
                                   opencv_cudawarping
                                   Eigen3::Eigen)

target_compile_features(KinectFusion PUBLIC cxx_std_14)
