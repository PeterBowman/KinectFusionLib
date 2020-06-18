cmake_minimum_required(VERSION 3.10 FATAL_ERROR)

project(KinectFusionLib LANGUAGES CXX CUDA)

include(GNUInstallDirs)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)

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

set(_headers ${CMAKE_SOURCE_DIR}/include/kinectfusion.h
             ${CMAKE_SOURCE_DIR}/include/data_types.h
             ${CMAKE_SOURCE_DIR}/src/cuda/include/common.h
             ${CMAKE_SOURCE_DIR}/src/cuda/include/mc_tables.h)

set(_sources ${CMAKE_SOURCE_DIR}/src/kinectfusion.cpp
             ${CMAKE_SOURCE_DIR}/src/pose_estimation.cpp
             ${CMAKE_SOURCE_DIR}/src/surface_measurement.cpp
             ${CMAKE_SOURCE_DIR}/src/cuda/extract_pointcloud.cu
             ${CMAKE_SOURCE_DIR}/src/cuda/marching_cubes.cu
             ${CMAKE_SOURCE_DIR}/src/cuda/pose_estimation.cu
             ${CMAKE_SOURCE_DIR}/src/cuda/surface_measurement.cu
             ${CMAKE_SOURCE_DIR}/src/cuda/surface_prediction.cu
             ${CMAKE_SOURCE_DIR}/src/cuda/surface_reconstruction.cu)

add_library(KinectFusion STATIC)

target_sources(KinectFusion PRIVATE ${_headers} ${_sources})

set_target_properties(KinectFusion PROPERTIES PUBLIC_HEADER "${_headers}")

target_include_directories(KinectFusion PUBLIC $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
                                               $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/cuda/include>
                                               $<INSTALL_INTERFACE:{CMAKE_INSTALL_INCLUDE_DIR}>
                                               ${CUDA_INCLUDE_DIRS})

target_link_libraries(KinectFusion PUBLIC opencv_core
                                          opencv_cudaimgproc
                                          opencv_cudawarping
                                          Eigen3::Eigen)

target_compile_features(KinectFusion PUBLIC cxx_std_14)

install(TARGETS KinectFusion
        EXPORT KinectFusionLib
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
        PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

export(PACKAGE KinectFusionLib)

set(_cmake_destination ${CMAKE_INSTALL_LIBDIR}/cmake/KinectFusionLib)

file(WRITE ${CMAKE_BINARY_DIR}/KinectFusionLibConfig.cmake
    "include(\${CMAKE_CURRENT_LIST_DIR}/KinectFusionLibTargets.cmake)")

install(FILES ${CMAKE_BINARY_DIR}/KinectFusionLibConfig.cmake
        DESTINATION ${_cmake_destination})

export(EXPORT KinectFusionLib
       FILE ${CMAKE_BINARY_DIR}/KinectFusionLibTargets.cmake)

install(EXPORT KinectFusionLib
        DESTINATION ${_cmake_destination}
        FILE KinectFusionLibTargets.cmake)

include(AddUninstallTarget)
