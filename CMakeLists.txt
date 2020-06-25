cmake_minimum_required(VERSION 3.7 FATAL_ERROR)

project(KinectFusionLib LANGUAGES CXX)

include(GNUInstallDirs)

# Set compiler flags
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Weffc++")
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
	set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /MT")
	set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /MTd")
endif()

# Pick custom CMake modules
list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)

# Setting output paths
set(LIBRARY_OUTPUT_PATH ${CMAKE_BINARY_DIR})

# ------------------------------------------------
#                 Dependencies
# ------------------------------------------------
find_package(CUDA 8.0 REQUIRED)
set(CUDA_PROPAGATE_HOST_FLAGS OFF)
cuda_select_nvcc_arch_flags(gpu_flags)
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS};-O3;-std=c++11;${gpu_flags}")

find_package(OpenCV 3.0 REQUIRED COMPONENTS opencv_core
                                            opencv_cudaimgproc
                                            opencv_cudawarping)

find_package(Eigen3 3.3 REQUIRED)
# ------------------------------------------------

file(GLOB _public_headers "${CMAKE_SOURCE_DIR}/include/*.h")
file(GLOB _private_headers "${CMAKE_SOURCE_DIR}/src/cuda/include/*.h")
file(GLOB _cxx_sources "${CMAKE_SOURCE_DIR}/src/*.cpp")
file(GLOB _cuda_sources "${CMAKE_SOURCE_DIR}/src/cuda/*.cu")

cuda_add_library(KinectFusion STATIC ${_public_headers}
                                     ${_private_headers}
                                     ${_cxx_sources}
                                     ${_cuda_sources})

set_target_properties(KinectFusion PROPERTIES PUBLIC_HEADER "${_public_headers}")

target_include_directories(KinectFusion PUBLIC $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
                                               $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDE_DIR}>
                                               ${CUDA_INCLUDE_DIRS}
                                        PRIVATE ${CMAKE_SOURCE_DIR}/src/cuda/include)

# We cannot use PUBLIC here, see https://gitlab.kitware.com/cmake/cmake/-/issues/16602
target_link_libraries(KinectFusion opencv_core
                                   opencv_cudaimgproc
                                   opencv_cudawarping
                                   Eigen3::Eigen)

target_compile_options(KinectFusion PUBLIC -std=c++14)

install(TARGETS KinectFusion
        EXPORT KinectFusionLib
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
        PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

export(PACKAGE KinectFusionLib)

set(_cmake_destination ${CMAKE_INSTALL_LIBDIR}/cmake/KinectFusionLib)

configure_file(${CMAKE_SOURCE_DIR}/cmake/templates/KinectFusionLibConfig.cmake.in
               ${CMAKE_BINARY_DIR}/KinectFusionLibConfig.cmake @ONLY)

install(FILES ${CMAKE_BINARY_DIR}/KinectFusionLibConfig.cmake
        DESTINATION ${_cmake_destination})

export(EXPORT KinectFusionLib
       FILE ${CMAKE_BINARY_DIR}/KinectFusionLibTargets.cmake)

install(EXPORT KinectFusionLib
        DESTINATION ${_cmake_destination}
        FILE KinectFusionLibTargets.cmake)

include(AddUninstallTarget)
