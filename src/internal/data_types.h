// Author: Christian Diller, git@christian-diller.de

#ifndef KINECTFUSION_DATA_TYPES_INTERNAL_H
#define KINECTFUSION_DATA_TYPES_INTERNAL_H

#include "kinectfusion/data_types.h"

#include <cstdlib> // std::size_t
#include <utility> // std::move
#include <vector>
#include <cuda_runtime.h>
#include <opencv2/core/core.hpp>
#include <opencv2/core/cuda.hpp>

namespace kinectfusion {
    namespace internal {
        /*
         * Contains the internal data representation of one single frame as read by the depth camera
         * Consists of depth, smoothed depth and color pyramids as well as vertex and normal pyramids
         */
        struct FrameData {
            std::vector<cv::cuda::GpuMat> depth_pyramid;
            std::vector<cv::cuda::GpuMat> smoothed_depth_pyramid;
            std::vector<cv::cuda::GpuMat> color_pyramid;

            std::vector<cv::cuda::GpuMat> vertex_pyramid;
            std::vector<cv::cuda::GpuMat> normal_pyramid;

            explicit FrameData(const std::size_t pyramid_height) :
                    depth_pyramid(pyramid_height), smoothed_depth_pyramid(pyramid_height),
                    color_pyramid(pyramid_height), vertex_pyramid(pyramid_height), normal_pyramid(pyramid_height)
            { }

            // No copying
            FrameData(const FrameData&) = delete;
            FrameData& operator=(const FrameData& other) = delete;

            FrameData(FrameData&& data) noexcept :
                    depth_pyramid(std::move(data.depth_pyramid)),
                    smoothed_depth_pyramid(std::move(data.smoothed_depth_pyramid)),
                    color_pyramid(std::move(data.color_pyramid)),
                    vertex_pyramid(std::move(data.vertex_pyramid)),
                    normal_pyramid(std::move(data.normal_pyramid))
            { }

            FrameData& operator=(FrameData&& data) noexcept
            {
                depth_pyramid = std::move(data.depth_pyramid);
                smoothed_depth_pyramid = std::move(data.smoothed_depth_pyramid);
                color_pyramid = std::move(data.color_pyramid);
                vertex_pyramid = std::move(data.vertex_pyramid);
                normal_pyramid = std::move(data.normal_pyramid);
                return *this;
            }
        };

        /*
         * Contains the internal data representation of one single frame as raycast by surface prediction
         * Consists of depth, smoothed depth and color pyramids as well as vertex and normal pyramids
         */
        struct ModelData {
            std::vector<cv::cuda::GpuMat> vertex_pyramid;
            std::vector<cv::cuda::GpuMat> normal_pyramid;
            std::vector<cv::cuda::GpuMat> color_pyramid;

            ModelData(const std::size_t pyramid_height, const CameraParameters camera_parameters) :
                    vertex_pyramid(pyramid_height), normal_pyramid(pyramid_height),
                    color_pyramid(pyramid_height)
            {
                for (std::size_t level = 0; level < pyramid_height; ++level) {
                    vertex_pyramid[level] =
                            cv::cuda::createContinuous(camera_parameters.level(level).image_height,
                                                       camera_parameters.level(level).image_width,
                                                       CV_32FC3);
                    normal_pyramid[level] =
                            cv::cuda::createContinuous(camera_parameters.level(level).image_height,
                                                       camera_parameters.level(level).image_width,
                                                       CV_32FC3);
                    color_pyramid[level] =
                            cv::cuda::createContinuous(camera_parameters.level(level).image_height,
                                                       camera_parameters.level(level).image_width,
                                                       CV_8UC3);
                    vertex_pyramid[level].setTo(0);
                    normal_pyramid[level].setTo(0);
                }
            }

            // No copying
            ModelData(const ModelData&) = delete;
            ModelData& operator=(const ModelData& data) = delete;

            ModelData(ModelData&& data) noexcept :
                    vertex_pyramid(std::move(data.vertex_pyramid)),
                    normal_pyramid(std::move(data.normal_pyramid)),
                    color_pyramid(std::move(data.color_pyramid))
            { }

            ModelData& operator=(ModelData&& data) noexcept
            {
                vertex_pyramid = std::move(data.vertex_pyramid);
                normal_pyramid = std::move(data.normal_pyramid);
                color_pyramid = std::move(data.color_pyramid);
                return *this;
            }
        };

        /*
         *
         * \brief Contains the internal volume representation
         *
         * This internal representation contains two volumes:
         * (1) TSDF volume: The global volume used for depth frame fusion and
         * (2) Color volume: Simple color averaging for colorized vertex output
         *
         * It also contains two important parameters:
         * (1) Volume size: The x, y and z dimensions of the volume (in mm)
         * (2) Voxel scale: The scale of a single voxel (in mm)
         *
         */
        struct VolumeData {
            cv::cuda::GpuMat tsdf_volume; //short2
            cv::cuda::GpuMat color_volume; //uchar4
            int3 volume_size;
            float voxel_scale;

            VolumeData(const int3 _volume_size, const float _voxel_scale) :
                    tsdf_volume(cv::cuda::createContinuous(_volume_size.y * _volume_size.z, _volume_size.x, CV_16SC2)),
                    color_volume(cv::cuda::createContinuous(_volume_size.y * _volume_size.z, _volume_size.x, CV_8UC3)),
                    volume_size(_volume_size), voxel_scale(_voxel_scale)
            {
                tsdf_volume.setTo(0);
                color_volume.setTo(0);
            }
        };

        /*
         *
         * \brief Contains the internal pointcloud representation
         *
         * This is only used for exporting the data kept in the internal volumes
         *
         * It holds GPU containers for vertices, normals and vertex colors
         * It also contains host containers for this data and defines the total number of points
         *
         */
        struct CloudData {
            cv::cuda::GpuMat vertices;
            cv::cuda::GpuMat normals;
            cv::cuda::GpuMat color;

            cv::Mat host_vertices;
            cv::Mat host_normals;
            cv::Mat host_color;

            int* point_num;
            int host_point_num;

            explicit CloudData(const int max_number) :
                    vertices{cv::cuda::createContinuous(1, max_number, CV_32FC3)},
                    normals{cv::cuda::createContinuous(1, max_number, CV_32FC3)},
                    color{cv::cuda::createContinuous(1, max_number, CV_8UC3)},
                    host_vertices{}, host_normals{}, host_color{}, point_num{nullptr}, host_point_num{}
            {
                vertices.setTo(0.f);
                normals.setTo(0.f);
                color.setTo(0.f);

                cudaMalloc(&point_num, sizeof(int));
                cudaMemset(point_num, 0, sizeof(int));
            }

            // No copying
            CloudData(const CloudData&) = delete;
            CloudData& operator=(const CloudData& data) = delete;

            void download()
            {
                vertices.download(host_vertices);
                normals.download(host_normals);
                color.download(host_color);

                cudaMemcpy(&host_point_num, point_num, sizeof(int), cudaMemcpyDeviceToHost);
            }
        };

        /*
         *
         * \brief Contains the internal surface mesh representation
         *
         * This is only used for exporting the data kept in the internal volumes
         *
         * It holds several GPU containers needed for the MarchingCubes algorithm
         *
         */
        struct MeshData {
            cv::cuda::GpuMat occupied_voxel_ids_buffer;
            cv::cuda::GpuMat number_vertices_buffer;
            cv::cuda::GpuMat vertex_offsets_buffer;
            cv::cuda::GpuMat triangle_buffer;

            cv::cuda::GpuMat occupied_voxel_ids;
            cv::cuda::GpuMat number_vertices;
            cv::cuda::GpuMat vertex_offsets;

            explicit MeshData(const int buffer_size):
                    occupied_voxel_ids_buffer{cv::cuda::createContinuous(1, buffer_size, CV_32SC1)},
                    number_vertices_buffer{cv::cuda::createContinuous(1, buffer_size, CV_32SC1)},
                    vertex_offsets_buffer{cv::cuda::createContinuous(1, buffer_size, CV_32SC1)},
                    triangle_buffer{cv::cuda::createContinuous(1, buffer_size * 3, CV_32FC3)},
                    occupied_voxel_ids{}, number_vertices{}, vertex_offsets{}
            { }

            void create_view(const int length)
            {
                occupied_voxel_ids = cv::cuda::GpuMat(1, length, CV_32SC1, occupied_voxel_ids_buffer.ptr<int>(0),
                                                      occupied_voxel_ids_buffer.step);
                number_vertices = cv::cuda::GpuMat(1, length, CV_32SC1, number_vertices_buffer.ptr<int>(0),
                                                   number_vertices_buffer.step);
                vertex_offsets = cv::cuda::GpuMat(1, length, CV_32SC1, vertex_offsets_buffer.ptr<int>(0),
                                                  vertex_offsets_buffer.step);
            }
        };
    }
}

#endif // KINECTFUSION_DATA_TYPES_INTERNAL_H
