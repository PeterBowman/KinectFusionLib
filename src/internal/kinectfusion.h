// Author: Christian Diller, git@christian-diller.de

#ifndef KINECTFUSION_INTERNAL_H
#define KINECTFUSION_INTERNAL_H

#include "./data_types.h"

#include <cstdlib> // std::size_t
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/core/cuda.hpp>
#include <Eigen/Core>

namespace kinectfusion {
    namespace internal {
        /*
         * Step 1: SURFACE MEASUREMENT
         * Compute vertex and normal maps and their pyramids
         */
        FrameData surface_measurement(const cv::Mat_<float>& input_frame,
                                      const CameraParameters& camera_params,
                                      const std::size_t num_levels, const float depth_cutoff,
                                      const int kernel_size, const float color_sigma, const float spatial_sigma);


        /*
         * Step 2: POSE ESTIMATION
         * Use ICP with measured depth and predicted surface to localize camera
         */
        bool pose_estimation(Eigen::Matrix4f& pose,
                             const FrameData& frame_data,
                             const ModelData& model_data,
                             const CameraParameters& cam_params,
                             const int pyramid_height,
                             const float distance_threshold, const float angle_threshold,
                             const std::vector<int>& iterations);

        namespace cuda {
            /*
             * Step 3: SURFACE RECONSTRUCTION
             * Integration of surface measurements into a global volume
             */
            void surface_reconstruction(const cv::cuda::GpuMat& depth_image,
                                        const cv::cuda::GpuMat& color_image,
                                        VolumeData& volume,
                                        const CameraParameters& cam_params,
                                        const float truncation_distance,
                                        const Eigen::Matrix4f& model_view);


            /*
             * Step 4: SURFACE PREDICTION
             * Raycast volume in order to compute a surface prediction
             */
            void surface_prediction(const VolumeData& volume,
                                    cv::cuda::GpuMat& model_vertex,
                                    cv::cuda::GpuMat& model_normal,
                                    cv::cuda::GpuMat& model_color,
                                    const CameraParameters& cam_parameters,
                                    const float truncation_distance,
                                    const Eigen::Matrix4f& pose);

            PointCloud extract_points(const VolumeData& volume, const int buffer_size);

            SurfaceMesh marching_cubes(const VolumeData& volume, const int triangles_buffer_size);
        }
    }
}

#endif // KINECTFUSION_INTERNAL_H
