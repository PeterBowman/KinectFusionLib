// This header contains globally used data types
// Namespace kinectfusion contains those meant for external use
// Internal structures are located in kinectfusion::internal
// Author: Christian Diller, git@christian-diller.de

#ifndef KINECTFUSION_DATA_TYPES_H
#define KINECTFUSION_DATA_TYPES_H

#include <cmath> // std::pow
#include <vector>
#include <opencv2/core/core.hpp>

namespace kinectfusion {
    /**
     *
     * \brief The camera intrinsics
     *
     * This structure stores the intrinsic camera parameters.
     *
     * Consists of:
     * 1) Image width and height,
     * 2) focal length in x and y direction and
     * 3) The principal point in x and y direction
     *
     */
    struct CameraParameters {
        int image_width, image_height;
        float focal_x, focal_y;
        float principal_x, principal_y;

        /**
         * Returns camera parameters for a specified pyramid level; each level corresponds to a scaling of pow(.5, level)
         * @param level The pyramid level to get the parameters for with 0 being the non-scaled version,
         * higher levels correspond to smaller spatial size
         * @return A CameraParameters structure containing the scaled values
         */
        CameraParameters level(const size_t level) const
        {
            if (level == 0) return *this;

            const float scale_factor = std::pow(0.5f, static_cast<float>(level));
            return CameraParameters { image_width >> level, image_height >> level,
                                      focal_x * scale_factor, focal_y * scale_factor,
                                      (principal_x + 0.5f) * scale_factor - 0.5f,
                                      (principal_y + 0.5f) * scale_factor - 0.5f };
        }
    };

    /**
     *
     * \brief Representation of a cloud of three-dimensional points (vertices)
     *
     * This data structure contains
     * (1) the world coordinates of the vertices,
     * (2) the corresponding normals and
     * (3) the corresponding RGB color value
     *
     * - vertices: A 1 x buffer_size opencv matrix with CV_32FC3 values, representing the coordinates
     * - normals: A 1 x buffer_size opencv matrix with CV_32FC3 values, representing the normal direction
     * - color: A 1 x buffer_size opencv matrix with CV_8UC3 values, representing the RGB color
     *
     * Same indices represent the same point
     *
     * The total number of valid points in those buffers is stored in num_points
     *
     */
    struct PointCloud {
        // World coordinates of all vertices
        cv::Mat vertices;
        // Normal directions
        cv::Mat normals;
        // RGB color values
        cv::Mat color;

        // Total number of valid points
        int num_points;
    };

    /**
     *
     * \brief Representation of a dense surface mesh
     *
     * This data structure contains
     * (1) the mesh triangles (triangular faces) and
     * (2) the colors of the corresponding vertices
     *
     * - triangles: A 1 x num_vertices opencv matrix with CV_32FC3 values, representing the coordinates of one vertex;
     *              a sequence of three vertices represents one triangle
     * - colors: A 1 x num_vertices opencv matrix with CV_8Uc3 values, representing the RGB color of each vertex
     *
     * Same indices represent the same point
     *
     * Total number of vertices stored in num_vertices, total number of triangles in num_triangles
     *
     */
    struct SurfaceMesh {
        // Triangular faces
        cv::Mat triangles;
        // Colors of the vertices
        cv::Mat colors;

        // Total number of vertices
        int num_vertices;
        // Total number of triangles
        int num_triangles;
    };

    /**
     *
     * \brief The global configuration
     *
     * This data structure contains several parameters that control the workflow of the overall pipeline.
     * Most of them are based on the KinectFusion paper
     *
     * For an explanation of a specific parameter, see the corresponding comment
     *
     * The struct is preset with some default values so that you can use the configuration without modification.
     * However, you will probably want to adjust most of them to your specific use case.
     *
     * Spatial parameters are always represented in millimeters (mm).
     *
     */
    struct GlobalConfiguration {
        // The overall size of the volume (in mm). Will be allocated on the GPU and is thus limited by the amount of
        // storage you have available.
        int volume_size_x { 512 };
        int volume_size_y { 512 };
        int volume_size_z { 512 };

        // The amount of mm one single voxel will represent in each dimension. Controls the resolution of the volume.
        float voxel_scale { 2.f };

        // Parameters for the Bilateral Filter, applied to incoming depth frames.
        // Directly passed to cv::cuda::bilateralFilter(...); for further information, have a look at the opencv docs.
        int bfilter_kernel_size { 5 };
        float bfilter_color_sigma { 1.f };
        float bfilter_spatial_sigma { 1.f };

        // The initial distance of the camera from the volume center along the z-axis (in mm)
        float init_depth { 1000.f };

        // Downloads the model frame for each frame (for visualization purposes). If this is set to true, you can
        // retrieve the frame with Pipeline::get_last_model_frame()
        bool use_output_frame { true };

        // The truncation distance for both updating and raycasting the TSDF volume
        float truncation_distance { 25.f };

        // The distance (in mm) after which to set the depth in incoming depth frames to 0.
        // Can be used to separate an object you want to scan from the background
        float depth_cutoff_distance { 1000.f };

        // The number of pyramid levels to generate for each frame, including the original frame level
        int num_levels { 3 };

        // The maximum buffer size for exporting triangles; adjust if you run out of memory when exporting
        int triangles_buffer_size { 3 * 2000000 };
        // The maximum buffer size for exporting pointclouds; adjust if you run out of memory when exporting
        int pointcloud_buffer_size { 3 * 2000000 };

        // ICP configuration
        // The distance threshold (as described in the paper) in mm
        float distance_threshold { 10.f };
        // The angle threshold (as described in the paper) in degrees
        float angle_threshold { 20.f };
        // Number of ICP iterations for each level from original level 0 to highest scaled level (sparse to coarse)
        std::vector<int> icp_iterations {10, 5, 4};
    };
}

#endif // KINECTFUSION_DATA_TYPES_H
