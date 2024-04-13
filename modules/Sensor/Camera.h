//
// Created by whitby on 8/22/23.
//

#ifndef MONO_ORB_SLAM3_CAMERA_H
#define MONO_ORB_SLAM3_CAMERA_H

#include <Eigen/Core>
#include <opencv2/core/core.hpp>
#include <utility>

namespace mono_orb_slam3 {

    class Camera {
    private:
        Camera(int w, int h, int fps, cv::Mat matK, cv::Mat distCoeffs);

        // camera pointer
        static Camera *camera_ptr;

    public:
        Camera() = delete;

        Camera(const Camera &camera) = delete;

        const Camera &operator=(const Camera &camera) = delete;

        // create camera by yaml file
        static bool create(const cv::FileNode &cameraNode);

        // destroy camera
        static void destroy();

        // get const pointer of camera
        static const Camera *getCamera();

        // print camera information
        void print() const;

        // Project Pc to image plane, and return its 2D pixel coordinate
        [[nodiscard]] Eigen::Vector2d project(const Eigen::Vector3d &Pc) const;

        [[nodiscard]] cv::Point2f project(const Eigen::Vector3f &Pc) const;

        // point is in image
        [[nodiscard]] bool isInImage(const cv::Point2f &p) const;

        // Get camera projection jacobian (2 x 3 matrix)
        [[nodiscard]] Eigen::Matrix<double, 2, 3> getProjJacobian(const Eigen::Vector3d &Pc) const;

        // pixel coordinate convert to normalize plane coordinate
        [[nodiscard]] Eigen::Vector3f pixelToNormalizePlane(const cv::Point2f &pt) const;

        // undistort key-points
        void undistortKeyPoints(const std::vector<cv::KeyPoint> &keyPoints,
                                std::vector<cv::KeyPoint> &keyPointsUn) const;

        /* parameters */
        const int width, height;
        const int fps;
        Eigen::Matrix3f K;

    private:
        cv::Mat mat_K, dist_coeffs;
        float fx, fy, cx, cy;
        float inv_fx, inv_fy;
        int min_x, max_x, min_y, max_y;
    };

} // mono_orb_slam3

#endif //MONO_ORB_SLAM3_CAMERA_H
