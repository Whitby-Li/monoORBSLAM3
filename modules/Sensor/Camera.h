//
// Created by whitby on 8/22/23.
//

#ifndef MONO_ORB_SLAM3_CAMERA_H
#define MONO_ORB_SLAM3_CAMERA_H

#include "Frontend/TwoViewReconstruction.h"

#include <Eigen/Core>
#include <opencv2/core/core.hpp>
#include <utility>

namespace mono_orb_slam3 {

    class Camera {
    public:
        enum DistortModel {
            RAD_TAN = 0,
            EQUIDISTANT = 1
        };
    protected:
        Camera(int w, int h, int fps, cv::Mat matK, cv::Mat distCoeffs, DistortModel distModel);

        // camera pointer
        static Camera *camera_ptr;

    public:
        Camera() = delete;

        Camera(const Camera &camera) = delete;

        const Camera &operator=(const Camera &camera) = delete;

        virtual ~Camera() {
            delete reconstructor;
            reconstructor = nullptr;
        }

        // create camera by yaml file
        static bool create(const cv::FileNode &cameraNode);

        // destroy camera
        static void destroy();

        // get const pointer of camera
        static const Camera *getCamera();

        // print camera information
        void print() const;

        // Project Pc to image plane, and return its 2D pixel coordinate
        virtual Eigen::Vector2d project(const Eigen::Vector3d &Pc) const = 0;

        virtual cv::Point2f project(const Eigen::Vector3f &Pc) const = 0;

        // back project pixel coordinate to normal plane
        virtual Eigen::Vector3f backProject(const cv::Point2f &p) const = 0;

        // point is in image
        virtual bool isInImage(const cv::Point2f &p) const = 0;

        // get camera projection jacobian (2 x 3 matrix)
        virtual Eigen::Matrix<double, 2, 3> getProjJacobian(const Eigen::Vector3d &Pc) const = 0;

        // undistort key-points
        virtual void undistortKeyPoints(const std::vector<cv::KeyPoint> &keyPoints,
                                        std::vector<cv::KeyPoint> &keyPointsUn) const = 0;

        // reconstruct 3d point map with two images
        virtual bool reconstructWithTwoViews(const std::vector<cv::KeyPoint> &keyPoints1,
                                             const std::vector<cv::KeyPoint> &keyPoints2,
                                             const std::vector<int> &matches12, Eigen::Matrix3f &R21,
                                             Eigen::Vector3f &t21, std::vector<cv::Point3f> &points3D,
                                             std::vector<bool> &vecBeTriangulated) const = 0;

        /* parameters */
        const int width, height;
        const int fps;
        const DistortModel distort_model;
        Eigen::Matrix3f K;
        TwoViewReconstruction *reconstructor;

    protected:
        cv::Mat mat_K, dist_coeffs;
        float fx, fy, cx, cy;
        float inv_fx, inv_fy;

    };

} // mono_orb_slam3

#endif //MONO_ORB_SLAM3_CAMERA_H
