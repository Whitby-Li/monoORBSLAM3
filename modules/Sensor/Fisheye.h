//
// Created by whitby on 4/19/24.
//

#ifndef MONO_ORB_SLAM3_FISHEYE_H
#define MONO_ORB_SLAM3_FISHEYE_H

#include "Camera.h"

namespace mono_orb_slam3 {

    class Fisheye : public Camera {
    public:
        Fisheye(int w, int h, int fps, cv::Mat matK, cv::Mat distCoeffs, DistortModel distModel);

        Eigen::Vector2d project(const Eigen::Vector3d &Pc) const override;

        cv::Point2f project(const Eigen::Vector3f &Pc) const override;

        Eigen::Vector3f backProject(const cv::Point2f &p) const override;

        bool isInImage(const cv::Point2f &p) const override;

        Eigen::Matrix<double, 2, 3> getProjJacobian(const Eigen::Vector3d &Pc) const override;

        float uncertainty(const cv::Point2f &p) const override;

        void undistortKeyPoints(const std::vector<cv::KeyPoint> &keyPoints,
                                std::vector<cv::KeyPoint> &keyPointsUn) const;

        bool reconstructWithTwoViews(const std::vector<cv::KeyPoint> &keyPoints1,
                                     const std::vector<cv::KeyPoint> &keyPoints2,
                                     const std::vector<int> &matches12, Eigen::Matrix3f &R21,
                                     Eigen::Vector3f &t21, std::vector<cv::Point3f> &points3D,
                                     std::vector<bool> &vecBeTriangulated) const override;

    private:
        float computeScale(const cv::Point2i &pt);

        float k1, k2, k3, k4;
        cv::Mat scale_mat;
    };

} // mono_orb_slam3

#endif //MONO_ORB_SLAM3_FISHEYE_H
