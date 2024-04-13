//
// Created by whitby on 8/21/23.
//

#ifndef MONO_ORB_SLAM3_CONVERTER_H
#define MONO_ORB_SLAM3_CONVERTER_H

#include <opencv2/core/core.hpp>
#include <Eigen/Core>

#include <g2o/types/sba/types_six_dof_expmap.h>
#include "BasicObject/KeyFrame.h"

namespace mono_orb_slam3 {

    class Converter {
    public:
        static g2o::SE3Quat toSE3Quat(const Pose &Tcw);

        static cv::Mat toCvMat(const Eigen::Matrix4d &m);

        static cv::Mat toCvMat(const Eigen::Matrix4f &m);

        static cv::Mat toCvMat(const Eigen::Matrix3d &m);

        static cv::Mat toCvMat(const Eigen::Matrix3f &m);

        static cv::Mat toCvMat(const Eigen::Vector3d &v);

        static cv::Mat toCvMat(const Eigen::Vector3f &v);

        static Eigen::Matrix4d toMatrix4d(const cv::Mat &m);

        static Eigen::Matrix4f toMatrix4f(const cv::Mat &m);

        static Eigen::Matrix3d toMatrix3d(const cv::Mat &m);

        static Eigen::Matrix3f toMatrix3f(const cv::Mat &m);

        static Eigen::Vector3d toVector3d(const cv::Mat &v);

        static Eigen::Vector3f toVector3f(const cv::Mat &v);
    };

} // mono_orb_slam3

#endif //MONO_ORB_SLAM3_CONVERTER_H
