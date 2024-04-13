//
// Created by whitby on 8/22/23.
//

#ifndef MONO_ORB_SLAM3_LIEALGEBRA_H
#define MONO_ORB_SLAM3_LIEALGEBRA_H

#include <Eigen/Dense>

namespace mono_orb_slam3::lie {
    Eigen::Matrix3d Hat(double x, double y, double z);
    Eigen::Matrix3d Hat(const Eigen::Vector3d &w);
    Eigen::Matrix3f Hatf(const Eigen::Vector3f &w);

    Eigen::Matrix3d ExpSO3(double x, double y, double z);
    Eigen::Matrix3d ExpSO3(const Eigen::Vector3d &w);
    Eigen::Matrix3f ExpSO3f(const Eigen::Vector3f &w);

    Eigen::Vector3d LogSO3(const Eigen::Matrix3d &R);
    Eigen::Vector3f LogSO3f(const Eigen::Matrix3f &R);

    Eigen::Matrix3d RightJacobianSO3(const Eigen::Vector3d &w);
    Eigen::Matrix3f RightJacobianSO3f(const Eigen::Vector3f &w);

    Eigen::Matrix3d InverseRightJacobianSO3(const Eigen::Vector3d &w);
    Eigen::Matrix3f InverseRightJacobianSO3f(const Eigen::Vector3f &w);

    Eigen::Matrix3d NormalizeRotation(const Eigen::Matrix3d &R);
    Eigen::Matrix3f NormalizeRotationf(const Eigen::Matrix3f &R);
} // lie

#endif //MONO_ORB_SLAM3_LIEALGEBRA_H
