//
// Created by whitby on 8/22/23.
//

#include "LieAlgeBra.h"

namespace mono_orb_slam3::lie {
    const double eps = 1e-6;

    Eigen::Matrix3d Hat(double x, double y, double z) {
        Eigen::Matrix3d W;
        W << 0, -z, y,
                z, 0, -x,
                -y, x, 0;
        return W;
    }

    Eigen::Matrix3d Hat(const Eigen::Vector3d &w) {
        return Hat(w.x(), w.y(), w.z());
    }

    Eigen::Matrix3f Hatf(const Eigen::Vector3f &w) {
        Eigen::Matrix3f W;
        W << 0, -w[2], w[1],
                w[2], 0, -w[0],
                -w[1], w[0], 0;
        return W;
    }

    Eigen::Matrix3d ExpSO3(double x, double y, double z) {
        const double d2 = x * x + y * y + z * z;
        const double d = sqrt(d2);
        Eigen::Matrix3d W = Hat(x, y, z);
        if (d < eps) {
            Eigen::Matrix3d res = Eigen::Matrix3d::Identity() + W + 0.5 * W * W;
            return NormalizeRotation(res);
        } else {
            Eigen::Matrix3d res = Eigen::Matrix3d::Identity() + sin(d) / d * W + (1 - cos(d)) / d2 * W * W;
            return NormalizeRotation(res);
        }
    }

    Eigen::Matrix3d ExpSO3(const Eigen::Vector3d &w) {
        return ExpSO3(w.x(), w.y(), w.z());
    }

    Eigen::Matrix3f ExpSO3f(const Eigen::Vector3f &w) {
        const float d2 = w.squaredNorm();
        const float d = sqrtf(d2);
        Eigen::Matrix3f W = Hatf(w);
        if (d < eps) {
            Eigen::Matrix3f res = Eigen::Matrix3f::Identity() + W + 0.5 * W * W;
            return res;
        } else {
            Eigen::Matrix3f res = Eigen::Matrix3f::Identity() + sinf(d) / d * W + (1 - cosf(d)) / d2 * W * W;
            return res;
        }
    }

    Eigen::Vector3d LogSO3(const Eigen::Matrix3d &R) {
        const double tr = R.trace();
        if (tr == 3) return Eigen::Vector3d::Zero();
        Eigen::Vector3d w(R(2, 1) - R(1, 2), R(0, 2) - R(2, 0), R(1, 0) - R(0, 1));
        w = 0.5 * w;
        const double sinTheta = w.norm();
        const double cosTheta = 0.5f * (tr - 1);
        const double theta = atan2(sinTheta, cosTheta);
        if (abs(theta) < eps) return w;
        return theta * w / sinTheta;
    }

    Eigen::Vector3f LogSO3f(const Eigen::Matrix3f &R) {
        const float tr = R.trace();
        if (tr == 3) return Eigen::Vector3f::Zero();
        Eigen::Vector3f w(R(2, 1) - R(1, 2), R(0, 2) - R(2, 0), R(1, 0) - R(0, 1));
        w = 0.5 * w;
        const float sinTheta = w.norm();
        const float cosTheta = 0.5f * (tr - 1);
        const float theta = atan2f(sinTheta, cosTheta);
        if (abs(theta) < eps) return w;
        return theta * w / sinTheta;
    }

    Eigen::Matrix3d RightJacobianSO3(const Eigen::Vector3d &w) {
        const double d2 = w.squaredNorm();
        const double d = sqrt(d2);
        Eigen::Matrix3d W = Hat(w);
        if (d < eps)
            return Eigen::Matrix3d::Identity();
        else
            return Eigen::Matrix3d::Identity() - (1 - cos(d)) / d2 * W + (d - sin(d)) / (d2 * d) * W * W;
    }

    Eigen::Matrix3f RightJacobianSO3f(const Eigen::Vector3f &w) {
        const float d2 = w.squaredNorm();
        const float d = sqrtf(d2);
        Eigen::Matrix3f W = Hatf(w);
        if (d < eps)
            return Eigen::Matrix3f::Identity();
        else
            return Eigen::Matrix3f::Identity() - (1 - cosf(d)) / d2 * W + (d - sinf(d)) / (d2 * d) * W * W;
    }

    Eigen::Matrix3d InverseRightJacobianSO3(const Eigen::Vector3d &w) {
        const double d2 = w.squaredNorm();
        const double d = sqrt(d2);
        Eigen::Matrix3d W = Hat(w);
        if (d < eps)
            return Eigen::Matrix3d::Identity();
        else
            return Eigen::Matrix3d::Identity() + W / 2 + (1 / d2 - (1 + cos(d)) / (2 * d * sin(d))) * W * W;
    }

    Eigen::Matrix3f InverseRightJacobianSO3f(const Eigen::Vector3f &w) {
        const float d2 = w.squaredNorm();
        const float d = sqrtf(d2);
        Eigen::Matrix3f W = Hatf(w);
        if (d < eps)
            return Eigen::Matrix3f::Identity();
        else
            return Eigen::Matrix3f::Identity() + W / 2 + (1 / d2 - (1 + cosf(d)) / (2 * d * sinf(d))) * W * W;
    }

    Eigen::Matrix3d NormalizeRotation(const Eigen::Matrix3d &R) {
        Eigen::JacobiSVD<Eigen::Matrix3d> svd(R, Eigen::ComputeFullU | Eigen::ComputeFullV);
        return svd.matrixU() * svd.matrixV().transpose();
    }

    Eigen::Matrix3f NormalizeRotationf(const Eigen::Matrix3f &R) {
        Eigen::JacobiSVD<Eigen::Matrix3f> svd(R, Eigen::ComputeFullU | Eigen::ComputeFullV);
        return svd.matrixU() * svd.matrixV().transpose();
    }
} // lie