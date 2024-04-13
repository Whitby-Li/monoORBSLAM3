//
// Created by whitby on 8/30/23.
//

#ifndef MONO_ORB_SLAM3_POSE_H
#define MONO_ORB_SLAM3_POSE_H

#include <Eigen/Dense>

namespace mono_orb_slam3 {
    struct Pose {
        Pose() {
            R = Eigen::Matrix3f::Identity();
            t = Eigen::Vector3f::Zero();
        }

        Pose(Eigen::Matrix3f rotation, Eigen::Vector3f translation) : R(std::move(rotation)),
                                                                      t(std::move(translation)) {}

        Pose operator*(const Pose &pose) const;

        [[nodiscard]] Pose inverse() const;

        void normalize();

        [[nodiscard]] inline Eigen::Vector3f map(const Eigen::Vector3f &P) const {
            return R * P + t;
        }

        Eigen::Matrix3f R;
        Eigen::Vector3f t;
    };

    std::istream &operator>>(std::istream &input, Pose &pose);

    std::ostream &operator<<(std::ostream &output, const Pose &pose);
}

#endif //MONO_ORB_SLAM3_POSE_H
