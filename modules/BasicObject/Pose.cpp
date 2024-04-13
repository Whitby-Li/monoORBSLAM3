//
// Created by whitby on 9/3/23.
//

#include "Pose.h"

namespace mono_orb_slam3 {
    Pose Pose::operator*(const Pose &pose) const {
        return {R * pose.R, R * pose.t + t};
    }

    [[nodiscard]] Pose Pose::inverse() const {
        return {R.transpose(), -R.transpose() * t};
    }

    void Pose::normalize() {
        Eigen::JacobiSVD<Eigen::Matrix3f> svd(R, Eigen::ComputeFullU | Eigen::ComputeFullV);
        R = svd.matrixU() * svd.matrixV().transpose();
    }

    std::istream &operator>>(std::istream &input, Pose &pose) {
        float x, y, z, w;
        float t1, t2, t3;
        input >> x >> y >> z >> w >> t1 >> t2 >> t3;
        pose.R = Eigen::Quaternionf(w, x, y, z).toRotationMatrix();
        pose.t = Eigen::Vector3f(t1, t2, t3);
        return input;
    }

    std::ostream &operator<<(std::ostream &output, const Pose &pose) {
        Eigen::Quaternionf q(pose.R);
        output << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << " | " << pose.t[0] << " " << pose.t[1] << " "
               << pose.t[2];
        return output;
    }
}