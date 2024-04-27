//
// Created by whitby on 4/18/24.
//

#include "Pinhole.h"

#include <opencv2/calib3d.hpp>
#include <utility>

using namespace std;

namespace mono_orb_slam3 {

    Pinhole::Pinhole(int w, int h, int fps, cv::Mat matK, cv::Mat distCoeffs, DistortModel distModel)
            : Camera(w, h, fps, std::move(matK), std::move(distCoeffs), distModel) {
        // compute border
        cv::Mat mat = (cv::Mat_<float>(4, 2) << 0, 0, w, 0, 0, h, w, h);
        mat = mat.reshape(2);
        cv::undistortPoints(mat, mat, mat_K, dist_coeffs, cv::Mat(), mat_K);
        mat = mat.reshape(1);

        min_x = cvCeil(max(mat.at<float>(0, 0), mat.at<float>(2, 0)));
        max_x = cvCeil(min(mat.at<float>(1, 0), mat.at<float>(3, 0)));
        min_y = cvCeil(max(mat.at<float>(0, 1), mat.at<float>(1, 1)));
        max_y = cvCeil(min(mat.at<float>(2, 1), mat.at<float>(3, 1)));
    }

    Eigen::Vector2d Pinhole::project(const Eigen::Vector3d &Pc) const {
        const double x = Pc[0] / Pc[2], y = Pc[1] / Pc[2];
        double u = fx * x + cx, v = fy * y + cy;
        return {u, v};
    }

    cv::Point2f Pinhole::project(const Eigen::Vector3f &Pc) const {
        const float x = Pc[0] / Pc[2], y = Pc[1] / Pc[2];
        float u = fx * x + cx, v = fy * y + cy;
        return {u, v};
    }

    Eigen::Vector3f Pinhole::backProject(const cv::Point2f &p) const {
        return {(p.x - cx) * inv_fx, (p.y - cy) * inv_fy, 1};
    }

    bool Pinhole::isInImage(const cv::Point2f &p) const {
        if (p.x < min_x || p.x >= max_x || p.y < min_y || p.y >= max_y) return false;
        return true;
    }

    Eigen::Matrix<double, 2, 3> Pinhole::getProjJacobian(const Eigen::Vector3d &Pc) const {
        Eigen::Matrix<double, 2, 3> projectJacobian;
        projectJacobian << fx / Pc[2], 0, -fx * Pc[0] / (Pc[2] * Pc[2]), 0, fy / Pc[2], -fy * Pc[1] / (Pc[2] * Pc[2]);
        return projectJacobian;
    }

    float Pinhole::uncertainty(const cv::Point2f &p) const {
        return 1.f;
    }

    void Pinhole::undistortKeyPoints(const std::vector<cv::KeyPoint> &keyPoints,
                                     std::vector<cv::KeyPoint> &keyPointsUn) const {
        keyPointsUn = keyPoints;
        if (dist_coeffs.at<float>(0) == 0) {
            return;
        }

        int n = (int) keyPoints.size();
        // fill matrix with points
        cv::Mat mat(n, 2, CV_32F);
        for (int i = 0; i < n; ++i) {
            mat.at<float>(i, 0) = keyPoints[i].pt.x;
            mat.at<float>(i, 1) = keyPoints[i].pt.y;
        }

        // undistort points
        mat = mat.reshape(2);
        cv::undistortPoints(mat, mat, mat_K, dist_coeffs, cv::Mat(), mat_K);
        mat = mat.reshape(1);

        for (int i = 0; i < n; ++i) {
            keyPointsUn[i].pt.x = mat.at<float>(i, 0);
            keyPointsUn[i].pt.y = mat.at<float>(i, 1);
        }
    }

    bool Pinhole::reconstructWithTwoViews(const std::vector<cv::KeyPoint> &keyPoints1,
                                          const std::vector<cv::KeyPoint> &keyPoints2,
                                          const std::vector<int> &matches12, Eigen::Matrix3f &R21, Eigen::Vector3f &t21,
                                          std::vector<cv::Point3f> &points3D,
                                          std::vector<bool> &vecBeTriangulated) const {
        return reconstructor->Reconstruct(keyPoints1, keyPoints2, matches12, R21, t21, points3D, vecBeTriangulated);
    }

} // mono_orb_slam3