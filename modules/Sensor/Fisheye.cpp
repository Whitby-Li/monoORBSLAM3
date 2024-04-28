//
// Created by whitby on 4/19/24.
//

#include "Fisheye.h"

#include <iostream>
#include <opencv2/calib3d/calib3d.hpp>

using namespace std;

namespace mono_orb_slam3 {

    Fisheye::Fisheye(int w, int h, int fps, cv::Mat matK, cv::Mat distCoeffs, DistortModel distModel)
        : Camera(w, h, fps, std::move(matK), std::move(distCoeffs), distModel) {
        k1 = dist_coeffs.at<float>(0, 0);
        k2 = dist_coeffs.at<float>(1, 0);
        k3 = dist_coeffs.at<float>(2, 0);
        k4 = dist_coeffs.at<float>(3, 0);

        scale_mat = cv::Mat(height, width, CV_32F);
        float maxScale = 0, minScale = 100;
        for (int u = 0; u < width; ++u) {
            for (int v = 0; v < height; ++v) {
                float scale = computeScale({u, v});
                scale_mat.at<float>(v, u) = scale;
                maxScale = fmaxf(scale, maxScale);
                minScale = fminf(scale, minScale);
            }
        }

        cout << "scale range: [" << minScale << ", " << maxScale << "]" << endl;
    }

    Eigen::Vector2d Fisheye::project(const Eigen::Vector3d &Pc) const {
        double a = Pc[0] / Pc[2], b = Pc[1] / Pc[2];
        double r = sqrt(a * a + b * b);
        double theta = atan(r);

        double theta2 = theta * theta;
        double theta3 = theta * theta2;
        double theta5 = theta2 * theta3;
        double theta7 = theta2 * theta5;
        double theta9 = theta2 * theta7;
        double theta_d = theta + dist_coeffs.at<float>(0, 0) * theta3 + dist_coeffs.at<float>(1, 0) * theta5
                         + dist_coeffs.at<float>(2, 0) * theta7 + dist_coeffs.at<float>(3, 0) * theta9;

        return {fx * theta_d * a / r + cx, fy * theta_d * b / r + cy};

    }

    cv::Point2f Fisheye::project(const Eigen::Vector3f &Pc) const {
        float a = Pc[0] / Pc[2], b = Pc[1] / Pc[2];
        float r = sqrtf(a * a + b * b);
        float theta = atanf(r);

        float theta2 = theta * theta;
        float theta3 = theta * theta2;
        float theta5 = theta2 * theta3;
        float theta7 = theta2 * theta5;
        float theta9 = theta2 * theta7;
        float theta_d = theta + dist_coeffs.at<float>(0, 0) * theta3 + dist_coeffs.at<float>(1, 0) * theta5
                        + dist_coeffs.at<float>(2, 0) * theta7 + dist_coeffs.at<float>(3, 0) * theta9;

        return {fx * theta_d * a / r + cx, fy * theta_d * b / r + cy};
    }

    Eigen::Vector3f Fisheye::backProject(const cv::Point2f &p) const {
        // use newthon method to solve for theta with good precision
        float x = (p.x - cx) * inv_fx, y = (p.y - cy) * inv_fy;
        float scale = scale_mat.at<float>(p.y, p.x);
        return {x * scale, y * scale, 1};
    }

    bool Fisheye::isInImage(const cv::Point2f &p) const {
        if (p.x < 0 || p.x >= width || p.y < 0 || p.y >= height) return false;
        return true;
    }

    Eigen::Matrix<double, 2, 3> Fisheye::getProjJacobian(const Eigen::Vector3d &Pc) const {
        double x2 = Pc[0] * Pc[0], y2 = Pc[1] * Pc[1], z2 = Pc[2] * Pc[2];
        double r2 = x2 + y2;
        double r = sqrt(r2);
        double r3 = r2 * r;
        double theta = atan2(r, Pc[2]);

        double theta2 = theta * theta, theta3 = theta2 * theta;
        double theta4 = theta2 * theta2, theta5 = theta4 * theta;
        double theta6 = theta2 * theta4, theta7 = theta6 * theta;
        double theta8 = theta4 * theta4, theta9 = theta8 * theta;

        double f = theta + theta3 * dist_coeffs.at<float>(0, 0) + theta5 * dist_coeffs.at<float>(1, 0)
                   + theta7 * dist_coeffs.at<float>(2, 0) + theta9 * dist_coeffs.at<float>(3, 0);
        double fd = 1 + 3 * dist_coeffs.at<float>(0, 0) * theta2 + 5 * dist_coeffs.at<float>(1, 0) * theta4
                    + 7 * dist_coeffs.at<float>(2, 0) * theta6 + 9 * dist_coeffs.at<float>(3, 0) * theta8;

        Eigen::Matrix<double, 2, 3> JacGood;
        JacGood(0, 0) = fx * (fd * Pc[2] * x2 / (r2 * (r2 + z2)) + f * y2 / r3);
        JacGood(1, 0) = fy * (fd * Pc[2] * Pc[1] * Pc[0] / (r2 * (r2 + z2)) - f * Pc[1] * Pc[0] / r3);

        JacGood(0, 1) = fx * (fd * Pc[2] * Pc[1] * Pc[0] / (r2 * (r2 + z2)) - f * Pc[1] * Pc[0] / r3);
        JacGood(1, 1) = fy * (fd * Pc[2] * y2 / (r2 * (r2 + z2)) + f * x2 / r3);

        JacGood(0, 2) = -fx * fd * Pc[0] / (r2 + z2);
        JacGood(1, 2) = -fy * fd * Pc[1] / (r2 + z2);

        return JacGood;
    }

    float Fisheye::uncertainty(const cv::Point2f &p) const {
        return scale_mat.at<float>(p.y, p.x);
    }

    void Fisheye::undistortKeyPoints(const std::vector<cv::KeyPoint> &keyPoints,
                                     std::vector<cv::KeyPoint> &keyPointsUn) const {
        keyPointsUn = keyPoints;
    }

    bool Fisheye::reconstructWithTwoViews(const std::vector<cv::KeyPoint> &keyPoints1,
                                          const std::vector<cv::KeyPoint> &keyPoints2,
                                          const std::vector<int> &matches12, Eigen::Matrix3f &R21,
                                          Eigen::Vector3f &t21, std::vector<cv::Point3f> &points3D,
                                          std::vector<bool> &vecBeTriangulated) const {
        // correct fisheye distortion
        vector<cv::KeyPoint> unKeyPoints1 = keyPoints1, unKeyPoints2 = keyPoints2;
        size_t num_kp1 = keyPoints1.size(), num_kp2 = keyPoints2.size();
        vector<cv::Point2f> points1(keyPoints1.size()), points2(keyPoints2.size());

        for (size_t i = 0; i < num_kp1; ++i) points1[i] = keyPoints1[i].pt;
        for (size_t i = 0; i < num_kp2; ++i) points2[i] = keyPoints2[i].pt;

        cv::fisheye::undistortPoints(points1, points1, mat_K, dist_coeffs, cv::Mat(), mat_K);
        cv::fisheye::undistortPoints(points2, points2, mat_K, dist_coeffs,  cv::Mat(), mat_K);

        for (size_t i = 0; i < num_kp1; ++i) unKeyPoints1[i].pt = points1[i];
        for (size_t i = 0; i < num_kp2; ++i) unKeyPoints2[i].pt = points2[i];

        return reconstructor->Reconstruct(unKeyPoints1, unKeyPoints2, matches12, R21, t21, points3D, vecBeTriangulated);
    }

    float Fisheye::computeScale(const cv::Point2i &p) {
        float x = (p.x - cx) * inv_fx, y = (p.y - cy) * inv_fy;
        float scale = 1.f;
        float theta_d = sqrtf(x * x + y * y);
        float th = CV_PI * 0.45;
        theta_d = fminf(theta_d, th);

        if (theta_d < 0) cout << "error" << endl;

        if (theta_d > 1e-3) {
            // compensate distortion iteratively
            float theta = theta_d;

            for (int j = 0; j < 10; j++) {
                float theta2 = theta * theta, theta4 = theta2 * theta2, theta6 = theta4 * theta2, theta8 =
                        theta4 * theta4;
                float k0_theta2 = k1 * theta2;
                float k1_theta4 = k2 * theta4;
                float k2_theta6 = k3 * theta6;
                float k3_theta8 = k4 * theta8;
                float theta_fix = (theta * (1 + k0_theta2 + k1_theta4 + k2_theta6 + k3_theta8) - theta_d) /
                                  (1 + 3 * k0_theta2 + 5 * k1_theta4 + 7 * k2_theta6 + 9 * k3_theta8);

                theta = theta - theta_fix;
                if (fabsf(theta_fix) < 1e-6)
                    break;
            }

            scale = std::tan(theta) / theta_d;
        }
        return scale;
    }

} // mono_orb_slam3