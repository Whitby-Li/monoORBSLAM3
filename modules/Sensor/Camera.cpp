//
// Created by whitby on 8/22/23.
//

#include "Camera.h"

#include <iostream>
#include <opencv2/calib3d.hpp>

using namespace std;

namespace mono_orb_slam3 {
    Camera *Camera::camera_ptr = nullptr;

    Camera::Camera(int w, int h, int fps, cv::Mat matK, cv::Mat distCoeffs)
            : width(w), height(h), fps(fps), mat_K(std::move(matK)), dist_coeffs(std::move(distCoeffs)) {
        fx = mat_K.at<float>(0, 0), cx = mat_K.at<float>(0, 2);
        fy = mat_K.at<float>(1, 1), cy = mat_K.at<float>(1, 2);
        inv_fx = 1.f / fx, inv_fy = 1.f / fy;

        K << fx, 0, cx, 0, fy, cy, 0, 0, 1;

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

    bool Camera::create(const cv::FileNode &cameraNode) {
        if (camera_ptr != nullptr) {
            cerr << "Camera has crated!" << endl;
            return false;
        }
        int w, h, fps;
        cameraNode["Width"] >> w;
        cameraNode["Height"] >> h;
        cameraNode["fps"] >> fps;

        cv::Mat distortK, distCoeffs;
        cameraNode["CameraMatrix"] >> distortK;
        cameraNode["Distortion"] >> distCoeffs;

        camera_ptr = new Camera(w, h, fps, distortK, distCoeffs);
        return true;
    }

    void Camera::destroy() {
        delete camera_ptr;
        camera_ptr = nullptr;
    }

    const Camera *Camera::getCamera() {
        return camera_ptr;
    }

    void Camera::print() const {
        cout << endl << "Camera information" << endl;
        cout << " - size: " << width << " x " << height << endl;
        cout << " - fps: " << fps << endl;
        cout << " - projection parameters: " << fx << ", " << fy << ", " << cx << ", " << cy << endl;
        cout << " - distort coeffs: " << dist_coeffs.at<float>(0) << ", " << dist_coeffs.at<float>(1) << ", "
             << dist_coeffs.at<float>(2) << ", " << dist_coeffs.at<float>(3) << endl;
        cout << " - distort model: " << "radtan" << endl;
        cout << " - border: " << min_x << ", " << min_y << ", " << max_x << ", " << max_y << endl << endl;
    }

    Eigen::Vector2d Camera::project(const Eigen::Vector3d &P) const {
        const double x = P[0] / P[2], y = P[1] / P[2];
        double u = fx * x + cx, v = fy * y + cy;
        return {u, v};
    }

    cv::Point2f Camera::project(const Eigen::Vector3f &P) const {
        const float x = P[0] / P[2], y = P[1] / P[2];
        float u = fx * x + cx, v = fy * y + cy;
        return {u, v};
    }

    bool Camera::isInImage(const cv::Point2f &p) const {
        if (p.x < min_x || p.x >= max_x || p.y < min_y || p.y >= max_y) return false;
        return true;
    }

    Eigen::Matrix<double, 2, 3> Camera::getProjJacobian(const Eigen::Vector3d &Pc) const {
        Eigen::Matrix<double, 2, 3> projectJacobian;
        projectJacobian << fx / Pc[2], 0, -fx * Pc[0] / (Pc[2] * Pc[2]), 0, fy / Pc[2], -fy * Pc[1] / (Pc[2] * Pc[2]);
        return projectJacobian;
    }

    Eigen::Vector3f Camera::pixelToNormalizePlane(const cv::Point2f &pt) const {
        return {(pt.x - cx) * inv_fx, (pt.y - cy) * inv_fy, 1.f};
    }

    void Camera::undistortKeyPoints(const std::vector<cv::KeyPoint> &keyPoints,
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

} // mono_orb_slam3