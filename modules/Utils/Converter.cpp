//
// Created by whitby on 8/21/23.
//

#include "Converter.h"
#include "Utils/LieAlgeBra.h"

namespace mono_orb_slam3 {
    g2o::SE3Quat Converter::toSE3Quat(const Pose &Tcw) {
        return {lie::NormalizeRotation(Tcw.R.cast<double>()), Tcw.t.cast<double>()};
    }

    cv::Mat Converter::toCvMat(const Eigen::Matrix4d &m) {
        cv::Mat mat(4, 4, CV_32F);
        for (int i = 0; i < 4; ++i)
            for (int j = 0; j < 4; ++j)
                mat.at<float>(i, j) = m(i, j);

        return mat;
    }

    cv::Mat Converter::toCvMat(const Eigen::Matrix4f &m) {
        cv::Mat mat(4, 4, CV_32F);
        for (int i = 0; i < 4; ++i)
            for (int j = 0; j < 4; ++j)
                mat.at<float>(i, j) = m(i, j);

        return mat;
    }

    cv::Mat Converter::toCvMat(const Eigen::Matrix3d &m) {
        cv::Mat mat(3, 3, CV_32F);
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                mat.at<float>(i, j) = m(i, j);

        return mat;
    }

    cv::Mat Converter::toCvMat(const Eigen::Matrix3f &m) {
        cv::Mat mat(3, 3, CV_32F);
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                mat.at<float>(i, j) = m(i, j);

        return mat;
    }

    cv::Mat Converter::toCvMat(const Eigen::Vector3d &v) {
        cv::Mat mat = (cv::Mat_<float>(3, 1) << v.x(), v.y(), v.z());
        return mat;
    }

    cv::Mat Converter::toCvMat(const Eigen::Vector3f &v) {
        cv::Mat mat = (cv::Mat_<float>(3, 1) << v.x(), v.y(), v.z());
        return mat;
    }

    Eigen::Matrix4d Converter::toMatrix4d(const cv::Mat &m) {
        Eigen::Matrix4d matrix;
        matrix << m.at<float>(0, 0), m.at<float>(0, 1), m.at<float>(0, 2), m.at<float>(0, 3),
                m.at<float>(1, 0), m.at<float>(1, 1), m.at<float>(1, 2), m.at<float>(1, 3),
                m.at<float>(2, 0), m.at<float>(2, 1), m.at<float>(2, 2), m.at<float>(2, 3),
                m.at<float>(3, 0), m.at<float>(3, 1), m.at<float>(3, 2), m.at<float>(3, 3);
        return matrix;
    }

    Eigen::Matrix4f Converter::toMatrix4f(const cv::Mat &m) {
        Eigen::Matrix4f matrix;
        matrix << m.at<float>(0, 0), m.at<float>(0, 1), m.at<float>(0, 2), m.at<float>(0, 3),
                m.at<float>(1, 0), m.at<float>(1, 1), m.at<float>(1, 2), m.at<float>(1, 3),
                m.at<float>(2, 0), m.at<float>(2, 1), m.at<float>(2, 2), m.at<float>(2, 3),
                m.at<float>(3, 0), m.at<float>(3, 1), m.at<float>(3, 2), m.at<float>(3, 3);
        return matrix;
    }

    Eigen::Matrix3d Converter::toMatrix3d(const cv::Mat &m) {
        Eigen::Matrix3d matrix;
        matrix << m.at<float>(0, 0), m.at<float>(0, 1), m.at<float>(0, 2),
                m.at<float>(1, 0), m.at<float>(1, 1), m.at<float>(1, 2),
                m.at<float>(2, 0), m.at<float>(2, 1), m.at<float>(2, 2);
        return matrix;
    }

    Eigen::Matrix3f Converter::toMatrix3f(const cv::Mat &m) {
        Eigen::Matrix3f matrix;
        matrix << m.at<float>(0, 0), m.at<float>(0, 1), m.at<float>(0, 2),
                m.at<float>(1, 0), m.at<float>(1, 1), m.at<float>(1, 2),
                m.at<float>(2, 0), m.at<float>(2, 1), m.at<float>(2, 2);
        return matrix;
    }

    Eigen::Vector3d Converter::toVector3d(const cv::Mat &v) {
        return {v.at<float>(0, 0), v.at<float>(1, 0), v.at<float>(2, 0)};
    }

    Eigen::Vector3f Converter::toVector3f(const cv::Mat &v) {
        return {v.at<float>(0, 0), v.at<float>(1, 0), v.at<float>(2, 0)};
    }
} // mono_orb_slam3