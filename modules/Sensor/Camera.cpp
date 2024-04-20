//
// Created by whitby on 8/22/23.
//

#include "Camera.h"
#include "Pinhole.h"
#include "Fisheye.h"

#include <iostream>
#include <opencv2/calib3d.hpp>

using namespace std;

namespace mono_orb_slam3 {
    Camera *Camera::camera_ptr = nullptr;

    Camera::Camera(int w, int h, int fps, cv::Mat matK, cv::Mat distCoeffs, DistortModel distModel)
            : width(w), height(h), fps(fps), mat_K(std::move(matK)), dist_coeffs(std::move(distCoeffs)),
              distort_model(distModel) {
        fx = mat_K.at<float>(0, 0), cx = mat_K.at<float>(0, 2);
        fy = mat_K.at<float>(1, 1), cy = mat_K.at<float>(1, 2);
        inv_fx = 1.f / fx, inv_fy = 1.f / fy;

        K << fx, 0, cx, 0, fy, cy, 0, 0, 1;
        reconstructor = new TwoViewReconstruction(K);
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
        const string distModel = cameraNode["DistortionModel"];

        if (distModel == "radtan")
            camera_ptr = new Pinhole(w, h, fps, distortK, distCoeffs, RAD_TAN);
        else if (distModel == "equidistant")
            camera_ptr = new Fisheye(w, h, fps, distortK, distCoeffs, EQUIDISTANT);
        else {
            cerr << "un-recognition distort model: " << distModel << endl;
            return false;
        }
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
        cout << " - distort model: " << distort_model << endl;
    }

} // mono_orb_slam3