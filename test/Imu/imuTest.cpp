//
// Created by whitby on 9/22/23.
//

#include "BasicObject/Frame.h"
#include "Sensor/Camera.h"
#include "Utils/LieAlgeBra.h"
#include "Data.h"

#include <fstream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace mono_orb_slam3;

void loadTrajectory(const string &path, vector<int> &indices, vector<Pose> &poses);

void predictCurFramePose(const shared_ptr<Frame> &lastFrame, const shared_ptr<Frame> &curFrame);

int main(int argc, char **argv) {
    if (argc != 4) {
        cout << "./imu_test phone.yaml dataset_path trajectory.txt" << endl;
        return -1;
    }

    // 1. create camera, imu and extractor
    cv::FileStorage fs(argv[1], cv::FileStorage::READ);
    Camera::create(fs["Camera"]);
    const Camera *camera_ptr = Camera::getCamera();
    camera_ptr->print();

    ImuCalib::create(fs["IMU"]);
    const ImuCalib *imu_ptr = ImuCalib::getImuCalib();
    imu_ptr->print();

    ORBExtractor extractor(4000, 1.2, 8, 20, 7);

    // 2. load trajectory.txt
    vector<int> kfIndices;
    vector<Pose> kfPoses;
    loadTrajectory(argv[3], kfIndices, kfPoses);
    int numKF = (int) kfIndices.size();
    cout << "load " << numKF << " keyframes' pose" << endl;

    // 3. load datasets
    const string dataFolder = argv[2];
    vector<double> timestamps;
    loadCameraData(dataFolder + "/camera.txt", timestamps);
    int num_camera = (int) timestamps.size();
    cout << "load " << num_camera << " camera data" << endl;

    vector<ImuData> vecImu;
    loadImuData(dataFolder + "/imu.txt", vecImu);
    int num_imu = (int) vecImu.size();
    cout << "load " << num_imu << " imu data" << endl;

    // 4. main loop
    int idx1 = 0, idx2 = 0, idx3 = 0;
    Eigen::Vector3f init_velo(-0.0620251, 0.111336, 0.14212);
    Bias initial_bias({-0.00502106, 0.00208642, 0.00210957}, {-0.0725798, -0.338067, -0.119612});
    shared_ptr<Frame> lastFrame, currentFrame;
    while (vecImu[idx2].t < timestamps[idx1]) idx2++;
    while (idx1 < num_camera && idx2 < num_imu) {
        cv::Mat img = cv::imread(dataFolder + cv::format("/images/%08d.png", idx1), cv::IMREAD_GRAYSCALE);

        vector<ImuData> iterImus;
        while (vecImu[idx2].t < timestamps[idx1]) {
            iterImus.push_back(vecImu[idx2]);
            idx2++;
        }

        currentFrame = make_shared<Frame>(img, timestamps[idx1], &extractor, initial_bias);
        if (!lastFrame) {
            if (idx1 == kfIndices[idx3]) {
                currentFrame->setPose(kfPoses[idx3].inverse());
                currentFrame->v_w = init_velo;
                cout << "iter: " << idx1 << endl;
                cout << kfPoses[idx3] << endl;
                cout << currentFrame->T_cw.inverse() << endl;
                cout << endl;
                idx3++;
            }
        } else {
            lastFrame->computePreIntegration(iterImus, timestamps[idx1]);
            predictCurFramePose(lastFrame, currentFrame);
            if (idx1 == kfIndices[idx3]) {
                cout << "iter: " << idx1 << endl;
                cout << kfPoses[idx3] << endl;
                cout << currentFrame->T_cw.inverse() << endl;
                cout << currentFrame->v_w.transpose() << endl;
                cout << endl;
                idx3++;
            }
        }

        lastFrame = currentFrame;
        idx1++;
    }

    return 0;
}

void loadTrajectory(const string &path, vector<int> &indices, vector<Pose> &poses) {
    ifstream fin(path);
    indices.reserve(500), poses.reserve(500);
    while (!fin.eof()) {
        string lineStr;
        getline(fin, lineStr);
        if (!lineStr.empty()) {
            stringstream ss(lineStr);
            int idx;
            double t;
            Pose Twc;
            ss >> idx >> t >> Twc;

            indices.push_back(idx);
            poses.push_back(Twc);
        }
    }

    fin.close();
}

void predictCurFramePose(const shared_ptr<Frame> &lastFrame, const shared_ptr<Frame> &curFrame) {
    shared_ptr<PreIntegrator> preIntegrator = lastFrame->pre_integrator;
    const float dt = preIntegrator->delta_t;
    const Eigen::Vector3f g(0, GRAVITY_VALUE, 0);

    Eigen::Matrix3f Rwb2 = lie::NormalizeRotationf(lastFrame->T_wb.R * preIntegrator->getUpdatedDeltaRotation());
    Eigen::Vector3f twb2 = lastFrame->T_wb.t + lastFrame->v_w * dt + 0.5 * g * dt * dt +
                           lastFrame->T_wb.R * preIntegrator->getUpdatedDeltaPosition();
    Eigen::Vector3f v2 = lastFrame->v_w + g * dt + lastFrame->T_wb.R * preIntegrator->getUpdatedDeltaVelocity();

    curFrame->setImuPoseAndVelocity({Rwb2, twb2}, v2);
}