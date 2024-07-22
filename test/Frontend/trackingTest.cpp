//
// Created by whitby on 8/23/23.
//

#include "System.h"
#include "Data.h"

#include <opencv2/opencv.hpp>
#include <iostream>

using namespace std;
using namespace mono_orb_slam3;

int main(int argc, char *argv[]) {
    if (argc != 5) {
        cout << "Usage: ./tracking_test setting.yaml vocabulary_file data_folder trajectory_save_path" << endl;
        return -1;
    }

    // 1. create SLAM system
    System SLAM(argv[1], argv[2]);

    // 2. load data
    const string dataFolder = argv[3];
    vector<double> timestamps;
    loadCameraData(dataFolder + "/camera.txt", timestamps);
    int num_camera = (int) timestamps.size();
    cout << "load " << num_camera << " camera data" << endl;

    vector<ImuData> vecImu;
    loadImuData(dataFolder + "/imu.txt", vecImu);
    int num_imu = (int) vecImu.size();
    cout << "load " << num_imu << " imu data" << endl;

    // 3. main loop
    int idx1 = 0, idx2 = 0;
    while (vecImu[idx2].t < timestamps[idx1]) idx2++;
    while (idx1 < 80 && idx2 < num_imu) {
        cout << endl << "iter: " << idx1 << endl;
        cv::Mat img = cv::imread(dataFolder + cv::format("/images/%08d.png", idx1), cv::IMREAD_GRAYSCALE);
        if (img.empty()) {
            cout << "fail to load image" << endl;
            break;
        }

        vector<ImuData> iterImus;
        while (vecImu[idx2].t < timestamps[idx1]) {
            iterImus.push_back(vecImu[idx2]);
            idx2++;
        }

        SLAM.Track(timestamps[idx1], img, iterImus);

        idx1++;
    }

    SLAM.ShutDown();
    SLAM.saveKeyFrameTrajectory();

    return 0;
}