//
// Created by whitby on 10/17/23.
//

#include "System.h"
#include "Data.h"

#include <opencv2/opencv.hpp>
#include <iostream>

using namespace std;
using namespace mono_orb_slam3;

int main(int argc, char *argv[]) {
    if (argc < 6) {
        cout << "Usage: ./kitti_demo setting.yaml vocabulary_file data_folder trajectory_save_path" << endl;
        return -1;
    }

    // 1. create SLAM system
    System SLAM(argv[1], argv[2], true);

    // 2. load data
    const string dataFolder = argv[3];
    vector<double> timestamps;
    loadCameraData(dataFolder + "/image_00/times.txt", timestamps);
    int num_camera = (int) timestamps.size();
    cout << "load " << num_camera << " camera data" << endl;

    vector<ImuData> vecImu;
    loadImuData(dataFolder + "/oxts/imu.txt", vecImu);
    int num_imu = (int) vecImu.size();
    cout << "load " << num_imu << " imu data" << endl;

    // 3. main loop
    int idx1 = 0, idx2 = 0;
    while (vecImu[idx2].t < timestamps[idx1]) idx2++;
    while (idx1 < num_camera && idx2 < num_imu) {
        cout << endl << "iter: " << idx1 << endl;
        cv::Mat img = cv::imread(dataFolder + cv::format("/image_00/data/%010d.png", idx1), cv::IMREAD_GRAYSCALE);
        if (img.empty()) {
            cout << "fail to load image" << endl;
            break;
        }

        vector<ImuData> iterImus;
        while (vecImu[idx2].t < timestamps[idx1]) {
            iterImus.push_back(vecImu[idx2]);
            idx2++;
        }

        chrono::steady_clock::time_point t1 = chrono::steady_clock::now();

        SLAM.Track(timestamps[idx1], img, iterImus);

        chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
        double t_track = chrono::duration_cast<chrono::duration<double>>(t2 - t1).count();

        // wait to load the next frame
        double t = 0;
        if (idx1 < num_camera - 1) {
            t = timestamps[idx1 + 1] - timestamps[idx1];
        } else {
            t = timestamps[idx1] - timestamps[idx1 - 1];
        }

        if (t_track < t) {
            int sleep_mics = cvFloor((t - t_track) * 1e6);
            this_thread::sleep_for(chrono::microseconds(sleep_mics));
        }

        idx1++;
    }

    SLAM.ShutDown();
    SLAM.saveKeyFrameTrajectory(argv[4]);
    SLAM.saveKeyFrameVelocityAndBias(argv[5]);

    return 0;
}