//
// Created by whitby on 4/20/24.
//

#include "System.h"
#include "Data.h"

#include <opencv2/opencv.hpp>
#include <iostream>
#include <thread>
#include <chrono>

using namespace std;
using namespace mono_orb_slam3;

int main(int argc, char *argv[]) {
    if (argc < 4) {
        cout << "Usage: " << argv[0]
             << " <setting yaml> <vocabulary file> <dataset folder> <output folder>";
        return 1;
    }

    // 1. load data
    const string dataFolder = argv[3];
    vector<double> timestamps;
    loadCameraData(dataFolder + "/cam0/times.txt", timestamps);
    int num_camera = (int) timestamps.size();
    cout << "load " << num_camera << " camera data" << endl;

    vector<ImuData> vecImu;
    loadImuData(dataFolder + "/imu.txt", vecImu);
    int num_imu = (int) vecImu.size();
    cout << "load " << num_imu << " imu data" << endl;

    // 2. create SLAM system
    System SLAM(argv[1], argv[2], true, true);
    SLAM.setSaveFolder(argv[4]);

    // 3. main loop
    int idx1 = 0, idx2 = 0;
    while (vecImu[idx2].t < timestamps[idx1]) idx2++;
    while (idx1 < num_camera && idx2 < num_imu) {
        cout << endl << "iter: " << idx1 << endl;
        cv::Mat img = cv::imread(dataFolder + cv::format("/cam0/data/%08d.png", idx1), cv::IMREAD_GRAYSCALE);
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

    // 4. shutdown SLAM system, and save trajectory and others
    SLAM.ShutDown();
    /*SLAM.saveKeyFrameTrajectory();
    SLAM.saveKeyFrameVelocityAndBias();*/
    return 0;
}