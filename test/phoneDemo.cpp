//
// Created by whitby on 8/27/23.
//

#include "System.h"
#include "Data.h"

#include <opencv2/opencv.hpp>
#include <iostream>

using namespace std;
using namespace mono_orb_slam3;

int main(int argc, char *argv[]) {
    if (argc < 5) {
        cout << "Usage: ./test/bin/phone_demo setting.yaml vocabulary.txt data_folder results/trajectory.txt result/velo.txt" << endl;
        return -1;
    }

    // 1. load data
    const string dataFolder = argv[3];

    // video.mp4
    cv::VideoCapture videoCapture(dataFolder + "/video.mp4");
    if (!videoCapture.isOpened()) {
        cerr << "fail to load video.mp4 at " << dataFolder << endl;
        return -1;
    }

    // camera timestamps
    vector<double> timestamps;
    timestamps.reserve(5000);
    loadCameraData(dataFolder + "/times.txt", timestamps);
    int numCamera = (int) timestamps.size();
    cout << "load " << numCamera << " camera timestamps" << endl;

    // imu measurements
    vector<ImuData> vecImus;
    vecImus.reserve(60000);
    loadImuData(dataFolder + "/imu.txt", vecImus);
    int numImu = (int) vecImus.size();
    cout << "load " << numImu << " imu measurements" << endl;

    // 2. create SLAM system
    System SLAM(argv[1], argv[2], true);

    // 3. main loop
    int idx1 = 0, idx2 = 0;
    vector<ImuData> curImus;
    while (vecImus[idx2 + 1].t <= timestamps[idx1]) idx2++;
    while (videoCapture.isOpened() && idx1 < numCamera) {
        cv::Mat frame;
        if (videoCapture.read(frame)) {
            cout << endl << "iter: " << idx1 << endl;

            cv::Mat grayImg;
            cv::cvtColor(frame, grayImg, cv::COLOR_BGR2GRAY);

            // load current imu measurements
            curImus.clear();
            while (idx2 < numImu && vecImus[idx2].t < timestamps[idx1]) {
                curImus.push_back(vecImus[idx2]);
                idx2++;
            }
            cout << curImus.size() << " imus" << endl;

            SLAM.Track(timestamps[idx1], grayImg, curImus);

            idx1++;
        } else {
            cout << "video end" << endl;
            break;
        }
    }

    SLAM.ShutDown();
    SLAM.saveKeyFrameTrajectory();
    SLAM.savePointCloudMap();
    SLAM.saveKeyFrameDepth();
    videoCapture.release();

    return 0;
}