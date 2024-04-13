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
    if (argc != 4) {
        cout << "Usage: ./initialize_test setting.yaml vocabulary_file data_folder" << endl;
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

    // 3. main loop
    int idx = 0;
    while (idx < num_camera) {
        cout << endl << "iter: " << idx << endl;
        cv::Mat img = cv::imread(dataFolder + cv::format("/images/%08d.png", idx), cv::IMREAD_GRAYSCALE);
        if (img.empty()) {
            cout << "fail to load image" << endl;
            break;
        }

        SLAM.Track(timestamps[idx], img, {});

        if (SLAM.getTrackingState() == Tracking::OK) break;
        idx++;
    }

    SLAM.ShutDown();

    return 0;
}