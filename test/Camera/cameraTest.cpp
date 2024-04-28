//
// Created by whitby on 12/17/23.
//

#include "Sensor/Camera.h"

#include <opencv2/opencv.hpp>

using namespace std;
using namespace mono_orb_slam3;

int main() {
    const string yamlFile = "/home/whitby/Development/CLionProjects/slam/mono_orb_slam3/settings/tum.yaml";
    cv::FileStorage fs(yamlFile, cv::FileStorage::READ);
    if (!fs.isOpened()) {
        cerr << "fail to open file at " << yamlFile << endl;
        return -1;
    }

    cv::FileNode cameraNode = fs["Camera"];
    Camera::create(cameraNode);
    Camera::getCamera()->print();

    return 0;
}