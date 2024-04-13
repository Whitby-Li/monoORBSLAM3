//
// Created by whitby on 8/22/23.
//

#ifndef MONO_ORB_SLAM3_DATA_H
#define MONO_ORB_SLAM3_DATA_H

#include <fstream>
#include <sstream>
#include <vector>

#include <Eigen/Core>

void loadCameraData(const std::string &path, std::vector<double> &timestamps) {
    std::ifstream fin(path);
    while (!fin.eof()) {
        std::string lineStr;
        getline(fin, lineStr);
        if (!lineStr.empty()) {
            std::stringstream ss(lineStr);
            double t;
            ss >> t;
            timestamps.push_back(t);
        }
    }
    fin.close();
}

void loadImuData(const std::string &path, std::vector<mono_orb_slam3::ImuData> &vecImu) {
    std::ifstream fin(path);
    double last_t = 0;
    while (!fin.eof()) {
        std::string lineStr;
        getline(fin, lineStr);
        if (!lineStr.empty()) {
            std::stringstream ss(lineStr);
            double t;
            ss >> t;
            if (t > last_t) {
                last_t = t;
                float gx, gy, gz, ax, ay, az;
                ss >> gx >> gy >> gz >> ax >> ay >> az;
                vecImu.emplace_back(Eigen::Vector3f(gx, gy, gz), Eigen::Vector3f(ax, ay, az), t);
            }
        }
    }

    fin.close();
}

#endif //MONO_ORB_SLAM3_DATA_H
