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
    if (!fin.is_open()) {
        std::cerr << "Could not open file " << path << std::endl;
        return;
    }

    std::string lineStr;
    std::getline(fin, lineStr);
    while (std::getline(fin, lineStr)) {
        std::stringstream ss(lineStr);
        std::string itemStr;
        getline(ss, itemStr, ',');
        timestamps.push_back(std::stod(itemStr));
    }

    fin.close();
}

void loadImuData(const std::string &path, std::vector<mono_orb_slam3::ImuData> &vecImu) {
    std::ifstream fin(path);
    if (!fin.is_open()) {
        std::cerr << "Could not open file " << path << std::endl;
        return;
    }

    double last_t = 0;
    std::string lineStr;
    std::getline(fin, lineStr);
    while (std::getline(fin, lineStr)) {
        std::stringstream ss(lineStr);
        std::string itemStr;
        std::vector<double> rowData;
        while (std::getline(ss, itemStr, ',')) {
            rowData.push_back(std::stod(itemStr));
        }

        if (rowData[0] > last_t) {
            last_t = rowData[0];
            vecImu.emplace_back(Eigen::Vector3f(rowData[1], rowData[2], rowData[3]), Eigen::Vector3f(rowData[4], rowData[5], rowData[6]), rowData[0]);
        }
    }

    fin.close();
}

#endif //MONO_ORB_SLAM3_DATA_H
