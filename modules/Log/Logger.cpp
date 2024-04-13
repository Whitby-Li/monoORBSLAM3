//
// Created by whitby on 8/24/23.
//

#include <Eigen/Geometry>
#include "Logger.h"

using namespace std;

namespace mono_orb_slam3 {
    string const titles[3] = {"  ", "    ", "      "};
    const string PROJECT_PATH = "/home/whitby/Development/CLionProjects/slam/mono_orb_slam3";
    const string log_folder = PROJECT_PATH + "/logs";

    Logger initial_logger(log_folder + "/initial_log.txt");
    Logger mapper_logger(log_folder + "/mapper_log.txt");
    Logger tracker_logger(log_folder + "/tracker_log.txt");

    int Logger::iter = 0;
    mutex Logger::iter_mutex;

    Logger::Logger(const std::string &path) {
        lout.open(path);
    }

    void Logger::flush() {
        lout.flush();
    }

    void Logger::recordIter() {
        std::unique_lock<std::mutex> lock(iter_mutex);
        lout << "iter: " << iter << "\n";
    }

    Logger &Logger::operator<<(int x) {
        lout << x;
        return *this;
    }

    Logger &Logger::operator<<(size_t x) {
        lout << x;
        return *this;
    }

    Logger &Logger::operator<<(float f) {
        lout << f;
        return *this;
    }

    Logger &Logger::operator<<(double d) {
        lout << d;
        return *this;
    }

    Logger &Logger::operator<<(const string &str) {
        lout << str;
        return *this;
    }

    Logger &Logger::operator<<(const Eigen::Matrix3f &R) {
        Eigen::Quaternionf q(R);
        q.normalize();
        lout << q.x() << ", " << q.y() << ", " << q.z() << ", " << q.w();
        return *this;
    }

    Logger &Logger::operator<<(const Eigen::Vector3f &t) {
        lout << t.x() << ", " << t.y() << ", " << t.z();
        return *this;
    }

    Logger &Logger::operator<<(const Pose &pose) {
        *this << "[" << pose.R << " | " << pose.t << "]";
        return *this;
    }

    Logger &Logger::operator<<(const Bias &bias) {
        lout << "[" << bias.bg.x() << ", " << bias.bg.y() << ", " << bias.bg.z() << " | " << bias.ba.x() << ", "
             << bias.ba.y() << ", " << bias.ba.z() << "]";
        return *this;
    }

} // mono_orb_slam3