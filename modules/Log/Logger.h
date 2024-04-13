//
// Created by whitby on 8/24/23.
//

#ifndef MONO_ORB_SLAM3_LOGGER_H
#define MONO_ORB_SLAM3_LOGGER_H

#include "BasicObject/KeyFrame.h"

#include <fstream>
#include <mutex>

namespace mono_orb_slam3 {
    
    struct Pose;

    class Logger {
    public:
        explicit Logger(const std::string &path);

        ~Logger() { lout.close(); }

        void flush();

        void recordIter();

        Logger &operator<<(int x);

        Logger &operator<<(size_t x);

        Logger &operator<<(float x);

        Logger &operator<<(double x);

        Logger &operator<<(const std::string &str);

        Logger &operator<<(const Eigen::Matrix3f &R);

        Logger &operator<<(const Eigen::Vector3f &t);

        Logger &operator<<(const Pose &pose);

        Logger &operator<<(const Bias &bias);

        void static iterate() {
            std::unique_lock<std::mutex> lock(iter_mutex);
            iter++;
        }

    private:
        std::ofstream lout;
        static int iter;
        static std::mutex iter_mutex;
    };

    extern const std::string titles[3];
    extern const std::string PROJECT_PATH;
    extern const std::string log_folder;

    extern Logger initial_logger;
    extern Logger mapper_logger;
    extern Logger tracker_logger;

} // mono_orb_slam3

#endif //MONO_ORB_SLAM3_LOGGER_H
