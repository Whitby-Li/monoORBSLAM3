//
// Created by whitby on 8/22/23.
//

#ifndef MONO_ORB_SLAM3_SYSTEM_H
#define MONO_ORB_SLAM3_SYSTEM_H

#include "Frontend/Tracking.h"
#include "BasicObject/Map.h"
#include "View/Viewer.h"

#include <string>
#include <opencv2/core/core.hpp>
#include <Eigen/Core>
#include <thread>

namespace mono_orb_slam3 {

    class Tracking;

    class LocalMapping;

    class Viewer;

    class FrameDrawer;

    class MapDrawer;

    class System {
    public:
        System(const std::string &settingYaml, const std::string &vocabularyFile, bool useViewer = false, bool recordViewer = false);

        ~System();

        void Track(double timeStamp, cv::Mat grayImg, const std::vector<ImuData> &imus);

        void Reset();

        void ShutDown();

        void saveKeyFrameTrajectory();

        void saveKeyFrameVelocityAndBias();

        void savePointCloudMap();

        void saveKeyFrameDepth();

        int getTrackingState();

        void setSaveFolder(const std::string &path);

        std::string getSaveFolder() {
            return save_folder;
        }

    private:
        /* Map */
        Map *point_map;

        /* tracking state */
        int tracking_state;
        std::mutex state_mutex;

        /* Threads */
        Tracking *tracker;

        LocalMapping *local_mapper;
        std::thread *local_mapping;

        Viewer *viewer = nullptr;
        FrameDrawer *frame_drawer = nullptr;
        MapDrawer *map_drawer = nullptr;
        std::thread *viewing;

        bool be_reset;
        std::mutex reset_mutex;

        /* Save Folder */
        std::string save_folder;
    };

} // mono_orb_slam3

#endif //MONO_ORB_SLAM3_SYSTEM_H
