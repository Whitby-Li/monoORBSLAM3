//
// Created by whitby on 8/26/23.
//

#ifndef MONO_ORB_SLAM3_VIEWER_H
#define MONO_ORB_SLAM3_VIEWER_H

#include "FrameDrawer.h"
#include "MapDrawer.h"
#include "System.h"

#include <mutex>

namespace mono_orb_slam3 {

    class Tracking;

    class System;

    class FrameDrawer;

    class MapDrawer;

    class Viewer {
    public:
        Viewer(System *systemPtr, FrameDrawer *frameDrawer, MapDrawer *mapDrawer, const cv::FileNode &viewNode, bool bRecord);

        void Run();

        void requestFinish();

        void requestStop();

        bool isFinished();

        bool isStopped();

        void release();

        FrameDrawer *frame_drawer;
        MapDrawer *map_drawer;

    private:
        bool stop();

        bool checkFinish();

        void setFinish();

        System *system;

        int width, height;
        int fps;
        int delta_ms;
        float view_point_x, view_point_y, view_point_z, view_point_f;
        bool record;

        bool finish_requested;
        bool be_finished;
        std::mutex finish_mutex;

        bool be_stopped;
        bool stop_requested;
        std::mutex stop_mutex;
    };

} // mono_orb_slam3

#endif //MONO_ORB_SLAM3_VIEWER_H
