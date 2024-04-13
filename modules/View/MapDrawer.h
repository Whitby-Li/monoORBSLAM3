//
// Created by whitby on 8/25/23.
//

#ifndef MONO_ORB_SLAM3_MAPDRAWER_H
#define MONO_ORB_SLAM3_MAPDRAWER_H

#include "BasicObject/Map.h"

#include <pangolin/pangolin.h>
#include <mutex>

namespace mono_orb_slam3 {

    class MapDrawer {
    public:
        MapDrawer(Map *pointMap, const cv::FileNode &viewNode);

        void DrawMapPoints() const;

        void DrawKeyFrames(bool drawKF, bool drawGraph);

        void DrawCurrentCamera(pangolin::OpenGlMatrix &Twc) const;

        void SetCurrentCameraPose(const Pose &Tcw);

        void SetReferenceKeyFrame(const std::shared_ptr<KeyFrame> &keyFrame);

        void getCurrentOpenGLCameraMatrix(pangolin::OpenGlMatrix &matrix);

        Map *point_map;
        std::shared_ptr<KeyFrame> reference_kf;

    private:
        float kf_size;
        float kf_line_width;
        float graph_line_width;
        float point_size;
        float camera_size;
        float camera_line_width;

        Pose camera_pose;

        std::mutex map_drawer_mutex;
    };

} // mono_orb_slam3

#endif //MONO_ORB_SLAM3_MAPDRAWER_H
