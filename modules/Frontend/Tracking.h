//
// Created by whitby on 8/22/23.
//

#ifndef MONO_ORB_SLAM3_TRACKING_H
#define MONO_ORB_SLAM3_TRACKING_H

#include "System.h"
#include "TwoViewReconstruction.h"
#include "LocalMapping.h"
#include "BasicObject/KeyFrame.h"
#include "View/Viewer.h"

namespace mono_orb_slam3 {

    class System;

    class Tracking {
    public:
        enum State {
            NO_IMAGE_YET = 0,
            NOT_INITIALIZE = 1,
            OK = 2,
            RECENTLY_LOST = 3,
            LOST = 4
        };

        Tracking(const cv::FileNode &orbNode, Map *pointMap, System *system_ptr);

        ~Tracking() {
            delete initial_extractor;
            delete extractor;
            delete initial_reconstructor;
        }

        void setLocalMapper(LocalMapping *localMapper);

        void setViewer(Viewer *viewerPtr);

        void Track(double timeStamp, cv::Mat grayImg, const std::vector<ImuData> &imus);

        void updateFrameIMU();

        void predictCurFramePose();

        void predictCurFramePoseByKF();

        /// thread synchronize
        void reset();

        State state = NO_IMAGE_YET;
        State last_state = NO_IMAGE_YET;

        /* Basic */
        std::shared_ptr<Frame> last_frame, current_frame;

        /* Initialization */
        std::vector<cv::Point2f> priori_matches;
        std::vector<int> initial_matches;
        TwoViewReconstruction *initial_reconstructor;
        std::vector<cv::Point3f> initial_3d_points;

        int num_inlier = 0;

    private:
        /// Initialization
        void Initialization();

        void CreateInitialMap();

        /// Tracking
        void updateLastFramePose();

        bool trackReferenceKeyFrame();

        bool trackLastFrame();

        bool trackLastKeyFrame();

        bool trackLocalMap();

        void updateLocalMap();

        void searchLocalPoints();

        void updateLocalKeyFrames();

        void updateLocalMapPoints();

        /// Local Mapping
        bool needNewKeyFrame();

        void createNewKeyFrame();

        /* ORB */
        ORBExtractor *initial_extractor, *extractor;

        /* Map */
        Map *point_map;

        /* Tracking */
        std::shared_ptr<KeyFrame> last_kf, reference_kf;
        Pose Tlr, Tcl;
        bool have_velocity = false;
        double last_lost_time = 0;

        std::vector<std::shared_ptr<KeyFrame>> local_keyframes;
        std::vector<std::shared_ptr<MapPoint>> local_map_points;

        /* Threads */
        System *system;
        LocalMapping *local_mapper = nullptr;

        /* View */
        Viewer *viewer = nullptr;
        FrameDrawer *frame_drawer = nullptr;
        MapDrawer *map_drawer = nullptr;
        bool have_viewer = false;
    };

} // mono_orb_slam3

#endif //MONO_ORB_SLAM3_TRACKING_H
