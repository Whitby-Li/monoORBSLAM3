//
// Created by whitby on 8/22/23.
//

#ifndef MONO_ORB_SLAM3_LOCALMAPPING_H
#define MONO_ORB_SLAM3_LOCALMAPPING_H

#include "BasicObject/Map.h"
#include "Tracking.h"

namespace mono_orb_slam3 {

    class Tracking;

    class LocalMapping {
    public:
        enum ImuState {
                NOT_INITIALIZE = 0,
                INITIALIZED = 1,
                INITIALIZED_AGAIN = 2,
                FINISH = 3
        };

        explicit LocalMapping(Map *pointMap) : point_map(pointMap) {};

        /// Main function
        void Run();

        void addNewKeyFrame(const std::shared_ptr<KeyFrame> &keyFrame);

        /// thread synchronize
        void setTracker(Tracking *trackerPtr);

        bool acceptKeyFrames();

        void setAcceptKeyFrame(bool flag);

        void requestStop();

        void requestReset();

        bool stop();

        void release();

        bool isStopped();

        bool stopRequested();

        bool setNotStop(bool flag);

        void interruptBA();

        void requestFinish();

        bool isFinish();

        int numKeyFramesInQueue();

        bool isImuInitialized();

        bool finishImuInit();

        bool isInitializing() const;

    protected:
        bool checkNewKeyFrame();

        bool getNewKeyFrame();

        void processNewKeyFrame();

        void MapPointCulling();

        void createNewMapPoints();

        void searchInNeighbors();

        void KeyFrameCulling();

        void initializeIMU(float prioriG = 1e2, float prioriA = 1e6, bool beFirst = false);

        void gravityRefinement();

        bool checkFinish();

        void setFinish();

        void resetIfRequested();

        /* thread */
        Tracking *tracker = nullptr;

        ImuState imu_state = NOT_INITIALIZE;

        bool imu_initializing = false;

        bool reset_requested = false;
        std::mutex reset_mutex;

        bool request_finish = false;
        bool finished = true;
        std::mutex finish_mutex;

    private:
        Map *point_map;

        std::list<std::shared_ptr<MapPoint>> recent_map_points;

        double last_inertial_time = 0;

        std::shared_ptr<KeyFrame> current_kf = nullptr;
        std::list<std::shared_ptr<KeyFrame>> new_keyframes;
        bool accept_new_kf = true;
        bool abort_BA = false;
        std::mutex new_kf_mutex;

        bool stopped = false;
        bool stop_requested = false;
        bool not_stop = false;
        std::mutex stop_mutex;
    };

} // mono_orb_slam3

#endif //MONO_ORB_SLAM3_LOCALMAPPING_H
