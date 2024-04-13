//
// Created by whitby on 8/22/23.
//

#ifndef MONO_ORB_SLAM3_MAPPOINT_H
#define MONO_ORB_SLAM3_MAPPOINT_H

#include "KeyFrame.h"
#include <Eigen/Core>
#include <map>

namespace mono_orb_slam3 {

    class KeyFrame;

    class Map;

    class MapPoint : public std::enable_shared_from_this<MapPoint> {
    public:
        MapPoint() = delete;

        MapPoint(Eigen::Vector3f Pw, const std::shared_ptr<KeyFrame> &firstKF,
                 const std::shared_ptr<KeyFrame> &secondKF, Match match, Map *pointMap);

        MapPoint(const MapPoint &mapPoint) = delete;

        /// Pos
        void setPos(const Eigen::Vector3f &Pw);

        Eigen::Vector3f getPos();

        // update average_direction and min_distance, max_distance after optimization
        void update();

        Eigen::Vector3f getAverageDirection();

        float getMinDistanceInvariance();

        float getMaxDistanceInvariance();

        int predictScaleLevel(const float &dist);

        /// ORB Descriptor
        void computeDescriptor();

        cv::Mat getDescriptor();

        /// Covisibility
        size_t getFeatureId(const std::shared_ptr<KeyFrame> &keyFrame);

        bool isObserveKeyFrame(const std::shared_ptr<KeyFrame> &keyFrame);

        std::map<std::shared_ptr<KeyFrame>, size_t> getObservations();

        int getNumObs();

        void addObservation(const std::shared_ptr<KeyFrame> &keyFrame, size_t idx);

        void eraseObservation(const std::shared_ptr<KeyFrame> &keyFrame);

        void setBad();

        bool isBad();

        void replace(const std::shared_ptr<MapPoint>& mapPoint);

        /// Tracking Counter
        void increaseVisible(int n = 1);

        void increaseFound(int n = 1);

        float getFoundRatio();

        /* Basic */
        static long unsigned int next_id;
        long unsigned int id;
        long unsigned int first_kf_id;

        /* Tracking */
        float track_proj_x = -1;
        float track_proj_y = -1;
        bool track_in_view = false;
        int track_scale_level = -1;
        float track_view_cos = 0;
        unsigned int last_frame_seen;
        unsigned int track_frame_id = 0;

        /* Local Mapping */
        unsigned int BA_local_for_kf = 0;
        unsigned int fuse_candidate_for_kf = 0;

        static std::mutex global_mutex;

    protected:
        /* Position */
        Eigen::Vector3f P_w;
        Eigen::Vector3f average_direction;
        std::shared_ptr<KeyFrame> reference_kf;
        int num_obs;
        std::mutex pos_mutex;

        /* ORB */
        cv::Mat descriptor;
        float max_distance{}, min_distance{};

        /* Covisibility Graph */
        std::map<std::shared_ptr<KeyFrame>, size_t> observations;
        bool is_bad = false;

        /* Tracking counters */
        int num_visible;
        int num_found;

        std::mutex feature_mutex;

        Map *point_map;
    };

} // mono_orb_slam3

#endif //MONO_ORB_SLAM3_MAPPOINT_H
