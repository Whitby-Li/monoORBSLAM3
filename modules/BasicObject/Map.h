//
// Created by whitby on 8/22/23.
//

#ifndef MONO_ORB_SLAM3_MAP_H
#define MONO_ORB_SLAM3_MAP_H

#include "KeyFrame.h"
#include <set>

namespace mono_orb_slam3 {

    class KeyFrame;

    class MapPoint;

    struct KeyFrameCompare {
        bool operator()(const std::shared_ptr<mono_orb_slam3::KeyFrame> &kf1, const std::shared_ptr<mono_orb_slam3::KeyFrame> &kf2) const;
    };

    class Map {
    public:
        /// Basic
        void addKeyFrame(const std::shared_ptr<KeyFrame> &keyFrame);

        void eraseKeyFrame(const std::shared_ptr<KeyFrame> &keyFrame);

        void addMapPoint(const std::shared_ptr<MapPoint> &mapPoint);

        void eraseMapPoint(const std::shared_ptr<MapPoint> &mapPoint);

        std::vector<std::shared_ptr<KeyFrame>> getRecentKeyFrames(int num);

        int getNumKeyFrames();

        int getNumMapPoints();

        std::vector<std::shared_ptr<KeyFrame>> getAllKeyFrames();

        std::vector<std::shared_ptr<MapPoint>> getAllMapPoints();

        void clear();

        /// Viewer
        void setReferenceMapPoints(const std::vector<std::shared_ptr<MapPoint>> &mapPoints);

        std::set<std::shared_ptr<MapPoint>> getReferenceMapPoints();

        void applyScaleRotation(const Eigen::Matrix3f& Rwg, float s, bool beFirst);

        void increaseChangeIdx();

        int getMapChangeIdx();

        void updateRecordChangeIdx(int idx);

        int getRecordChangeIdx();

        std::mutex map_update_mutex;

    protected:
        std::set<std::shared_ptr<KeyFrame>, KeyFrameCompare> keyframes;
        std::set<std::shared_ptr<MapPoint>> map_points;

        /* Viewer */
        std::vector<std::shared_ptr<MapPoint>> reference_map_points;

        /* statistic map updates */
        int num_map_change = 0;     // mapping thread update map, num_map_change++
        int record_changes = 0;     // tracking thread recorded change times. if 'num_map_change' > 'record_change', map finish updating

        std::mutex map_mutex;
    };

} // mono_orb_slam3

#endif //MONO_ORB_SLAM3_MAP_H
