//
// Created by whitby on 8/22/23.
//

#ifndef MONO_ORB_SLAM3_KEYFRAME_H
#define MONO_ORB_SLAM3_KEYFRAME_H

#include "Frame.h"
#include "BasicObject/Map.h"
#include "DBoW2/BowVector.h"
#include "DBoW2/FeatureVector.h"

#include <set>
#include <Eigen/Core>

namespace mono_orb_slam3 {

    class Map;

    class MapPoint;

    class Frame;

    const int CONNECT_TH = 15;

    class KeyFrame : public std::enable_shared_from_this<KeyFrame> {
    public:
        KeyFrame() = delete;

        KeyFrame(const std::shared_ptr<Frame> &frame, Map *pointMap);

        /// Pose
        void setPose(const Eigen::Matrix3f &Rcw, const Eigen::Vector3f &tcw);

        void setImuPose(const Eigen::Matrix3f &Rwb, const Eigen::Vector3f &twb);

        Pose getPose();

        Pose getInversePose();

        Eigen::Vector3f getCameraCenter();

        Pose getImuPose();

        void setVelocity(const Eigen::Vector3f &velo);

        void setTrackVelocity(const Eigen::Vector3f &velo);

        Eigen::Vector3f getVelocity();

        void setImuBias(const Bias &newBias);

        void setPrioriInformation(const Eigen::Matrix<float, 15, 15> &C);

        void computePreIntegration(const std::vector<ImuData> &imus, double endTime) const;

        /// Map Points
        std::vector<std::shared_ptr<MapPoint>> getMapPoints();

        bool hasMapPoint(size_t idx);

        std::shared_ptr<MapPoint> getMapPoint(size_t idx);

        void eraseMapPoint(size_t idx);

        std::vector<std::shared_ptr<MapPoint>> getTrackedMapPoints();

        int getNumTrackedMapPoint(int minObs = 2);

        void addMapPoint(const std::shared_ptr<MapPoint> &mapPoint, size_t idx);

        float computeSceneMedianDepth();

        /// ORB
        [[nodiscard]] std::vector<size_t> getFeaturesInArea(const float &x, const float &y, const float &r,
                                                            int minLevel = -1, int maxLevel = -1) const;

        void computeBow();

        /// Covisibility Graph
        void updateConnections();

        void addConnection(const std::shared_ptr<KeyFrame> &keyFrame, int weight);

        void eraseConnection(const std::shared_ptr<KeyFrame> &keyFrame);

        void updateBestCovisibles();

        std::vector<std::shared_ptr<KeyFrame>> getBestCovisibleKFs(int num = 5);

        std::vector<std::shared_ptr<KeyFrame>> getConnectedKFs();

        std::vector<int> getConnectedWeights();

        int getWeight(const std::shared_ptr<KeyFrame> &keyFrame);

        void addChild(const std::shared_ptr<KeyFrame> &keyFrame);

        void eraseChild(const std::shared_ptr<KeyFrame> &keyFrame);

        std::set<std::shared_ptr<KeyFrame>> getChild();

        bool hasChild(const std::shared_ptr<KeyFrame> &keyFrame);

        std::shared_ptr<KeyFrame> getParent();

        void changeParent(const std::shared_ptr<KeyFrame> &keyFrame);

        bool isBad();

        void setBad();

        /* Basic */
        static long unsigned int next_id;
        long unsigned int id;
        const long unsigned int frame_id;
        double timestamp;
        std::shared_ptr<PreIntegrator> pre_integrator;      // integrate imu measurements from previous to current

        /* Imu priori */
        Eigen::Vector3f velo_priori;
        Eigen::Matrix3f velo_info;
        Bias bias_priori;
        Eigen::Matrix3f gyro_info;
        Eigen::Matrix3f acc_info;

        /* ORB */
        const std::vector<cv::KeyPoint> raw_key_points;
        const std::vector<cv::KeyPoint> key_points;
        const int num_kps;
        const cv::Mat descriptors;
        DBoW2::BowVector bow_vector;
        DBoW2::FeatureVector feature_vector;

        // key-points are assigned to cells in a grid, reducing matching complexity with a match position priori
        std::vector<std::vector<std::vector<size_t>>> grid;

        /* Tracking */
        unsigned int track_frame_id = 0;

        /* Local Mapping */
        unsigned int fuse_target_for_kf = 0;
        unsigned int BA_local_for_kf = 0;
        unsigned int BA_fixed_for_kf = 0;

    protected:
        /* Pose */
        Pose T_cw;
        Eigen::Vector3f O_w;

        Pose T_wb;
        Eigen::Vector3f v_w;
        std::mutex pose_mutex;

        std::vector<std::shared_ptr<MapPoint>> map_points;
        std::mutex feature_mutex;

        std::unordered_map<std::shared_ptr<KeyFrame>, int> connected_kf_weights;
        std::vector<std::shared_ptr<KeyFrame>> ordered_connected_kfs;
        std::vector<int> ordered_connected_weights;

        bool be_first_connection = true;
        std::shared_ptr<KeyFrame> parent = nullptr;
        std::set<std::shared_ptr<KeyFrame>> children_set;

        bool is_bad = false;

        Map *point_map;

        std::mutex connection_mutex;
    };

} // mono_orb_slam3

#endif //MONO_ORB_SLAM3_KEYFRAME_H
