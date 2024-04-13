//
// Created by whitby on 8/22/23.
//

#ifndef MONO_ORB_SLAM3_FRAME_H
#define MONO_ORB_SLAM3_FRAME_H

#include "Pose.h"
#include "ORB/ORBExtractor.h"
#include "Sensor/Imu.h"
#include "MapPoint.h"
#include "DBoW2/BowVector.h"
#include "DBoW2/FeatureVector.h"

#include <Eigen/Core>

namespace mono_orb_slam3 {
    const int GRID_SIZE = 40;

    class MapPoint;

    class Frame {
    public:
        Frame() = delete;

        Frame(cv::Mat &grayImg, const double &timeStamp, ORBExtractor *orbExtractor, const Bias &bias);

        /// Pose
        void setPose(const Pose &Tcw);

        void setImuPoseAndVelocity(const Pose &Twb, const Eigen::Vector3f &Vw);

        void computePreIntegration(const std::vector<ImuData> &imus, double endTime) const;

        /// ORB
        // Compute the cell coordinate of a key-point (return false if outside the grid)
        bool PosInGrid(const cv::KeyPoint &kp, int &gridX, int &gridY) const;

        [[nodiscard]] std::vector<size_t> getFeaturesInArea(const float &x, const float &y, const float &r,
                                                            int minLevel = -1, int maxLevel = -1) const;

        /// Tracking
        [[nodiscard]] bool isInFrustum(const std::shared_ptr<MapPoint> &mapPoint, float viewCosLimit) const;

        void computeBow();

        /* Basic */
        static long unsigned int next_id;
        long unsigned int id = 0;
        const double timestamp;
        cv::Mat img;

        /* Camera Pose */
        Pose T_cw;
        Eigen::Vector3f O_w;

        /* Imu Pose */
        Pose T_wb;
        Eigen::Vector3f v_w;
        std::shared_ptr<PreIntegrator> pre_integrator;      // integrate imu measurements from previous to current

        /* ORB */
        ORBExtractor *orb_extractor = nullptr;
        std::vector<cv::KeyPoint> key_points;
        std::vector<cv::KeyPoint> raw_key_points;
        int num_kps = 0;                        // num of key points
        cv::Mat descriptors;                    // ORB descriptor, each row associated to a keypoint
        DBoW2::BowVector bow_vector;
        DBoW2::FeatureVector feature_vector;

        static int GRID_COLS, GRID_ROWS;
        static bool grid_size_computed;
        // key-points are assigned to cells in a grid, reducing matching complexity with a match position priori
        std::vector<std::vector<std::vector<size_t>>> grid;

        /* Tracking */
        std::vector<std::shared_ptr<MapPoint>> map_points;
    };

} // mono_orb_slam3

#endif //MONO_ORB_SLAM3_FRAME_H
