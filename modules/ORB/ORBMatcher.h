//
// Created by whitby on 8/22/23.
//

#ifndef MONO_ORB_SLAM3_ORBMATCHER_H
#define MONO_ORB_SLAM3_ORBMATCHER_H

#include "BasicObject/Frame.h"
#include "BasicObject/Map.h"

namespace mono_orb_slam3 {
    class ORBMatcher {
    public:
        explicit ORBMatcher(float nnRatio = 0.6, bool checkOrientation = true)
                : nn_ratio(nnRatio), be_check_orientation(checkOrientation) {}

        // Computes the Hamming distance between two Pyramid descriptors
        static int DescriptorDistance(const cv::Mat &a, const cv::Mat &b);

        /// Initialization
        int SearchForInitialization(const std::shared_ptr<Frame> &frame1, const std::shared_ptr<Frame> &frame2,
                                    std::vector<cv::Point2f> &vecPreMatched,
                                    std::vector<int> &matches12, int windowSize = 100) const;

        /// Tracking
        [[nodiscard]] int SearchByBow(const std::shared_ptr<KeyFrame> &keyFrame, const std::shared_ptr<Frame> &frame) const;

        [[nodiscard]] int
        SearchByProjection(const std::shared_ptr<Frame> &lastFrame, const std::shared_ptr<Frame> &curFrame,
                           float th = 5) const;

        [[nodiscard]] int SearchByProjection(const std::shared_ptr<KeyFrame> &lastKF, const std::shared_ptr<Frame> &curFrame,
                                             float th = 5) const;

        [[nodiscard]] int
        SearchByProjection(const std::shared_ptr<Frame> &frame, const std::vector<std::shared_ptr<MapPoint>> &mapPoints,
                           float th = 3) const;

        /// Local Mapping
        int
        SearchForTriangulation(const std::shared_ptr<KeyFrame> &keyFrame1, const std::shared_ptr<KeyFrame> &keyFrame2,
                               std::vector<int> &matches12) const;

        static int SearchByProjection(const std::shared_ptr<KeyFrame> &keyFrame,
                                      const std::vector<std::shared_ptr<MapPoint>> &mapPoints, Map *pointMap, float th = 3);

    protected:
        static void ComputeThreeMaxima(std::vector<int> *histo, int &ind1, int &ind2, int &ind3);

        float nn_ratio;
        bool be_check_orientation;
    };
} // mono_orb_slam3


#endif //MONO_ORB_SLAM3_ORBMATCHER_H
