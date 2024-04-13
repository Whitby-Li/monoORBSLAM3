//
// Created by whitby on 8/22/23.
//

#include "MapPoint.h"

#include <utility>
#include "ORB/ORBMatcher.h"

using namespace std;

namespace mono_orb_slam3 {
    long unsigned int MapPoint::next_id = 0;
    mutex MapPoint::global_mutex;

    MapPoint::MapPoint(Eigen::Vector3f Pw, const std::shared_ptr<KeyFrame> &lastKF,
                       const std::shared_ptr<KeyFrame> &curKF, Match match, Map *pointMap)
            : P_w(std::move(Pw)), reference_kf(curKF), first_kf_id(curKF->id), last_frame_seen(curKF->frame_id),
              point_map(pointMap) {
        id = next_id++;

        observations[lastKF] = match.first;
        observations[curKF] = match.second;
        num_visible = 1;
        num_found = 1;
        num_obs = 2;

        descriptor = curKF->descriptors.row(match.second).clone();
        update();
    }

    void MapPoint::setPos(const Eigen::Vector3f &Pw) {
        lock_guard<mutex> lock1(global_mutex);
        lock_guard<mutex> lock2(pos_mutex);
        P_w = Pw;
    }

    Eigen::Vector3f MapPoint::getPos() {
        lock_guard<mutex> lock(pos_mutex);
        return P_w;
    }

    void MapPoint::update() {
        map<shared_ptr<KeyFrame>, size_t> obs;
        shared_ptr<KeyFrame> refKeyFrame;
        Eigen::Vector3f Pw;
        {
            lock_guard<mutex> lock1(feature_mutex);
            lock_guard<mutex> lock2(pos_mutex);
            if (is_bad) return;
            obs = observations;
            refKeyFrame = reference_kf;
            Pw = P_w;
        }

        // update average direction
        Eigen::Vector3f sumDirection = Eigen::Vector3f::Zero();
        int n = 0;
        for (const auto &obsPair: obs) {
            shared_ptr<KeyFrame> kf = obsPair.first;
            sumDirection += (Pw - kf->getCameraCenter()).normalized();
            n++;
        }

        // update distances
        const float dist = (Pw - refKeyFrame->getCameraCenter()).norm();
        const cv::KeyPoint &kp = refKeyFrame->key_points[obs[refKeyFrame]];

        {
            lock_guard<mutex> lock3(pos_mutex);
            max_distance = dist * kp.size;
            min_distance = max_distance / ORBExtractor::getMaxScaleFactor();
            average_direction = sumDirection / n;
        }

    }

    Eigen::Vector3f MapPoint::getAverageDirection() {
        lock_guard<mutex> lock(pos_mutex);
        return average_direction;
    }

    float MapPoint::getMinDistanceInvariance() {
        lock_guard<mutex> lock(pos_mutex);
        return 0.8f * min_distance;
    }

    float MapPoint::getMaxDistanceInvariance() {
        lock_guard<mutex> lock(pos_mutex);
        return 1.2f * max_distance;
    }

    size_t MapPoint::getFeatureId(const std::shared_ptr<KeyFrame> &keyFrame) {
        lock_guard<mutex> lock(feature_mutex);
        return observations[keyFrame];
    }

    bool MapPoint::isObserveKeyFrame(const std::shared_ptr<KeyFrame> &keyFrame) {
        lock_guard<mutex> lock(feature_mutex);
        return observations.count(keyFrame);
    }

    void MapPoint::computeDescriptor() {
        // retrieve all observed descriptors
        map<shared_ptr<KeyFrame>, size_t> obs;
        {
            lock_guard<mutex> lock(feature_mutex);
            if (is_bad) return;
            obs = observations;
        }

        vector<cv::Mat> descriptors;
        descriptors.reserve(obs.size());

        for (auto &obsPair: obs) {
            const shared_ptr<KeyFrame> &kf = obsPair.first;

            if (!kf->isBad())
                descriptors.push_back(kf->descriptors.row(obsPair.second));
        }

        if (descriptors.empty()) return;
        const unsigned int N = descriptors.size();

        // compute distance between them
        int distances[N][N];
        for (unsigned int i = 0; i < N; ++i) {
            distances[i][i] = 0;
            for (unsigned int j = i + 1; j < N; ++j) {
                int dist = ORBMatcher::DescriptorDistance(descriptors[i], descriptors[j]);
                distances[i][j] = distances[j][i] = dist;
            }
        }

        // take the descriptor with the least median distance to the rest
        int bestMedian = 256, bestIdx = 0;
        for (unsigned int i = 0; i < N; ++i) {
            vector<int> rowDists(distances[i], distances[i] + N);
            sort(rowDists.begin(), rowDists.end());
            int median = rowDists[(N - 1) / 2];

            if (median < bestMedian) {
                bestMedian = median;
                bestIdx = i;
            }
        }

        {
            lock_guard<mutex> lock(feature_mutex);
            descriptor = descriptors[bestIdx].clone();
        }
    }

    cv::Mat MapPoint::getDescriptor() {
        lock_guard<mutex> lock(feature_mutex);
        return descriptor.clone();
    }

    int MapPoint::predictScaleLevel(const float &dist) {
        float distRatio;
        {
            lock_guard<mutex> lock(pos_mutex);
            distRatio = max_distance / dist;
        }
        int predictLevel = cvCeil(log(distRatio) / ORBExtractor::getLogScaleFactor());
        int maxLevel = ORBExtractor::getNumLevels() - 1;
        if (predictLevel < 0) return 0;
        else if (predictLevel > maxLevel) return maxLevel;
        return predictLevel;
    }

    std::map<std::shared_ptr<KeyFrame>, size_t> MapPoint::getObservations() {
        lock_guard<mutex> lock(feature_mutex);
        return observations;
    }

    int MapPoint::getNumObs() {
        lock_guard<mutex> lock(feature_mutex);
        return num_obs;
    }

    void MapPoint::addObservation(const std::shared_ptr<KeyFrame> &keyFrame, size_t idx) {
        lock_guard<mutex> lock(feature_mutex);
        if (!observations.count(keyFrame)) {
            observations[keyFrame] = idx;
            num_obs++;
        }
    }

    void MapPoint::eraseObservation(const std::shared_ptr<KeyFrame> &keyFrame) {
        bool beBad = false;
        {
            lock_guard<mutex> lock(feature_mutex);
            if (observations.count(keyFrame)) {
                observations.erase(keyFrame);
                num_obs--;

                if (reference_kf == keyFrame)
                    reference_kf = observations.begin()->first;

                // if only 2 obs or less, discard point
                if (num_obs <= 2) beBad = true;
            }
        }

        if (beBad)
            setBad();
    }

    void MapPoint::setBad() {
        map<shared_ptr<KeyFrame>, size_t> obs;
        {
            lock_guard<mutex> lock1(feature_mutex);
            lock_guard<mutex> lock2(pos_mutex);
            is_bad = true;
            obs = observations;
            observations.clear();
            num_obs = 0;
        }

        for (const auto &obsPair: obs) {
            obsPair.first->eraseMapPoint(obsPair.second);
        }

        point_map->eraseMapPoint(shared_from_this());
    }

    bool MapPoint::isBad() {
        lock_guard<mutex> lock(feature_mutex);
        return is_bad;
    }

    void MapPoint::replace(const std::shared_ptr<MapPoint> &mapPoint) {
        if (mapPoint->id == id) return;

        int numVisible, numFound;
        map<shared_ptr<KeyFrame>, size_t> obs;
        {
            lock_guard<mutex> lock1(feature_mutex);
            lock_guard<mutex> lock2(pos_mutex);
            obs = observations;
            observations.clear();
            num_obs = 0;
            is_bad = true;
            numVisible = num_visible;
            numFound = num_found;
        }

        for (const auto &obsPair: obs) {
            const shared_ptr<KeyFrame> &kf = obsPair.first;
            if (!mapPoint->isObserveKeyFrame(kf)) {
                kf->addMapPoint(mapPoint, obsPair.second);
                mapPoint->addObservation(kf, obsPair.second);
            } else {
                kf->eraseMapPoint(obsPair.second);
            }
        }

        mapPoint->increaseFound(numFound);
        mapPoint->increaseFound(numVisible);
        mapPoint->computeDescriptor();

        point_map->eraseMapPoint(shared_from_this());
    }

    void MapPoint::increaseVisible(int n) {
        lock_guard<mutex> lock(feature_mutex);
        num_visible += n;
    }

    void MapPoint::increaseFound(int n) {
        lock_guard<mutex> lock(feature_mutex);
        num_found += n;
    }

    float MapPoint::getFoundRatio() {
        lock_guard<mutex> lock(feature_mutex);
        return float(num_found) / float(num_visible);
    }
} // mono_orb_slam3