//
// Created by whitby on 8/22/23.
//

#include "Map.h"

using namespace std;

namespace mono_orb_slam3 {

    bool KeyFrameCompare::operator()(const std::shared_ptr<mono_orb_slam3::KeyFrame> &kf1,
                                     const std::shared_ptr<mono_orb_slam3::KeyFrame> &kf2) const {
        return kf1->id < kf2->id;
    }

    void Map::addKeyFrame(const shared_ptr <KeyFrame> &keyFrame) {
        lock_guard<mutex> lock(map_mutex);
        keyframes.insert(keyFrame);
    }

    void Map::eraseKeyFrame(const shared_ptr <KeyFrame> &keyFrame) {
        lock_guard<mutex> lock(map_mutex);
        auto iter = keyframes.find(keyFrame);
        if (iter == keyframes.end() || iter == keyframes.begin()) return;

        const shared_ptr<KeyFrame> &preKF = *(--iter);
        preKF->pre_integrator->MergeNext(keyFrame->pre_integrator);

        keyframes.erase((++iter));
    }

    void Map::addMapPoint(const shared_ptr <MapPoint> &mapPoint) {
        lock_guard<mutex> lock(map_mutex);
        map_points.insert(mapPoint);
    }

    void Map::eraseMapPoint(const shared_ptr <MapPoint> &mapPoint) {
        lock_guard<mutex> lock(map_mutex);
        map_points.erase(mapPoint);
    }

    vector<std::shared_ptr<KeyFrame>> Map::getRecentKeyFrames(int num) {
        lock_guard<mutex> lock(map_mutex);
        auto iterEnd = keyframes.end();
        if (num > keyframes.size()) {
            return {keyframes.begin(), iterEnd};
        }
        auto iterStart = iterEnd;
        while (num-- > 0) {
            iterStart--;
        }
        return {iterStart, iterEnd};
    }

    int Map::getNumKeyFrames() {
        lock_guard<mutex> lock(map_mutex);
        return (int) keyframes.size();
    }

    int Map::getNumMapPoints() {
        lock_guard<mutex> lock(map_mutex);
        return (int) map_points.size();
    }

    vector <shared_ptr<KeyFrame>> Map::getAllKeyFrames() {
        lock_guard<mutex> lock(map_mutex);
        return {keyframes.begin(), keyframes.end()};
    }

    std::vector<std::shared_ptr<MapPoint>> Map::getAllMapPoints() {
        lock_guard<mutex> lock(map_mutex);
        return {map_points.begin(), map_points.end()};
    }

    void Map::clear() {
        map_points.clear();
        keyframes.clear();

        num_map_change = 0;
        record_changes = 0;
    }

    void Map::setReferenceMapPoints(const std::vector<std::shared_ptr<MapPoint>> &mapPoints) {
        lock_guard<mutex> lock(map_mutex);
        reference_map_points = mapPoints;
    }

    std::set<std::shared_ptr<MapPoint>> Map::getReferenceMapPoints() {
        lock_guard<mutex> lock(map_mutex);
        if (reference_map_points.empty()) {
            return map_points;
        }
        return {reference_map_points.begin(), reference_map_points.end()};
    }

    void Map::applyScaleRotation(const Eigen::Matrix3f &Rwy, const float s, bool beFirst) {
        Eigen::Matrix3f Ryw = Rwy.transpose();

        lock_guard<mutex> lock(map_mutex);
        for (const auto &kf: keyframes) {
            Pose Tcw = kf->getPose();
            kf->setPose(Tcw.R * Rwy, Tcw.t * s);

            Eigen::Vector3f Vw = kf->getVelocity();
            if (beFirst) {
                kf->setVelocity(Ryw * Vw * s);
            } else {
                kf->setVelocity(Ryw * Vw);
            }
        }

        for (const auto &mp: map_points) {
            mp->setPos(Ryw * mp->getPos() * s);
            mp->update();
        }

        num_map_change++;
    }

    void Map::increaseChangeIdx() {
        lock_guard<mutex> lock(map_mutex);
        num_map_change++;
    }

    int Map::getMapChangeIdx() {
        lock_guard<mutex> lock(map_mutex);
        return num_map_change;
    }

    void Map::updateRecordChangeIdx(int idx) {
        lock_guard<mutex> lock(map_mutex);
        record_changes = idx;
    }

    int Map::getRecordChangeIdx() {
        lock_guard<mutex> lock(map_mutex);
        return record_changes;
    }

} // mono_orb_slam3