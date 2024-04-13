//
// Created by whitby on 8/22/23.
//

#include "KeyFrame.h"
#include "ORB/ORBVocabulary.h"

#include <thread>

using namespace std;

namespace mono_orb_slam3 {
    long unsigned int KeyFrame::next_id = 0;

    KeyFrame::KeyFrame(const shared_ptr<Frame> &frame, Map *pointMap)
            : frame_id(frame->id), timestamp(frame->timestamp), point_map(pointMap),
              raw_key_points(frame->raw_key_points), key_points(frame->key_points),
              num_kps(frame->num_kps), descriptors(frame->descriptors.clone()), grid(frame->grid),
              T_cw(frame->T_cw), O_w(frame->O_w), T_wb(frame->T_wb), v_w(frame->v_w), map_points(frame->map_points) {
        id = next_id++;

        pre_integrator = make_shared<PreIntegrator>(frame->pre_integrator);
        velo_priori = v_w;
        bias_priori = frame->pre_integrator->updated_bias;
    }

    void KeyFrame::setPose(const Eigen::Matrix3f &Rcw, const Eigen::Vector3f &tcw) {
        lock_guard<mutex> lock(pose_mutex);
        T_cw.R = Rcw, T_cw.t = tcw;
        O_w = -Rcw.transpose() * tcw;

        const ImuCalib *imu_ptr = ImuCalib::getImuCalib();
        T_wb = T_cw.inverse() * imu_ptr->T_cb;
    }

    void KeyFrame::setImuPose(const Eigen::Matrix3f &Rwb, const Eigen::Vector3f &twb) {
        lock_guard<mutex> lock(pose_mutex);
        T_wb.R = Rwb, T_wb.t = twb;

        const ImuCalib *imu_ptr = ImuCalib::getImuCalib();
        T_cw = imu_ptr->T_cb * T_wb.inverse();
        O_w = -T_cw.R.transpose() * T_cw.t;
    }

    Pose KeyFrame::getPose() {
        lock_guard<mutex> lock(pose_mutex);
        return T_cw;
    }

    Pose KeyFrame::getInversePose() {
        lock_guard<mutex> lock(pose_mutex);
        return T_cw.inverse();
    }

    Eigen::Vector3f KeyFrame::getCameraCenter() {
        lock_guard<mutex> lock(pose_mutex);
        return O_w;
    }

    Pose KeyFrame::getImuPose() {
        lock_guard<mutex> lock(pose_mutex);
        return T_wb;
    }

    void KeyFrame::setVelocity(const Eigen::Vector3f &velo) {
        lock_guard<mutex> lock(pose_mutex);
        v_w = velo;
        velo_priori = velo;
    }

    void KeyFrame::setTrackVelocity(const Eigen::Vector3f &velo) {
        lock_guard<mutex> lock(pose_mutex);
        v_w = velo;
    }

    Eigen::Vector3f KeyFrame::getVelocity() {
        lock_guard<mutex> lock(pose_mutex);
        return v_w;
    }

    void KeyFrame::setImuBias(const Bias &newBias) {
        lock_guard<mutex> lock(pose_mutex);
        pre_integrator->setNewBias(newBias);
    }

    void KeyFrame::setPrioriInformation(const Eigen::Matrix<float, 15, 15> &C) {
        velo_info = C.block<3, 3>(3, 3);
        velo_info = (velo_info + velo_info.transpose()) / 2;
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> eigenSolver(velo_info);
        Eigen::Vector3f eigenValues = eigenSolver.eigenvalues();
        for (int i = 0; i < 3; ++i) {
            if (eigenValues[i] < 1e-12) eigenValues[i] = 0;
        }
        velo_info = eigenSolver.eigenvectors() * eigenValues.asDiagonal() * eigenSolver.eigenvectors().transpose();

        gyro_info = C.block<3, 3>(9, 9).inverse();
        acc_info = C.block<3, 3>(12, 12).inverse();
    }

    void KeyFrame::computePreIntegration(const std::vector<ImuData> &imus, double endTime) const {
        double startTime = timestamp + pre_integrator->delta_t;
        if (imus.size() == 1) {
            pre_integrator->IntegrateNewMeasurement(imus[0].w, imus[0].a, endTime - startTime);
            return;
        }

        for (int i = 0, end = (int) imus.size(); i < end; ++i) {
            if (i == 0)
                pre_integrator->IntegrateNewMeasurement(imus[i].w, imus[i].a, imus[i + 1].t - startTime);
            else if (i == end - 1)
                pre_integrator->IntegrateNewMeasurement(imus[i].w, imus[i].a, endTime - imus[i].t);
            else
                pre_integrator->IntegrateNewMeasurement(imus[i].w, imus[i].a, imus[i + 1].t - imus[i].t);
        }
    }

    vector<std::shared_ptr<MapPoint>> KeyFrame::getMapPoints() {
        lock_guard<mutex> lock(feature_mutex);
        return map_points;
    }

    bool KeyFrame::hasMapPoint(size_t idx) {
        lock_guard<mutex> lock(feature_mutex);
        return map_points[idx] != nullptr;
    }

    std::shared_ptr<MapPoint> KeyFrame::getMapPoint(size_t idx) {
        lock_guard<mutex> lock(feature_mutex);
        return map_points[idx];
    }

    void KeyFrame::eraseMapPoint(size_t idx) {
        lock_guard<mutex> lock(feature_mutex);
        map_points[idx] = nullptr;
    }

    std::vector<shared_ptr<MapPoint>> KeyFrame::getTrackedMapPoints() {
        vector<shared_ptr<MapPoint>> mapPoints = getMapPoints();
        vector<shared_ptr<MapPoint>> trackedMapPoints;
        for (auto &mp: mapPoints) {
            if (mp && !mp->isBad()) trackedMapPoints.push_back(mp);
        }
        return trackedMapPoints;
    }

    int KeyFrame::getNumTrackedMapPoint(int minObs) {
        lock_guard<mutex> lock(feature_mutex);
        int num = 0;
        for (const auto &mp: map_points)
            if (mp && mp->getNumObs() >= minObs) num++;
        return num;
    }

    void KeyFrame::addMapPoint(const shared_ptr<MapPoint> &mapPoint, size_t idx) {
        lock_guard<mutex> lock(feature_mutex);
        map_points[idx] = mapPoint;
    }

    float KeyFrame::computeSceneMedianDepth() {
        vector<shared_ptr<MapPoint>> mapPoints = getMapPoints();
        const Pose Tcw = getPose();
        vector<float> vecDepth;
        vecDepth.reserve(1000);

        Eigen::Matrix<float, 1, 3> Rcw2 = Tcw.R.row(2);
        float zcw = Tcw.t[2];
        for (int i = 0; i < num_kps; ++i) {
            if (mapPoints[i] != nullptr) {
                Eigen::Vector3f Pw = mapPoints[i]->getPos();
                float z = Rcw2.dot(Pw) + zcw;
                vecDepth.push_back(z);
            }
        }

        sort(vecDepth.begin(), vecDepth.end());
        int numDepth = (int) vecDepth.size();
        float medianDepth = vecDepth[numDepth / 2];
        return medianDepth;
    }

    vector<size_t> KeyFrame::getFeaturesInArea(const float &x, const float &y, const float &r, int minLevel,
                                               int maxLevel) const {
        const int minCellX = max(0, cvFloor(x - r) / GRID_SIZE);
        const int maxCellX = min(Frame::GRID_COLS - 1, cvFloor(x + r) / GRID_SIZE);
        if (minCellX > maxCellX) return {};
        const int minCellY = max(0, cvFloor(y - r) / GRID_SIZE);
        const int maxCellY = min(Frame::GRID_ROWS - 1, cvFloor(y + r) / GRID_SIZE);
        if (minCellY > maxCellY) return {};

        vector<size_t> indices;
        const bool beCheckLevel = minLevel > 0 || maxLevel >= 0;
        for (int cx = minCellX; cx <= maxCellX; ++cx) {
            for (int cy = minCellY; cy <= maxCellY; ++cy) {
                const vector<size_t> cellIndices = grid[cx][cy];
                if (cellIndices.empty()) continue;

                for (auto idx: cellIndices) {
                    const cv::KeyPoint &kp = key_points[idx];
                    if (beCheckLevel) {
                        if (kp.octave < minLevel) continue;
                        if (maxLevel >= 0 && kp.octave > maxLevel) continue;
                    }

                    if (abs(kp.pt.x - x) < r && abs(kp.pt.y - y) < r)
                        indices.push_back(idx);
                }
            }
        }

        return indices;
    }

    void KeyFrame::computeBow() {
        if (bow_vector.empty() || feature_vector.empty()) {
            vector<cv::Mat> vecDescriptor(num_kps);
            for (int i = 0; i < num_kps; ++i) {
                vecDescriptor[i] = descriptors.row(i);
            }

            const Vocabulary *vocabulary = ORBVocabulary::getORBVocabulary();
            vocabulary->transform(vecDescriptor, bow_vector, feature_vector, 4);
        }
    }

    void KeyFrame::updateConnections() {
        unordered_map<shared_ptr<KeyFrame>, int> kfCounter;
        vector<shared_ptr<MapPoint>> mapPoints;
        {
            lock_guard<mutex> lock(feature_mutex);
            mapPoints = map_points;
        }

        // for all map-points in keyframe check in which other keyframes are they seen
        // increase counter for those keyframes
        for (const auto &mp: mapPoints) {
            if (!mp || mp->isBad()) continue;
            map<shared_ptr<KeyFrame>, size_t> observations = mp->getObservations();
            for (const auto &obsPair: observations) {
                if (obsPair.first->id == id) continue;
                kfCounter[obsPair.first]++;
            }
        }

        if (kfCounter.empty()) return;

        // if the counter is greater than threshold, add connection
        // in case no keyframe counter is over threshold, add the one with maximum counter
        int maxObs = 0;
        shared_ptr<KeyFrame> maxKF;

        vector<pair<int, shared_ptr<KeyFrame>>> vecPairs;
        vecPairs.reserve(kfCounter.size());
        for (const auto &counterPair: kfCounter) {
            if (counterPair.second > maxObs) {
                maxObs = counterPair.second;
                maxKF = counterPair.first;
            }

            if (counterPair.second >= CONNECT_TH) {
                vecPairs.emplace_back(counterPair.second, counterPair.first);
                counterPair.first->addConnection(shared_from_this(), counterPair.second);
            }
        }

        if (vecPairs.empty()) {
            vecPairs.emplace_back(maxObs, maxKF);
            maxKF->addConnection(shared_from_this(), maxObs);
        }

        sort(vecPairs.begin(), vecPairs.end());
        list<shared_ptr<KeyFrame>> listKeyFrames;
        list<int> listWeights;
        for (const auto &connectPair: vecPairs) {
            listKeyFrames.push_front(connectPair.second);
            listWeights.push_front(connectPair.first);
        }

        {
            lock_guard<mutex> lock(connection_mutex);

            connected_kf_weights = kfCounter;
            ordered_connected_kfs = vector<shared_ptr<KeyFrame>>(listKeyFrames.begin(), listKeyFrames.end());
            ordered_connected_weights = vector<int>(listWeights.begin(), listWeights.end());

            if (be_first_connection && id != 0) {
                parent = ordered_connected_kfs.front();
                parent->addChild(shared_from_this());
                be_first_connection = false;
            }
        }
    }

    void KeyFrame::addConnection(const std::shared_ptr<KeyFrame> &keyFrame, int weight) {
        {
            lock_guard<mutex> lock(connection_mutex);
            if (!connected_kf_weights.count(keyFrame)) {
                connected_kf_weights[keyFrame] = weight;
            } else if (connected_kf_weights[keyFrame] != weight) {
                connected_kf_weights[keyFrame] = weight;
            } else return;
        }

        updateBestCovisibles();
    }

    void KeyFrame::eraseConnection(const std::shared_ptr<KeyFrame> &keyFrame) {
        bool beUpdate = false;
        {
            lock_guard<mutex> lock(connection_mutex);
            if (connected_kf_weights.count(keyFrame)) {
                connected_kf_weights.erase(keyFrame);
                beUpdate = true;
            }
        }

        if (beUpdate) updateBestCovisibles();
    }

    void KeyFrame::updateBestCovisibles() {
        lock_guard<mutex> lock(connection_mutex);
        vector<pair<int, shared_ptr<KeyFrame>>> vecPairs;
        vecPairs.reserve(connected_kf_weights.size());
        for (const auto &connectPair: connected_kf_weights) {
            vecPairs.emplace_back(connectPair.second, connectPair.first);
        }

        sort(vecPairs.begin(), vecPairs.end());
        list<shared_ptr<KeyFrame>> listKFs;
        list<int> listWeights;
        for (const auto &connectPair: vecPairs) {
            listKFs.push_front(connectPair.second);
            listWeights.push_front(connectPair.first);
        }

        ordered_connected_kfs = vector<shared_ptr<KeyFrame>>(listKFs.begin(), listKFs.end());
        ordered_connected_weights = vector<int>(listWeights.begin(), listWeights.end());
    }

    vector<shared_ptr<KeyFrame>> KeyFrame::getBestCovisibleKFs(int num) {
        lock_guard<mutex> lock(connection_mutex);
        if (ordered_connected_kfs.size() < num) {
            return ordered_connected_kfs;
        } else
            return {ordered_connected_kfs.begin(), ordered_connected_kfs.begin() + num};
    }

    vector<shared_ptr<KeyFrame>> KeyFrame::getConnectedKFs() {
        lock_guard<mutex> lock(connection_mutex);
        return ordered_connected_kfs;
    }

    vector<int> KeyFrame::getConnectedWeights() {
        lock_guard<mutex> lock(connection_mutex);
        return ordered_connected_weights;
    }

    int KeyFrame::getWeight(const std::shared_ptr<KeyFrame> &keyFrame) {
        lock_guard<mutex> lock(connection_mutex);
        if (connected_kf_weights.count(keyFrame))
            return connected_kf_weights[keyFrame];
        else return 0;
    }

    void KeyFrame::addChild(const std::shared_ptr<KeyFrame> &keyFrame) {
        lock_guard<mutex> lock(connection_mutex);
        assert(keyFrame->id != id);
        children_set.insert(keyFrame);
    }

    void KeyFrame::eraseChild(const std::shared_ptr<KeyFrame> &keyFrame) {
        lock_guard<mutex> lock(connection_mutex);
        children_set.erase(keyFrame);
    }

    set<std::shared_ptr<KeyFrame>> KeyFrame::getChild() {
        lock_guard<mutex> lock(connection_mutex);
        return children_set;
    }

    bool KeyFrame::hasChild(const std::shared_ptr<KeyFrame> &keyFrame) {
        lock_guard<mutex> lock(connection_mutex);
        return children_set.count(keyFrame);
    }

    std::shared_ptr<KeyFrame> KeyFrame::getParent() {
        lock_guard<mutex> lock(connection_mutex);
        return parent;
    }

    void KeyFrame::changeParent(const std::shared_ptr<KeyFrame> &keyFrame) {
        lock_guard<mutex> lock(connection_mutex);
        assert(keyFrame->id != id);
        parent = keyFrame;
        keyFrame->addChild(shared_from_this());
    }

    bool KeyFrame::isBad() {
        lock_guard<mutex> lock(connection_mutex);
        return is_bad;
    }

    void KeyFrame::setBad() {
        for (auto &connectPair: connected_kf_weights) {
            connectPair.first->eraseConnection(shared_from_this());
        }

        vector<shared_ptr<MapPoint>> mapPoints = getMapPoints();
        for (auto &mp: mapPoints) {
            if (mp) {
                mp->eraseObservation(shared_from_this());
            }
        }

        {
            lock_guard<mutex> lock1(connection_mutex);
            lock_guard<mutex> lock2(feature_mutex);

            map_points.clear();
            connected_kf_weights.clear();
            ordered_connected_kfs.clear();

            // update spanning tree
            set<shared_ptr<KeyFrame>> parentCandidates;
            parentCandidates.insert(parent);

            // assign at each iteration on child with a parent (the pair with highest covisibility weight)
            // include that child as new parent candidate for the rest
            while (!children_set.empty()) {
                bool beContinue = false;

                int maxWeight = 0;
                shared_ptr<KeyFrame> childKF, parentKF;
                for (const auto &child_kf: children_set) {
                    if (child_kf->isBad()) continue;

                    // check if a parent candidate is connected to the keyframe
                    for (const auto &parent_kf: parentCandidates) {
                        int weight = child_kf->getWeight(parent_kf);
                        if (weight > maxWeight) {
                            childKF = child_kf;
                            parentKF = parent_kf;
                            maxWeight = weight;
                            beContinue = true;
                        }
                    }
                }

                if (beContinue) {
                    childKF->changeParent(parentKF);
                    parentKF->addChild(childKF);
                    children_set.erase(childKF);
                } else break;
            }

            // if a child has no covisibility links with any parent candidate, assign to the original parent of this keyframe
            if (!children_set.empty()) {
                for (const auto &child_kf: children_set) {
                    child_kf->changeParent(parent);
                }
            }

            parent->eraseChild(shared_from_this());
            is_bad = true;
        }

        point_map->eraseKeyFrame(shared_from_this());
    }
} // mono_orb_slam3