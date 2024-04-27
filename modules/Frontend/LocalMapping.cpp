//
// Created by whitby on 8/22/23.
//

#include "LocalMapping.h"
#include "Sensor/Camera.h"
#include "ORB/ORBMatcher.h"
#include "Backend/Optimize.h"
#include "TwoViewReconstruction.h"
#include "Utils/LieAlgeBra.h"
#include "Log/Logger.h"

#include <unistd.h>

using namespace std;

namespace mono_orb_slam3 {

    void LocalMapping::Run() {
        finished = false;
        while (true) {
            // tracking will see that local mapping is busy
            setAcceptKeyFrame(false);

            if (getNewKeyFrame()) {
                mapper_logger.recordIter();

                // process new keyframe
                processNewKeyFrame();

                // check recent map-points
                MapPointCulling();
                mapper_logger.flush();

                mapper_logger << "have a new keyframe (id " << current_kf->id << ")\n";
                createNewMapPoints();
                mapper_logger.flush();

                searchInNeighbors();
                mapper_logger.flush();

                abort_BA = false;

                if (!checkNewKeyFrame() && !stopRequested()) {
                    if (current_kf->id > 1) {
                        if (imu_state != NOT_INITIALIZE) {
                            if (tracker->state == Tracking::RECENTLY_LOST)
                                Optimize::localInertialBundleAdjustment(current_kf, point_map, true);
                            Optimize::localFullBundleAdjustment(current_kf, point_map, false, &abort_BA);
                        } else {
                            Optimize::localBundleAdjustment(current_kf, point_map, &abort_BA);
                        }
                        mapper_logger.flush();
                    }

                    // initialize imu here
                    if (imu_state == NOT_INITIALIZE && current_kf->id > 15) {
                        mapper_logger << "try to initialize imu\n";
                        initializeIMU(1e+6, 1e+12, true);
                    }

                    KeyFrameCulling();

                    if (imu_state == INITIALIZED && current_kf->timestamp - last_inertial_time > 3.0) {
                        gravityRefinement();
                    }

                }

                mapper_logger << "\n";
                mapper_logger.recordIter();
                mapper_logger << "\n";
                mapper_logger.flush();
            }

            resetIfRequested();

            setAcceptKeyFrame(true);

            if (checkFinish()) break;

            usleep(3000);
        }

        setFinish();
    }

    void LocalMapping::processNewKeyFrame() {
        mapper_logger << "processNewKeyFrame\n";
        current_kf->computeBow();

        // associate map-points to new keyframe and update normal and descriptor
        const vector<shared_ptr<MapPoint>> mapPoints = current_kf->getMapPoints();
        for (int i = 0; i < current_kf->num_kps; ++i) {
            const shared_ptr<MapPoint> &mp = mapPoints[i];
            if (mp) {
                if (mp->isBad()) {
                    current_kf->eraseMapPoint(i);
                } else {
                    mp->addObservation(current_kf, i);
                    mp->computeDescriptor();
                    mp->update();
                }
            }
        }

        // update links in the covisible graph
        current_kf->updateConnections();

        mapper_logger << titles[0] << "keyframe(id " << current_kf->id << "), frame id " << current_kf->frame_id
                      << "\n";
        mapper_logger << titles[0] << " - velo: " << current_kf->getVelocity() << "\n";

        point_map->addKeyFrame(current_kf);
    }

    void LocalMapping::MapPointCulling() {
        mapper_logger << "MapPointCulling\n";
        auto iter = recent_map_points.begin();
        const unsigned int curKFId = current_kf->id;

        int numFoundRatio = 0, numBad = 0;

        while (iter != recent_map_points.end()) {
            const shared_ptr<MapPoint> &mp = *iter;
            if (mp->isBad()) {
                iter = recent_map_points.erase(iter);
                numBad++;
            } else if (mp->getFoundRatio() < 0.25f) {
                mp->setBad();
                iter = recent_map_points.erase(iter);
                numFoundRatio++;
            } else if (curKFId - mp->first_kf_id >= 2 && mp->getNumObs() <= 2) {
                mp->setBad();
                iter = recent_map_points.erase(iter);
            } else if (curKFId - mp->first_kf_id > 2) {
                iter = recent_map_points.erase(iter);
            } else
                iter++;
        }

        mapper_logger << titles[0] << "delete " << numBad << " bad map-points, " << numFoundRatio
                      << " found not enough\n";
    }

    void LocalMapping::createNewMapPoints() {
        mapper_logger << "createNewMapPoints: triangulate with 20 recent keyframes\n";
        vector<shared_ptr<KeyFrame>> recentKFs = point_map->getRecentKeyFrames(21);
        recentKFs.pop_back();

        ORBMatcher matcher(0.6, false);
        const Camera *camera_ptr = Camera::getCamera();
        const Pose T2w = current_kf->getPose();
        const Eigen::Vector3f O2 = current_kf->getCameraCenter();
        const vector<float> squareSigmas = ORBExtractor::getSquareSigmas();
        const float ratioFactor = 1.5f * ORBExtractor::getScaleFactor(1);

        for (auto &kf: recentKFs) {
            const Pose T1w = kf->getPose();
            const Eigen::Vector3f O1 = kf->getCameraCenter();

            // check first that baseline is not too short
            const float baseline = (O2 - O1).norm();
            const float medianDepthKF1 = kf->computeSceneMedianDepth();
            if (baseline / medianDepthKF1 < 0.01) continue;

            vector<int> matches12;
            int numMatch = matcher.SearchForTriangulation(kf, current_kf, matches12);
            mapper_logger << titles[0] << "search " << numMatch << " in keyframe (id " << kf->id << "), ";

            // projection matrix
            Eigen::Matrix<float, 3, 4> P1, P2;
            P1.block<3, 3>(0, 0) = T1w.R, P1.block<3, 1>(0, 3) = T1w.t;
            P2.block<3, 3>(0, 0) = T2w.R, P2.block<3, 1>(0, 3) = T2w.t;

            int numGood = 0;

            int triangulateFail = 0, illegalPoint = 0, smallParallax = 0, negativePoint = 0, errorPoint = 0, scale_inconsistent = 0;
            for (unsigned int i = 0, end = matches12.size(); i < end; ++i) {
                if (matches12[i] == -1) continue;
                const cv::KeyPoint &kp1 = kf->key_points[i];
                const cv::KeyPoint &kp2 = current_kf->key_points[matches12[i]];
                Eigen::Vector3f x1 = camera_ptr->backProject(kp1.pt);
                Eigen::Vector3f x2 = camera_ptr->backProject(kp2.pt);
                Eigen::Vector3f Pw;
                if (TwoViewReconstruction::Triangulate(x1, x2, P1, P2, Pw)) {
                    if (!isfinite(Pw[0]) || !isfinite(Pw[1]) || !isfinite(Pw[2])) {
                        illegalPoint++;
                        continue;
                    }

                    // check parallax
                    Eigen::Vector3f n1 = Pw - O1;
                    const float dist1 = n1.norm();
                    n1 = n1 / dist1;
                    Eigen::Vector3f n2 = Pw - O2;
                    const float dist2 = n2.norm();
                    n2 = n2 / dist2;

                    const float cosParallax = n1.dot(n2);
                    if (cosParallax > 0.99998) {
                        smallParallax++;
                        continue;
                    }

                    // check re-projection error in first keyframe
                    Eigen::Vector3f Pc1 = T1w.R * Pw + T1w.t;
                    if (Pc1[2] <= 0) {
                        negativePoint++;
                        continue;
                    }
                    cv::Point2f project_p1 = camera_ptr->project(Pc1);
                    const float squareError1 = (project_p1.x - kp1.pt.x) * (project_p1.x - kp1.pt.x) +
                                               (project_p1.y - kp1.pt.y) * (project_p1.y - kp1.pt.y);
                    const float levelSquareSigma1 = squareSigmas[kp1.octave];
                    if (squareError1 > levelSquareSigma1 * 5.991) {
                        errorPoint++;
                        continue;
                    }

                    // check re-projection error in second keyframe
                    Eigen::Vector3f Pc2 = T2w.R * Pw + T2w.t;
                    if (Pc2[2] <= 0) {
                        negativePoint++;
                        continue;
                    }
                    cv::Point2f project_p2 = camera_ptr->project(Pc2);
                    const float squareError2 = (project_p2.x - kp2.pt.x) * (project_p2.x - kp2.pt.x) +
                                               (project_p2.y - kp2.pt.y) * (project_p2.y - kp2.pt.y);
                    const float levelSquareSigma2 = squareSigmas[kp2.octave];
                    if (squareError2 > levelSquareSigma2 * 5.991) {
                        errorPoint++;
                        continue;
                    }

                    const float distRatio = dist1 / dist2;
                    const float levelRatio = sqrtf(levelSquareSigma2) / sqrtf(levelSquareSigma1);
                    if (distRatio * ratioFactor < levelRatio || distRatio > levelRatio * ratioFactor) {
                        scale_inconsistent++;
                        continue;
                    }

                    shared_ptr<MapPoint> mp = make_shared<MapPoint>(Pw, kf, current_kf, Match(i, matches12[i]),
                                                                    point_map);
                    kf->addMapPoint(mp, i);
                    current_kf->addMapPoint(mp, matches12[i]);
                    point_map->addMapPoint(mp);
                    recent_map_points.push_back(mp);

                    numGood++;
                } else
                    triangulateFail++;
            }

            mapper_logger << "triangulate " << numGood << ", " << smallParallax << " parallax small, " << negativePoint
                          << " negative, " << illegalPoint << " illegal, " << errorPoint << " error, "
                          << scale_inconsistent << " scale inconsistent, " << triangulateFail << " triangulate fail\n";
        }
    }

    void LocalMapping::searchInNeighbors() {
        mapper_logger << "searchInNeighbors: project new map-points to neighbored keyframes\n";
        vector<shared_ptr<KeyFrame>> neighKFs = current_kf->getBestCovisibleKFs(20);
        vector<shared_ptr<KeyFrame>> targetKFs;
        for (const auto &kf: neighKFs) {
            if (kf->fuse_target_for_kf == current_kf->id) continue;
            targetKFs.push_back(kf);
            kf->fuse_target_for_kf = current_kf->id;

            // extend to some second neighbors
            const vector<shared_ptr<KeyFrame>> neighKFs2 = kf->getBestCovisibleKFs();
            for (const auto &kf2: neighKFs2) {
                if (kf2->fuse_target_for_kf == current_kf->id || kf2->id == current_kf->id) continue;
                kf2->fuse_target_for_kf = current_kf->id;
                targetKFs.push_back(kf2);
            }
        }

        // search matches by projection from current_kf to other local keyframes
        vector<shared_ptr<MapPoint>> curMapPoints = current_kf->getMapPoints();
        for (const auto &kf: targetKFs) {
            int numMatch = ORBMatcher::SearchByProjection(kf, curMapPoints, point_map);
            mapper_logger << titles[1] << "keyframe (id " << kf->id << ") match " << numMatch << " map points\n";
        }

        // search matches by projection from other local keyframe to current_kf
        vector<shared_ptr<MapPoint>> fuseMapPoints;
        fuseMapPoints.reserve(targetKFs.size() * curMapPoints.size());
        for (const auto &kf: targetKFs) {
            vector<shared_ptr<MapPoint>> mapPoints = kf->getMapPoints();
            for (const auto &mp: mapPoints) {
                if (mp && mp->isBad()) {
                    cerr << "kf has bad map-point" << endl;
                    continue;
                }
                if (!mp || mp->fuse_candidate_for_kf == current_kf->id) continue;
                mp->fuse_candidate_for_kf = current_kf->id;
                fuseMapPoints.push_back(mp);
            }
        }
        ORBMatcher::SearchByProjection(current_kf, fuseMapPoints, point_map);

        // update map-points and connection
        curMapPoints = current_kf->getMapPoints();
        for (const auto &mp: curMapPoints) {
            if (mp) {
                if (mp->isBad()) {
                    cerr << "kf has bad map-point" << endl;
                    continue;
                }
                mp->computeDescriptor();
                mp->update();
            }
        }
        current_kf->updateConnections();
    }

    void LocalMapping::KeyFrameCulling() {
        mapper_logger << "KeyFrameCulling\n";
        mapper_logger.flush();

        // check redundant keyframes (only local keyframes)
        // a keyframe is considered redundant if it's 90% map-points are seen in at least other 3 keyframes (in the same or finer scale)
        vector<shared_ptr<KeyFrame>> recentKeyFrames = point_map->getRecentKeyFrames(25);
        size_t numKF = recentKeyFrames.size();

        size_t last_kf_idx = 0;
        for (size_t idx = 1; idx < numKF - 1; ++idx) {
            if (recentKeyFrames[idx]->id == 0 ||
                recentKeyFrames[idx + 1]->timestamp - recentKeyFrames[last_kf_idx]->timestamp > 1.5)
                continue;

            auto &kf = recentKeyFrames[idx];
            const vector<shared_ptr<MapPoint>> mapPoints = kf->getMapPoints();
            const int thObs = 3;
            int numRedundantObs = 0;
            int numMP = 0;

            for (unsigned int i = 0; i < kf->num_kps; ++i) {
                const shared_ptr<MapPoint> &mp = mapPoints[i];
                if (mp != nullptr && !mp->isBad()) {
                    numMP++;

                    if (mp->getNumObs() > thObs) {
                        const int scaleLevel = kf->key_points[i].octave;
                        const map<shared_ptr<KeyFrame>, size_t> observations = mp->getObservations();
                        int numObs = 0;
                        for (const auto &obsPair: observations) {
                            const shared_ptr<KeyFrame> &kf2 = obsPair.first;
                            if (kf2->id == kf->id) continue;
                            const int scaleLevel2 = kf2->key_points[obsPair.second].octave;

                            if (scaleLevel2 <= scaleLevel + 1) {
                                numObs++;
                                if (numObs >= thObs) break;
                            }
                        }

                        if (numObs >= thObs) numRedundantObs++;
                    }
                }
            }

            if (numRedundantObs > 0.9 * numMP) {
                kf->setBad();
                mapper_logger << titles[0] << "keyframe (id " << kf->id << ") has set bad\n";
                mapper_logger.flush();
            } else {
                last_kf_idx = idx;
            }
        }
    }

    void LocalMapping::initializeIMU(float prioriG, float prioriA, bool beFirst) {
        mapper_logger << "InitializeIMU\n";
        mapper_logger << titles[0] << "prioriG: " << prioriG << ", prioriA: " << prioriA << ", beFirst: " << beFirst
                      << "\n";

        vector<shared_ptr<KeyFrame>> keyFrames = point_map->getAllKeyFrames();

        imu_initializing = true;

        while (getNewKeyFrame()) {
            processNewKeyFrame();
            keyFrames.push_back(current_kf);
        }

        int numKF = (int) keyFrames.size();

        Eigen::Matrix3d Rwg;
        if (imu_state == NOT_INITIALIZE) {
            Eigen::Vector3f gravityDir = Eigen::Vector3f::Zero();

            for (int i = 0; i < numKF - 1; ++i) {
                gravityDir -= keyFrames[i]->getImuPose().R * keyFrames[i]->pre_integrator->getUpdatedDeltaVelocity();
                Eigen::Vector3f velo = (keyFrames[i + 1]->getCameraCenter() - keyFrames[i]->getCameraCenter()) /
                                       keyFrames[i]->pre_integrator->delta_t;
                keyFrames[i]->setVelocity(velo);
                keyFrames[i + 1]->setVelocity(velo);
            }

            gravityDir.normalize();
            Eigen::Vector3f gI(0, 0, -1);
            Eigen::Vector3f v = gI.cross(gravityDir);
            const float nv = v.norm();
            const float theta = asin(nv);
            Rwg = lie::ExpSO3f(theta * v / nv).cast<double>();
        } else {
            Rwg.setIdentity();
        }

        double scale = 1.0;
        Eigen::Matrix3f Rwg_f = Rwg.cast<float>();
        mapper_logger << titles[0] << " - priori Rwg: " << Rwg_f << "\n";
        mapper_logger << titles[0] << " - priori scale: " << scale << "\n";
        mapper_logger.flush();

        mapper_logger << titles[0] << "BEFORE INERTIAL OPTIMIZE\n";
        for (const auto &kf: keyFrames) {
            mapper_logger << titles[1] << "keyframe id " << kf->id << ", frame_id " << kf->frame_id << "\n";
            mapper_logger << titles[1] << " - pose: " << kf->getPose() << "\n";
            mapper_logger << titles[1] << " - velo: " << kf->getVelocity() << "\n";
            mapper_logger << titles[1] << " - bias: " << kf->pre_integrator->updated_bias << "\n";
        }
        mapper_logger.flush();

        Optimize::inertialOptimize(point_map, Rwg, scale, prioriG, prioriA, beFirst);

        Rwg_f = Rwg.cast<float>();
        mapper_logger << titles[0] << "after optimize: \n";
        mapper_logger << titles[0] << " - posteriori Rwg: " << Rwg_f << "\n";
        mapper_logger << titles[0] << " - posteriori scale: " << scale << "\n";
        mapper_logger.flush();

        if (scale < 1e-1) {
            mapper_logger << titles[0] << "scale too small\n";
            imu_initializing = false;
            return;
        }

        {
            // changing the map
            lock_guard<mutex> lock(point_map->map_update_mutex);
            point_map->applyScaleRotation(Rwg.cast<float>(), scale, beFirst);
            tracker->updateFrameIMU();
        }

        if (imu_state == NOT_INITIALIZE) {
            imu_state = INITIALIZED;
        }

        mapper_logger << titles[0] << "after changing map\n";
        for (const auto &kf: keyFrames) {
            mapper_logger << titles[1] << "keyframe id " << kf->id << ", frame_id " << kf->frame_id << "\n";
            mapper_logger << titles[1] << " - pose: " << kf->getPose() << "\n";
            mapper_logger << titles[1] << " - velo: " << kf->getVelocity() << "\n";
            mapper_logger << titles[1] << " - bias: " << kf->pre_integrator->updated_bias << "\n";
        }
        mapper_logger.flush();

        // full inertial BA
        mapper_logger << titles[0] << "fullInertialOptimize\n";
        Optimize::fullInertialOptimize(point_map, 100, beFirst, false, prioriG, prioriA);

        // process keyframes in the queue
        while (getNewKeyFrame()) {
            processNewKeyFrame();
            keyFrames.push_back(current_kf);
        }

        for (const auto &kf: keyFrames) {
            mapper_logger << titles[1] << "keyframe id " << kf->id << ", frame_id " << kf->frame_id << "\n";
            mapper_logger << titles[1] << " - pose: " << kf->getPose() << "\n";
            mapper_logger << titles[1] << " - velo: " << kf->getVelocity() << "\n";
            mapper_logger << titles[1] << " - bias: " << kf->pre_integrator->updated_bias << "\n";
        }
        mapper_logger.flush();

        imu_initializing = false;
        point_map->increaseChangeIdx();
        last_inertial_time = current_kf->timestamp;
    }

    void LocalMapping::gravityRefinement() {
        mapper_logger << "gravityRefinement\n";

        Eigen::Matrix3d Rwg = Eigen::Matrix3d::Identity();
        Eigen::Matrix3f Rwg_f = Rwg.cast<float>();
        mapper_logger << titles[0] << " - priori Rwg: " << Rwg_f << "\n";

        Optimize::gravityOptimize(point_map, Rwg);

        {
            // changing the map
            lock_guard<mutex> lock(point_map->map_update_mutex);
            point_map->applyScaleRotation(Rwg.cast<float>(), 1, false);
            tracker->updateFrameIMU();
        }

        Rwg_f = Rwg.cast<float>();
        mapper_logger << titles[0] << " - posteriori Rwg: " << Rwg_f << "\n";

        imu_state = FINISH;
    }

    void LocalMapping::setTracker(Tracking *trackerPtr) {
        tracker = trackerPtr;
    }

    bool LocalMapping::acceptKeyFrames() {
        lock_guard<mutex> lock(new_kf_mutex);
        return accept_new_kf;
    }

    void LocalMapping::setAcceptKeyFrame(bool flag) {
        lock_guard<mutex> lock(new_kf_mutex);
        accept_new_kf = flag;
    }

    void LocalMapping::requestStop() {
        lock_guard<mutex> lock1(stop_mutex);
        stop_requested = true;
        lock_guard<mutex> lock2(new_kf_mutex);
        abort_BA = true;
    }

    void LocalMapping::requestReset() {
        {
            lock_guard<mutex> lock(reset_mutex);
            reset_requested = true;
        }

        while (true) {
            {
                lock_guard<mutex> lock(reset_mutex);
                if (!reset_requested)
                    break;
            }
            usleep(3000);
        }
        imu_state = NOT_INITIALIZE;
    }

    bool LocalMapping::stop() {
        lock_guard<mutex> lock(stop_mutex);
        if (stop_requested && !not_stop) {
            stopped = true;
            cout << "Local Mapping STOP" << endl;
            return true;
        }
        return false;
    }

    void LocalMapping::release() {
        lock_guard<mutex> lock1(stop_mutex);
        lock_guard<mutex> lock2(finish_mutex);

        if (finished) return;
        stopped = false;
        stop_requested = false;
        new_keyframes.clear();

        cout << "Local Mapping RELEASE" << endl;
    }

    bool LocalMapping::isStopped() {
        lock_guard<mutex> lock(stop_mutex);
        return stopped;
    }

    bool LocalMapping::stopRequested() {
        lock_guard<mutex> lock(stop_mutex);
        return stop_requested;
    }

    bool LocalMapping::setNotStop(bool flag) {
        lock_guard<mutex> lock(stop_mutex);

        if (flag && stopped) return false;

        not_stop = flag;
        return true;
    }

    void LocalMapping::interruptBA() {
        abort_BA = true;
    }

    void LocalMapping::addNewKeyFrame(const shared_ptr <KeyFrame> &keyFrame) {
        lock_guard<mutex> lock(new_kf_mutex);
        new_keyframes.push_back(keyFrame);
        abort_BA = true;
    }

    bool LocalMapping::checkNewKeyFrame() {
        lock_guard<mutex> lock(new_kf_mutex);
        return !new_keyframes.empty();
    }

    bool LocalMapping::getNewKeyFrame() {
        lock_guard<mutex> lock(new_kf_mutex);
        if (new_keyframes.empty()) return false;
        current_kf = new_keyframes.front();
        new_keyframes.pop_front();
        return true;
    }

    void LocalMapping::requestFinish() {
        lock_guard<mutex> lock(finish_mutex);
        request_finish = true;
    }

    bool LocalMapping::isFinish() {
        lock_guard<mutex> lock(finish_mutex);
        return finished;
    }

    bool LocalMapping::checkFinish() {
        lock_guard<mutex> lock(finish_mutex);
        return request_finish;
    }

    void LocalMapping::setFinish() {
        lock_guard<mutex> lock1(finish_mutex);
        finished = true;
        lock_guard<mutex> lock2(stop_mutex);
        stopped = true;
    }

    int LocalMapping::numKeyFramesInQueue() {
        lock_guard<mutex> lock(new_kf_mutex);
        return (int) new_keyframes.size();
    }

    bool LocalMapping::isImuInitialized() {
        return imu_state != NOT_INITIALIZE;
    }

    bool LocalMapping::finishImuInit() {
        return imu_state == FINISH;
    }

    bool LocalMapping::isInitializing() const {
        return imu_initializing;
    }

    void LocalMapping::resetIfRequested() {
        lock_guard<mutex> lock(reset_mutex);
        if (reset_requested) {
            new_keyframes.clear();
            recent_map_points.clear();
            reset_requested = false;
        }
    }

} // mono_orb_slam3