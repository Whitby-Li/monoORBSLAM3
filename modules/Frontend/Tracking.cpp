//
// Created by whitby on 8/22/23.
//

#include "Tracking.h"
#include "Sensor/Camera.h"
#include "Utils/LieAlgeBra.h"
#include "ORB/ORBMatcher.h"
#include "Backend/Optimize.h"

#include <iostream>

using namespace std;

namespace mono_orb_slam3 {

    Tracking::Tracking(const cv::FileNode &orbNode, Map *pointMap, System *system_ptr)
            : point_map(pointMap), system(system_ptr), camera_ptr(Camera::getCamera()) {
        // create ORB extractor
        int nFeatures = orbNode["Features"];
        extractor = new ORBExtractor(nFeatures, orbNode["ScaleFactor"], orbNode["Levels"], orbNode["IniThFAST"],
                                     orbNode["MinThFAST"]);
        initial_extractor = new ORBExtractor(2 * nFeatures, *extractor);
        extractor->print();
    }

    void Tracking::setLocalMapper(LocalMapping *localMapper) {
        local_mapper = localMapper;
    }

    void Tracking::setViewer(Viewer *viewerPtr) {
        viewer = viewerPtr;
        frame_drawer = viewer->frame_drawer;
        map_drawer = viewer->map_drawer;
        have_viewer = true;
    }

    void Tracking::reset() {
        cout << "System Resetting" << endl;
        if (have_viewer) {
            viewer->requestStop();
            while (!viewer->isStopped())
                usleep(3000);
        }

        // reset local mapping
        cout << "Resetting Local Mapper... ";
        local_mapper->requestReset();
        cout << "done" << endl;

        // clear map
        point_map->clear();

        KeyFrame::next_id = 0;
        Frame::next_id = 0;
        state = NO_IMAGE_YET;
        last_frame = nullptr;
        current_frame = nullptr;

        Tcl = Pose();
        Tlr = Pose();
        have_velocity = false;

        if (have_viewer)
            viewer->release();
    }

    void Tracking::Track(double timeStamp, cv::Mat grayImg, const vector <ImuData> &imus) {
        last_state = state;
        if (state == NO_IMAGE_YET) state = NOT_INITIALIZE;

        // get map update mutex -> point_map cannot be changed
        lock_guard<mutex> lock(point_map->map_update_mutex);

        if (state == NOT_INITIALIZE) {
            if (last_frame) last_frame->computePreIntegration(imus, timeStamp);

            current_frame = make_shared<Frame>(grayImg, timeStamp, initial_extractor,
                                               ImuCalib::getImuCalib()->initial_bias);
            cout << "extractor " << current_frame->num_kps << " features" << endl;
            Initialization();

            if (have_viewer && last_frame) {
                frame_drawer->Update(this);
            }

        } else if (state != LOST) {
            if (last_frame) last_frame->computePreIntegration(imus, timeStamp);
            if (last_kf) last_kf->computePreIntegration(imus, timeStamp);

            current_frame = make_shared<Frame>(grayImg, timeStamp, extractor, last_frame->pre_integrator->updated_bias);
            cout << "extractor " << current_frame->num_kps << " features" << endl;

            // 1. check replaced map-points in last frame and update last_frame pose according to last_keyframe
            updateLastFramePose();

            // 2. estimate initial pose
            bool isOK = false;

            if (local_mapper->isImuInitialized()) {
                if (state == RECENTLY_LOST) predictCurFramePoseByKF();
                else predictCurFramePose();
            } else if (have_velocity) {
                // estimate initial pose with motion model
                current_frame->setPose(Tcl * last_frame->T_cw);
            }

            // 3. track
            if (local_mapper->isImuInitialized()) {
                if (num_inlier < 100 || state == RECENTLY_LOST)
                    isOK = trackLastKeyFrame();
                else {
                    isOK = trackLastFrame();
                    if (!isOK) {
                        fill(current_frame->map_points.begin(), current_frame->map_points.end(), nullptr);
                        predictCurFramePose();
                        isOK = trackLastKeyFrame();
                    }
                }
                if (!isOK) {
                    fill(current_frame->map_points.begin(), current_frame->map_points.end(), nullptr);
                    predictCurFramePoseByKF();
                }

                isOK = trackLocalMap();
            } else {
                if (have_velocity) isOK = trackLastFrame();
                if (!isOK) isOK = trackReferenceKeyFrame();

                if (isOK) isOK = trackLocalMap();
            }

            if (have_viewer) {
                frame_drawer->Update(this);
                map_drawer->SetCurrentCameraPose(current_frame->T_cw.inverse());
                map_drawer->SetReferenceKeyFrame(reference_kf);
            }

            if (isOK) {
                // update motion model
                Tcl = current_frame->T_cw * last_frame->T_cw.inverse();
                have_velocity = true;
                state = OK;
                last_lost_time = timeStamp;
            } else {
                cerr << "RECENTLY LOST" << endl;
                state = RECENTLY_LOST;
                predictCurFramePoseByKF();
                if (!local_mapper->finishImuInit() || current_frame->timestamp - last_lost_time > 3) {
                    state = LOST;
                }
            }

            // 4. check new keyframe
            if (needNewKeyFrame())
                createNewKeyFrame();

            last_frame = current_frame;
            const Pose Twr = reference_kf->getInversePose();
            Tlr = last_frame->T_cw * Twr;
        } else {
            cerr << "Track Lost, RESET" << endl;
            system->Reset();
        }
    }

    void Tracking::updateFrameIMU() {
        last_frame->pre_integrator->setNewBias(last_kf->pre_integrator->updated_bias);

        if (last_frame->id == last_kf->frame_id) {
            last_frame->setImuPoseAndVelocity(last_kf->getImuPose(), last_kf->getVelocity());
        } else {
            const Eigen::Vector3f g(0, 0, -GRAVITY_VALUE);
            const Pose Twb1 = last_kf->getImuPose();
            const Eigen::Vector3f v1 = last_kf->getVelocity();
            shared_ptr<PreIntegrator> preIntegrator = last_kf->pre_integrator;
            const float dt = preIntegrator->delta_t;

            Eigen::Matrix3f Rwb2 = lie::NormalizeRotationf(Twb1.R * preIntegrator->getUpdatedDeltaRotation());
            Eigen::Vector3f twb2 =
                    Twb1.t + v1 * dt + 0.5 * g * dt * dt + Twb1.R * preIntegrator->getUpdatedDeltaPosition();
            Eigen::Vector3f v2 = v1 + g * dt + Twb1.R * preIntegrator->getUpdatedDeltaVelocity();
            last_frame->setImuPoseAndVelocity({Rwb2, twb2}, v2);
        }

        const Pose Twr = reference_kf->getInversePose();
        Tlr = last_frame->T_cw * Twr;
        have_velocity = false;
    }

    void Tracking::predictCurFramePose() {
        shared_ptr<PreIntegrator> preIntegrator = last_frame->pre_integrator;
        const float dt = preIntegrator->delta_t;
        const Eigen::Vector3f g(0, 0, -GRAVITY_VALUE);

        Eigen::Matrix3f Rwb2 = lie::NormalizeRotationf(last_frame->T_wb.R * preIntegrator->getUpdatedDeltaRotation());
        Eigen::Vector3f twb2 = last_frame->T_wb.t + last_frame->v_w * dt + 0.5 * g * dt * dt +
                               last_frame->T_wb.R * preIntegrator->getUpdatedDeltaPosition();
        Eigen::Vector3f v2 = last_frame->v_w + g * dt + last_frame->T_wb.R * preIntegrator->getUpdatedDeltaVelocity();

        current_frame->setImuPoseAndVelocity({Rwb2, twb2}, v2);
    }

    void Tracking::predictCurFramePoseByKF() {
        shared_ptr<PreIntegrator> preIntegrator = last_kf->pre_integrator;
        const float dt = preIntegrator->delta_t;
        const Eigen::Vector3f g(0, 0, -GRAVITY_VALUE);

        Pose Twb1 = last_kf->getImuPose();
        Eigen::Vector3f v1 = last_kf->getVelocity();
        Eigen::Matrix3f Rwb2 = lie::NormalizeRotationf(Twb1.R * preIntegrator->getUpdatedDeltaRotation());
        Eigen::Vector3f twb2 = Twb1.t + v1 * dt + 0.5 * g * dt * dt + Twb1.R * preIntegrator->getUpdatedDeltaPosition();
        Eigen::Vector3f v2 = v1 + g * dt + Twb1.R * preIntegrator->getUpdatedDeltaVelocity();

        current_frame->setImuPoseAndVelocity({Rwb2, twb2}, v2);
    }

    void Tracking::updateLastFramePose() {
        last_frame->setPose(Tlr * reference_kf->getPose());
    }

    bool Tracking::trackReferenceKeyFrame() {
        // compute bag of worlds vector
        current_frame->computeBow();

        // we perform first an ORB matching with the reference keyframe
        ORBMatcher matcher(0.7, true);
        int numMatch = matcher.SearchByBow(reference_kf, current_frame);

        if (numMatch < 15) {
            cerr << "trackReferenceKeyFrame: not enough matches" << endl;
            return false;
        }

        current_frame->setPose(last_frame->T_cw);

        // optimize frame pose with all matches
        int numInlier = Optimize::poseOptimize(current_frame);

        if (numInlier < 10) {
            cerr << "trackReferenceKeyFrame: not enough inlier matches" << endl;
            return false;
        }

        return true;
    }

    bool Tracking::trackLastFrame() {
        float th = 15;
        ORBMatcher matcher(0.9, true);
        int numMatch = matcher.SearchByProjection(last_frame, current_frame, th);

        if (numMatch < 30) {
            fill(current_frame->map_points.begin(), current_frame->map_points.end(), nullptr);
            numMatch = matcher.SearchByProjection(last_frame, current_frame, 2 * th);
        }

        if (numMatch < 30) {
            cerr << "trackLastFrame: not enough match" << endl;
            return false;
        }

        if (local_mapper->finishImuInit() && numMatch < 80) return true;

        // optimize frame pose with all matches
        num_inlier = Optimize::poseOptimize(current_frame);

        if (num_inlier < 15) {
            cerr << "trackLastFrame: not enough inlier matches" << endl;
            return false;
        }

        return true;
    }

    bool Tracking::trackLastKeyFrame() {
        float th = 15;
        ORBMatcher matcher(0.9, true);
        int numMatch = matcher.SearchByProjection(last_kf, current_frame, th);

        if (numMatch < 30) {
            fill(current_frame->map_points.begin(), current_frame->map_points.end(), nullptr);
            numMatch = matcher.SearchByProjection(last_kf, current_frame, 2 * th);
        }

        if (numMatch < 30) {
            cerr << "trackLastKeyFrame: not enough match" << endl;
            return false;
        }

        if (local_mapper->finishImuInit() && numMatch < 80) return true;

        num_inlier = Optimize::poseOptimize(current_frame);
        if (num_inlier < 15) {
            cerr << "trackLastKeyFrame: not enough inlier matches" << endl;
            return false;
        }

        return true;
    }

    bool Tracking::trackLocalMap() {
        updateLocalMap();

        // search local map-points
        searchLocalPoints();

        if (local_mapper->finishImuInit())
            num_inlier = Optimize::poseFullOptimize(last_kf, current_frame);
        else {
            num_inlier = Optimize::poseOptimize(current_frame);
            if (local_mapper->isImuInitialized())
                Optimize::poseInertialOptimize(last_kf, current_frame);
        }

        for (const auto &mp: current_frame->map_points) {
            if (mp != nullptr) mp->increaseFound();
        }

        if (num_inlier < 20) {
            cerr << "trackLocalMap: not enough inlier matches" << endl;
            return false;
        }

        cout << "track " << num_inlier << " map-points" << endl;
        return true;
    }

    void Tracking::updateLocalMap() {
        // this is for visualization
        if (have_viewer) point_map->setReferenceMapPoints(local_map_points);

        // update
        updateLocalKeyFrames();
        updateLocalMapPoints();
    }

    void Tracking::searchLocalPoints() {
        // do not search map-points already matched
        for (auto &mp: current_frame->map_points) {
            if (mp) {
                if (mp->isBad()) {
                    mp = nullptr;
                } else {
                    mp->increaseVisible();
                    mp->last_frame_seen = current_frame->id;
                    mp->track_in_view = false;
                }
            }
        }

        int numToMatch = 0, outView = 0;

        // project points in frame and check its visibility
        for (const auto &mp: local_map_points) {
            if (mp->last_frame_seen == current_frame->id || mp->isBad()) continue;

            if (current_frame->isInFrustum(mp, 0.5)) {
                mp->increaseVisible();
                numToMatch++;
            } else {
                outView++;
            }
        }

        if (numToMatch > 0) {
            ORBMatcher matcher(0.8);
            float th = 1;
            if (state == OK) {
                if (local_mapper->finishImuInit()) th = 2;
            } else {
                th = 10;
            }

            matcher.SearchByProjection(current_frame, local_map_points, th);
        }
    }

    void Tracking::updateLocalKeyFrames() {
        // each map-point vote for the keyframes in which it has been observed
        unordered_map<shared_ptr<KeyFrame>, int> kfCounter;
        if (!local_mapper->isImuInitialized()) {
            for (int i = 0; i < current_frame->num_kps; ++i) {
                shared_ptr<MapPoint> &mp = current_frame->map_points[i];
                if (mp) {
                    if (!mp->isBad()) {
                        const map<shared_ptr<KeyFrame>, size_t> observations = mp->getObservations();
                        for (auto &obsPair: observations) {
                            kfCounter[obsPair.first]++;
                        }
                    } else {
                        current_frame->map_points[i] = nullptr;
                    }
                }
            }
        } else {
            for (int i = 0; i < last_frame->num_kps; ++i) {
                shared_ptr<MapPoint> &mp = last_frame->map_points[i];
                if (mp) {
                    if (!mp->isBad()) {
                        const map<shared_ptr<KeyFrame>, size_t> observations = mp->getObservations();
                        for (auto &obsPair: observations) {
                            kfCounter[obsPair.first]++;
                        }
                    } else {
                        last_frame->map_points[i] = nullptr;
                    }
                }
            }
        }

        int maxObs = 0;
        shared_ptr<KeyFrame> maxKF = nullptr;

        // add 10 current keyframes
        local_keyframes = point_map->getRecentKeyFrames(10);
        local_keyframes.reserve(200);
        for (const auto &kf: local_keyframes) {
            kf->track_frame_id = current_frame->id;
        }

        // all keyframes that observe a map-point are included in the local map.
        // also check which keyframe share the most points.
        for (const auto &countPair: kfCounter) {
            auto &kf = countPair.first;
            if (kf->isBad()) continue;

            if (countPair.second > maxObs) {
                maxObs = countPair.second;
                maxKF = kf;
            }

            if (kf->track_frame_id != current_frame->id) {
                local_keyframes.push_back(kf);
                kf->track_frame_id = current_frame->id;
            }
        }

        // include also some not-already-included keyframes that are neighbor to already-included keyframes
        for (auto iter = local_keyframes.begin(), iterEnd = local_keyframes.end(); iter < iterEnd; iter++) {
            // limit the number of keyframes
            if (local_keyframes.size() > 80) break;

            shared_ptr<KeyFrame> kf = *iter;
            const vector<shared_ptr<KeyFrame>> neighKFs = kf->getBestCovisibleKFs(10);

            for (const auto &neigh_kf: neighKFs) {
                if (!neigh_kf->isBad() && neigh_kf->track_frame_id != current_frame->id) {
                    local_keyframes.push_back(neigh_kf);
                    neigh_kf->track_frame_id = current_frame->id;
                }
            }

            const set<shared_ptr<KeyFrame>> children = kf->getChild();
            for (const auto &child_kf: children) {
                if (!child_kf->isBad() && child_kf->track_frame_id != current_frame->id) {
                    local_keyframes.push_back(child_kf);
                    child_kf->track_frame_id = current_frame->id;
                    break;
                }
            }

            shared_ptr<KeyFrame> parent = kf->getParent();
            if (parent && parent->track_frame_id != current_frame->id) {
                local_keyframes.push_back(parent);
                parent->track_frame_id = current_frame->id;
                break;
            }
        }

        // update reference keyframe
        if (maxKF) reference_kf = maxKF;
    }

    void Tracking::updateLocalMapPoints() {
        local_map_points.clear();
        for (auto &kf: local_keyframes) {
            const vector<shared_ptr<MapPoint>> mapPoints = kf->getMapPoints();

            for (auto &mp: mapPoints) {
                if (mp && !mp->isBad() && mp->track_frame_id != current_frame->id) {
                    local_map_points.push_back(mp);
                    mp->track_frame_id = current_frame->id;
                }
            }
        }
    }

    bool Tracking::needNewKeyFrame() {
        if (local_mapper->isStopped() || local_mapper->stopRequested()) return false;

        bool beMapperIdle = local_mapper->acceptKeyFrames();
        int minObs = point_map->getNumKeyFrames() > 2 ? 3 : 2;
        int numRefMatch = reference_kf->getNumTrackedMapPoint(minObs);
        float theRefRatio = 0.9;
        if (num_inlier > 350) theRefRatio = 0.75;

        // condition
        bool c1a = current_frame->id >= last_kf->frame_id + 10;
        bool c1b = current_frame->id >= last_kf->frame_id + 1 && beMapperIdle;
        bool c2 = num_inlier < numRefMatch * theRefRatio;
        bool c3 = current_frame->timestamp - last_kf->timestamp >= 0.5;
        bool c4 = (num_inlier < 75 && num_inlier > 15) || state == RECENTLY_LOST;

        if (((c1a || c1b) && c2) || c3 || c4) {
            if (beMapperIdle || local_mapper->isInitializing()) {
                return true;
            } else {
                local_mapper->interruptBA();
                return false;
            }
        }

        return false;
    }

    void Tracking::createNewKeyFrame() {
        if (!local_mapper->setNotStop(true)) return;

        shared_ptr<KeyFrame> kf = make_shared<KeyFrame>(current_frame, point_map);
        kf->setPrioriInformation(last_kf->pre_integrator->C);
        local_mapper->addNewKeyFrame(kf);

        reference_kf = kf;
        last_kf = kf;
        local_mapper->setNotStop(false);
    }

    void Tracking::Initialization() {
        if (current_frame->num_kps > 500) {
            if (last_frame == nullptr || current_frame->id - last_frame->id > 20) {
                priori_matches.resize(current_frame->num_kps);
                for (int i = 0; i < current_frame->num_kps; ++i)
                    priori_matches[i] = current_frame->key_points[i].pt;

                last_frame = current_frame;
            } else {
                ORBMatcher matcher(0.9, true);
                int numMatches = matcher.SearchForInitialization(last_frame, current_frame, priori_matches,
                                                                 initial_matches, 100);

                if (numMatches < 200) {
                    last_frame = nullptr;
                    priori_matches.clear();
                    return;
                }

                Eigen::Matrix3f R21;
                Eigen::Vector3f t21;
                vector<bool> vbTriangulated;
                if (camera_ptr->reconstructWithTwoViews(last_frame->key_points, current_frame->key_points,
                                                       initial_matches, R21, t21, initial_3d_points, vbTriangulated)) {
                    for (size_t i = 0, end = initial_matches.size(); i < end; ++i) {
                        if (initial_matches[i] >= 0 && !vbTriangulated[i]) {
                            initial_matches[i] = -1;
                            numMatches--;
                        }
                    }

                    cout << "triangulate " << numMatches << " 3D points" << endl;

                    // set frames' pose
                    last_frame->setPose({Eigen::Matrix3f::Identity(), Eigen::Vector3f::Zero()});
                    current_frame->setPose({R21, t21});

                    CreateInitialMap();
                }
            }
        } else {
            last_frame = nullptr;
            priori_matches.clear();
        }
    }

    void Tracking::CreateInitialMap() {
        // create keyframes
        shared_ptr<KeyFrame> iniKF = make_shared<KeyFrame>(last_frame, point_map);
        iniKF->pre_integrator = last_frame->pre_integrator;
        shared_ptr<KeyFrame> curKF = make_shared<KeyFrame>(current_frame, point_map);
        curKF->setPrioriInformation(last_frame->pre_integrator->C);
        iniKF->computeBow();
        curKF->computeBow();

        for (size_t i = 0, end = initial_matches.size(); i < end; ++i) {
            if (initial_matches[i] < 0) continue;

            // create map-point
            cv::Point3f &Pw = initial_3d_points[i];
            shared_ptr<MapPoint> mp = make_shared<MapPoint>(Eigen::Vector3f(Pw.x, Pw.y, Pw.z), iniKF, curKF,
                                                            Match(i, initial_matches[i]), point_map);
            iniKF->addMapPoint(mp, i);
            curKF->addMapPoint(mp, initial_matches[i]);
            last_frame->map_points[i] = mp;
            current_frame->map_points[initial_matches[i]] = mp;

            point_map->addMapPoint(mp);
        }

        point_map->addKeyFrame(iniKF);
        point_map->addKeyFrame(curKF);

        // initial optimize
        Optimize::initialOptimize(iniKF, curKF);

        float medianDepth = curKF->computeSceneMedianDepth();
        float invMedianDepth = 1.f / medianDepth;

        Pose Tc2w = curKF->getPose();
        Tc2w.t = Tc2w.t * invMedianDepth;
        curKF->setPose(Tc2w.R, Tc2w.t);

        // update map-points
        vector<shared_ptr<MapPoint>> mapPoints = curKF->getTrackedMapPoints();
        for (auto &mp: mapPoints) {
            mp->setPos(mp->getPos() * invMedianDepth);
            mp->update();
        }

        curKF->updateConnections();

        last_frame->setPose(iniKF->getPose());
        current_frame->setPose(curKF->getPose());
        last_kf = curKF;
        reference_kf = curKF;
        last_frame = current_frame;
        local_mapper->addNewKeyFrame(curKF);

        if (have_viewer) map_drawer->SetCurrentCameraPose(curKF->getInversePose());

        state = OK;
    }

} // mono_orb_slam3