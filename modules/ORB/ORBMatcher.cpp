//
// Created by whitby on 8/22/23.
//

#include "ORBMatcher.h"
#include "Sensor/Camera.h"
#include "Utils/LieAlgeBra.h"
#include "Log/Logger.h"

using namespace std;

namespace mono_orb_slam3 {
    const int TH_LOW = 50;
    const int TH_HIGH = 100;
    const int HISTO_LENGTH = 30;

    int ORBMatcher::DescriptorDistance(const cv::Mat &a, const cv::Mat &b) {
        const int *pa = a.ptr<int32_t>();
        const int *pb = b.ptr<int32_t>();

        int dist = 0;

        for (int i = 0; i < 8; i++, pa++, pb++) {
            unsigned int v = *pa ^ *pb;
            v = v - ((v >> 1) & 0x55555555);
            v = (v & 0x33333333) + ((v >> 2) & 0x33333333);
            dist += (((v + (v >> 4)) & 0xF0F0F0F) * 0x1010101) >> 24;
        }

        return dist;
    }

    int ORBMatcher::SearchForInitialization(const std::shared_ptr<Frame> &frame1, const std::shared_ptr<Frame> &frame2,
                                            std::vector<cv::Point2f> &vecPreMatched, std::vector<int> &matches12,
                                            int windowSize) const {
        int numMatches = 0;
        matches12 = vector<int>(frame1->num_kps, -1);
        vector<int> matches21(frame2->num_kps, -1);
        vector<int> matchedDistance(frame2->num_kps, INT_MAX);

        vector<int> rotHist[HISTO_LENGTH];
        for (auto &rot: rotHist)
            rot.reserve(300);
        const float factor = 1.f / HISTO_LENGTH;

        for (int idx1 = 0; idx1 < frame1->num_kps; ++idx1) {
            const cv::KeyPoint &kp1 = frame1->key_points[idx1];
            int level1 = kp1.octave;
            if (level1 > 0) continue;

            vector<size_t> indices2 = frame2->getFeaturesInArea(vecPreMatched[idx1].x, vecPreMatched[idx1].y,
                                                                windowSize,
                                                                level1, level1);
            if (indices2.empty()) continue;

            cv::Mat desc1 = frame1->descriptors.row(idx1);
            int bestDist = INT_MAX - 1, bestDist2 = INT_MAX;
            int bestIdx2 = -1;
            for (auto &idx2: indices2) {
                cv::Mat desc2 = frame2->descriptors.row(idx2);
                int dist = DescriptorDistance(desc1, desc2);

                if (matchedDistance[idx2] <= dist) continue;

                if (dist < bestDist) {
                    bestDist2 = bestDist;
                    bestDist = dist;
                    bestIdx2 = idx2;
                } else if (dist < bestDist2) {
                    bestDist2 = dist;
                }
            }

            if (bestDist <= TH_LOW && bestDist < cvRound((float) bestDist2 * nn_ratio)) {
                if (matches21[bestIdx2] >= 0) {
                    matches12[matches21[bestIdx2]] = -1;
                    numMatches--;
                }
                matches12[idx1] = bestIdx2;
                matches21[bestIdx2] = idx1;
                matchedDistance[bestIdx2] = bestDist;
                numMatches++;

                if (be_check_orientation) {
                    float rotAngle = frame1->key_points[idx1].angle - frame2->key_points[bestIdx2].angle;
                    if (rotAngle < 0) rotAngle += 360;
                    int bin = cvRound(rotAngle * factor);
                    if (bin == HISTO_LENGTH) bin = 0;
                    assert(bin >= 0 && bin < HISTO_LENGTH && "check orientation error");
                    rotHist[bin].push_back(idx1);
                }
            }
        }

        if (be_check_orientation) {
            int ind1 = -1, ind2 = -1, ind3 = -1;
            ComputeThreeMaxima(rotHist, ind1, ind2, ind3);

            for (int i = 0; i < HISTO_LENGTH; ++i) {
                if (i == ind1 || i == ind2 || i == ind3) continue;
                for (auto idx1: rotHist[i]) {
                    if (matches12[idx1] >= 0) {
                        matches12[idx1] = -1;
                        numMatches--;
                    }
                }
            }
        }

        // update previous match
        for (int idx1 = 0; idx1 < frame1->num_kps; ++idx1)
            if (matches12[idx1] >= 0)
                vecPreMatched[idx1] = frame2->key_points[matches12[idx1]].pt;

        return numMatches;
    }

    int ORBMatcher::SearchByBow(const std::shared_ptr<KeyFrame> &keyFrame, const std::shared_ptr<Frame> &frame) const {
        vector<shared_ptr<MapPoint>> mapPoints = keyFrame->getMapPoints();
        const DBoW2::FeatureVector &featureVector1 = keyFrame->feature_vector;
        const DBoW2::FeatureVector &featureVector2 = frame->feature_vector;
        int numMatch = 0;

        vector<int> rotHist[HISTO_LENGTH];
        for (auto hist: rotHist) {
            hist.reserve(300);
        }
        const float factor = 1.f / HISTO_LENGTH;

        // we perform the matching over ORB that belong to the same vocabulary node
        auto iter1 = featureVector1.begin();
        auto iter2 = featureVector2.begin();
        auto iterEnd1 = featureVector1.end();
        auto iterEnd2 = featureVector2.end();

        while (iter1 != iterEnd1 && iter2 != iterEnd2) {
            if (iter1->first == iter2->first) {
                const vector<unsigned int> indices1 = iter1->second;
                const vector<unsigned int> indices2 = iter2->second;

                for (auto idx1: indices1) {
                    const shared_ptr<MapPoint> &mp = mapPoints[idx1];
                    if (mp == nullptr || mp->isBad()) continue;

                    const cv::KeyPoint &kp1 = keyFrame->key_points[idx1];
                    const cv::Mat desc1 = keyFrame->descriptors.row(idx1);

                    int bestDist = 256, secondDist = 256, bestIdx2 = -1;
                    for (auto idx2: indices2) {
                        if (frame->map_points[idx2] != nullptr) continue;

                        const cv::Mat desc2 = frame->descriptors.row(idx2);
                        const int dist = DescriptorDistance(desc1, desc2);

                        if (dist < bestDist) {
                            secondDist = bestDist;
                            bestDist = dist;
                            bestIdx2 = idx2;
                        } else if (dist < secondDist) {
                            secondDist = dist;
                        }
                    }

                    if (bestDist <= TH_LOW && bestDist < nn_ratio * secondDist) {
                        frame->map_points[bestIdx2] = mp;
                        numMatch++;

                        if (be_check_orientation) {
                            const cv::KeyPoint &kp2 = frame->key_points[bestIdx2];
                            float rotAngle = kp1.angle - kp2.angle;
                            if (rotAngle < 0) rotAngle += 360;
                            int bin = cvRound(rotAngle * factor);
                            if (bin == HISTO_LENGTH) bin = 0;
                            rotHist[bin].push_back(bestIdx2);
                        }
                    }
                }
                iter1++;
                iter2++;
            } else if (iter1->first < iter2->first) {
                iter1 = featureVector1.lower_bound(iter2->first);
            } else {
                iter2 = featureVector2.lower_bound(iter1->first);
            }
        }

        if (be_check_orientation) {
            int ind1 = -1, ind2 = -1, ind3 = -1;
            ComputeThreeMaxima(rotHist, ind1, ind2, ind3);

            for (int i = 0; i < HISTO_LENGTH; ++i) {
                if (i == ind1 || i == ind2 || i == ind3) continue;
                for (auto idx2: rotHist[i]) {
                    frame->map_points[idx2] = nullptr;
                    numMatch--;
                }
            }
        }

        return numMatch;
    }

    int ORBMatcher::SearchByProjection(const std::shared_ptr<Frame> &lastFrame, const std::shared_ptr<Frame> &curFrame,
                                       float th) const {
        const float factor = 1.f / HISTO_LENGTH;
        vector<int> rotHist[HISTO_LENGTH];

        const Camera *camera = Camera::getCamera();
        const Pose &Tcw = curFrame->T_cw;
        int numMatch = 0;

        for (int i = 0; i < lastFrame->num_kps; ++i) {
            const shared_ptr<MapPoint> &mp = lastFrame->map_points[i];
            if (mp != nullptr) {
                if (mp->isBad()) {
                    continue;
                }

                const Eigen::Vector3f Pw = mp->getPos();
                const Eigen::Vector3f Pc = Tcw.map(Pw);
                if (Pc[2] < 0) continue;

                const cv::Point2f p = camera->project(Pc);
                if (!camera->isInImage(p)) continue;

                int lastLevel = lastFrame->key_points[i].octave;
                vector<size_t> indices2 = curFrame->getFeaturesInArea(p.x, p.y,
                                                                      th * lastFrame->key_points[i].size,
                                                                      lastLevel - 1, lastLevel + 1);
                if (indices2.empty()) continue;

                cv::Mat desc1 = mp->getDescriptor();
                int bestDist = TH_HIGH + 1, bestIdx2 = -1;
                for (auto idx2: indices2) {
                    if (curFrame->map_points[idx2]) continue;
                    cv::Mat desc2 = curFrame->descriptors.row(idx2);
                    int dist = DescriptorDistance(desc1, desc2);
                    if (dist < bestDist) {
                        bestDist = dist;
                        bestIdx2 = idx2;
                    }
                }

                if (bestDist <= TH_HIGH) {
                    curFrame->map_points[bestIdx2] = mp;
                    numMatch++;

                    if (be_check_orientation) {
                        float rotAngle = lastFrame->key_points[i].angle - curFrame->key_points[bestIdx2].angle;
                        if (rotAngle < 0.f) rotAngle += 360;
                        int bin = cvRound(rotAngle * factor);
                        if (bin == HISTO_LENGTH) bin = 0;
                        rotHist[bin].push_back(bestIdx2);
                    }
                }
            }
        }

        // apply rotation consistency
        if (be_check_orientation) {
            int ind1 = -1, ind2 = -1, ind3 = -1;
            ComputeThreeMaxima(rotHist, ind1, ind2, ind3);

            for (int i = 0; i < HISTO_LENGTH; ++i) {
                if (i == ind1 || i == ind2 || i == ind3) continue;
                for (auto idx2: rotHist[i]) {
                    curFrame->map_points[idx2] = nullptr;
                    numMatch--;
                }
            }
        }

        return numMatch;
    }

    int ORBMatcher::SearchByProjection(const std::shared_ptr<KeyFrame> &lastKF, const std::shared_ptr<Frame> &curFrame,
                                       float th) const {
        const float factor = 1.f / HISTO_LENGTH;
        vector<int> rotHist[HISTO_LENGTH];

        const Camera *camera = Camera::getCamera();
        const Pose &Tcw = curFrame->T_cw;
        const vector<shared_ptr<MapPoint>> mapPoints = lastKF->getMapPoints();
        int numMatch = 0;

        for (int i = 0; i < lastKF->num_kps; ++i) {
            const shared_ptr<MapPoint> &mp = mapPoints[i];
            if (mp != nullptr) {
                if (mp->isBad()) {
                    continue;
                }

                const Eigen::Vector3f Pw = mp->getPos();
                const Eigen::Vector3f Pc = Tcw.map(Pw);
                if (Pc[2] < 0) continue;

                const cv::Point2f p = camera->project(Pc);
                if (!camera->isInImage(p)) continue;

                int lastLevel = lastKF->key_points[i].octave;
                vector<size_t> indices2 = curFrame->getFeaturesInArea(p.x, p.y,
                                                                      th * lastKF->key_points[i].size,
                                                                      lastLevel - 1, lastLevel + 1);
                if (indices2.empty()) continue;

                cv::Mat desc1 = mp->getDescriptor();
                int bestDist = TH_HIGH + 1, bestIdx2 = -1;
                for (auto idx2: indices2) {
                    if (curFrame->map_points[idx2]) continue;
                    cv::Mat desc2 = curFrame->descriptors.row(idx2);
                    int dist = DescriptorDistance(desc1, desc2);
                    if (dist < bestDist) {
                        bestDist = dist;
                        bestIdx2 = idx2;
                    }
                }

                if (bestDist <= TH_HIGH) {
                    curFrame->map_points[bestIdx2] = mp;
                    numMatch++;

                    if (be_check_orientation) {
                        float rotAngle = lastKF->key_points[i].angle - curFrame->key_points[bestIdx2].angle;
                        if (rotAngle < 0.f) rotAngle += 360;
                        int bin = cvRound(rotAngle * factor);
                        if (bin == HISTO_LENGTH) bin = 0;
                        rotHist[bin].push_back(bestIdx2);
                    }
                }
            }
        }

        // apply rotation consistency
        if (be_check_orientation) {
            int ind1 = -1, ind2 = -1, ind3 = -1;
            ComputeThreeMaxima(rotHist, ind1, ind2, ind3);

            for (int i = 0; i < HISTO_LENGTH; ++i) {
                if (i == ind1 || i == ind2 || i == ind3) continue;
                for (auto idx2: rotHist[i]) {
                    curFrame->map_points[idx2] = nullptr;
                    numMatch--;
                }
            }
        }

        return numMatch;
    }

    int ORBMatcher::SearchByProjection(const std::shared_ptr<Frame> &frame,
                                       const std::vector<std::shared_ptr<MapPoint>> &mapPoints, float th) const {
        int numMatch = 0;
        int numOutViewAndBad = 0, fail1 = 0, fail2 = 0;
        for (const auto &mp: mapPoints) {
            if (!mp->track_in_view || mp->isBad()) {
                numOutViewAndBad++;
                continue;
            }

            const int predictLevel = mp->track_scale_level;

            float radius = th;
            if (mp->track_view_cos > 0.998) radius *= 2.5f;
            else radius *= 4.f;
            radius *= ORBExtractor::getScaleFactor(predictLevel);

            const vector<size_t> indices = frame->getFeaturesInArea(mp->track_proj_x, mp->track_proj_y,
                                                                    radius,
                                                                    predictLevel - 1, predictLevel);

            if (indices.empty()) {
                continue;
            }

            const cv::Mat descMP = mp->getDescriptor();

            int bestDist = 256, bestLevel = -1;
            int secondDist = 257, secondLevel = -1;
            int bestIdx = -1;

            // get best and second matches with near key-points
            for (auto idx: indices) {
                if (frame->map_points[idx] && !frame->map_points[idx]->isBad()) continue;

                const cv::Mat &desc = frame->descriptors.row(idx);
                const int dist = DescriptorDistance(descMP, desc);

                if (dist < bestDist) {
                    secondDist = bestDist;
                    bestDist = dist;
                    secondLevel = bestLevel;
                    bestLevel = frame->key_points[idx].octave;
                    bestIdx = idx;
                } else if (dist < secondDist) {
                    secondDist = dist;
                    secondLevel = frame->key_points[idx].octave;
                }
            }

            // apply ratio to second match
            if (bestDist <= TH_HIGH) {
                if (bestLevel == secondLevel && bestDist > nn_ratio * secondDist) {
                    fail1++;
                    continue;
                }
                frame->map_points[bestIdx] = mp;
                numMatch++;
            } else fail2++;
        }

        tracker_logger << titles[0] << "out view and bad " << numOutViewAndBad << ", fail1 " << fail1 << ", fail2 "
                       << fail2 << "\n";

        return numMatch;
    }

    int ORBMatcher::SearchForTriangulation(const std::shared_ptr<KeyFrame> &keyFrame1,
                                           const std::shared_ptr<KeyFrame> &keyFrame2,
                                           std::vector<int> &matches12) const {
        const DBoW2::FeatureVector &featureVector1 = keyFrame1->feature_vector;
        const DBoW2::FeatureVector &featureVector2 = keyFrame2->feature_vector;

        // compute epipolar point in first image
        const Pose T1w = keyFrame1->getPose();
        const Pose T2w = keyFrame2->getPose();
        const Camera *camera_ptr = Camera::getCamera();
        const Eigen::Matrix3f K = camera_ptr->K;

        const Eigen::Matrix3f R12 = T1w.R * T2w.R.transpose();
        const Eigen::Vector3f t12 = T1w.t - R12 * T2w.t;
        const Eigen::Matrix3f F12 = K.transpose().inverse() * lie::Hatf(t12) * R12 * K.inverse();

        // find matches between not tracked key-points
        // matching speed-up by ORB vocabulary, compare only ORB that share the same node
        int numMatch = 0;
        vector<bool> vecBeMatched2(keyFrame2->num_kps, false);
        matches12 = vector<int>(keyFrame1->num_kps, -1);

        vector<int> rotHist[HISTO_LENGTH];
        for (auto hist: rotHist) {
            hist.reserve(300);
        }
        const float factor = 1.f / HISTO_LENGTH;

        auto iter1 = featureVector1.begin();
        auto iter2 = featureVector2.begin();
        auto iterEnd1 = featureVector1.end();
        auto iterEnd2 = featureVector2.end();
        while (iter1 != iterEnd1 && iter2 != iterEnd2) {
            if (iter1->first == iter2->first) {
                for (unsigned int i = 0, end = iter1->second.size(); i < end; ++i) {
                    const unsigned int idx1 = iter1->second[i];
                    if (keyFrame1->hasMapPoint(idx1)) continue;

                    const cv::KeyPoint &kp1 = keyFrame1->key_points[idx1];
                    const cv::Mat &desc1 = keyFrame1->descriptors.row(idx1);

                    // epipolar line in second image l = x1^T F12
                    const float a = kp1.pt.x * F12(0, 0) + kp1.pt.y * F12(1, 0) + F12(2, 0);
                    const float b = kp1.pt.x * F12(0, 1) + kp1.pt.y * F12(1, 1) + F12(2, 1);
                    const float c = kp1.pt.x * F12(0, 2) + kp1.pt.y * F12(1, 2) + F12(2, 2);
                    const float den = a * a + b * b;
                    if (den == 0) continue;

                    int bestDist = TH_LOW, bestIdx2 = -1;
                    for (auto idx2: iter2->second) {
                        if (vecBeMatched2[idx2] || keyFrame2->hasMapPoint(idx2)) continue;

                        const cv::KeyPoint &kp2 = keyFrame2->key_points[idx2];
                        const cv::Mat &desc2 = keyFrame2->descriptors.row(idx2);
                        const int dist = DescriptorDistance(desc1, desc2);

                        if (dist > bestDist) continue;

                        const float num = a * kp2.pt.x + b * kp2.pt.y + c;
                        if (num * num / den < 3.841 * kp2.size * kp2.size) {
                            bestIdx2 = idx2;
                            bestDist = dist;
                        }
                    }

                    if (bestIdx2 > 0) {
                        const cv::KeyPoint &kp2 = keyFrame2->key_points[bestIdx2];
                        matches12[idx1] = bestIdx2;
                        vecBeMatched2[bestIdx2] = true;
                        numMatch++;

                        if (be_check_orientation) {
                            float rotAngle = kp1.angle - kp2.angle;
                            if (rotAngle < 0) rotAngle += 360.f;
                            int bin = cvRound(rotAngle * factor);
                            if (bin == HISTO_LENGTH) bin = 0;
                            rotHist[bin].push_back(idx1);
                        }
                    }
                }
                iter1++;
                iter2++;
            } else if (iter1->first < iter2->first) {
                iter1 = featureVector1.lower_bound(iter2->first);
            } else {
                iter2 = featureVector2.lower_bound(iter1->first);
            }
        }

        if (be_check_orientation) {
            int ind1 = -1, ind2 = -1, ind3 = -1;
            ComputeThreeMaxima(rotHist, ind1, ind2, ind3);

            for (int i = 0; i < HISTO_LENGTH; ++i) {
                if (i == ind1 || i == ind2 || i == ind3) continue;
                for (auto idx1: rotHist[i]) {
                    matches12[idx1] = -1;
                    numMatch--;
                }
            }
        }

        return numMatch;
    }

    int ORBMatcher::SearchByProjection(const std::shared_ptr<KeyFrame> &keyFrame,
                                       const std::vector<std::shared_ptr<MapPoint>> &mapPoints, Map *pointMap,
                                       float th) {
        int numMatch = 0;
        const Camera *camera = Camera::getCamera();
        const Pose Tcw = keyFrame->getPose();
        const Eigen::Vector3f Ow = keyFrame->getCameraCenter();
        const int minLevel = 0, maxLevel = ORBExtractor::getNumLevels() - 1;

        for (const auto &mp: mapPoints) {
            if (mp == nullptr || mp->isBad() || mp->isObserveKeyFrame(keyFrame)) continue;

            const Eigen::Vector3f Pw = mp->getPos();
            const Eigen::Vector3f Pc = Tcw.R * Pw + Tcw.t;
            if (Pc[2] < 0) continue;

            const cv::Point2f p = camera->project(Pc);
            if (!camera->isInImage(p)) continue;

            const Eigen::Vector3f OP = Pw - Ow;
            const float distance = OP.norm();
            const float maxDistance = mp->getMaxDistanceInvariance();
            const float minDistance = mp->getMinDistanceInvariance();
            if (distance < minDistance || distance > maxDistance) continue;

            const Eigen::Vector3f Pn = mp->getAverageDirection();
            if (OP.dot(Pn) < 0.5 * distance) continue;

            int predictLevel = mp->predictScaleLevel(distance);
            const float radius = th * ORBExtractor::getScaleFactor(predictLevel);
            vector<size_t> indices1 = keyFrame->getFeaturesInArea(p.x, p.y, radius, predictLevel - 1, predictLevel);

            if (indices1.empty()) continue;

            // match to the most similar keypoint in the radius
            int bestDist = TH_LOW + 1, bestIdx1 = -1;
            cv::Mat desc2 = mp->getDescriptor();
            for (auto &idx1: indices1) {
                const cv::KeyPoint &kp = keyFrame->key_points[idx1];
                const float squareError2 = (p.x - kp.pt.x) * (p.x - kp.pt.x) + (p.y - kp.pt.y) * (p.y - kp.pt.y);
                if (squareError2 > 5.991 * ORBExtractor::getSquareSigma(kp.octave)) continue;

                cv::Mat desc1 = keyFrame->descriptors.row(idx1);
                int dist = DescriptorDistance(desc1, desc2);
                if (dist < bestDist) {
                    bestDist = dist;
                    bestIdx1 = idx1;
                }
            }

            if (bestIdx1 != -1) {
                shared_ptr<MapPoint> mp1 = keyFrame->getMapPoint(bestIdx1);
                if (mp1 == nullptr) {
                    mp->addObservation(keyFrame, bestIdx1);
                    keyFrame->addMapPoint(mp, bestIdx1);
                } else if (!mp1->isBad()) {
                    if (mp1->getNumObs() > mp->getNumObs()) {
                        mp->replace(mp1);
                    } else {
                        mp1->replace(mp);
                    }
                }

                numMatch++;
            }
        }

        return numMatch;
    }

    void ORBMatcher::ComputeThreeMaxima(std::vector<int> *histo, int &ind1, int &ind2, int &ind3) {
        int max1 = 0, max2 = -1, max3 = -2;
        for (int i = 0; i < HISTO_LENGTH; ++i) {
            const int n = (int) histo[i].size();
            if (n > max1) {
                max3 = max2;
                max2 = max1;
                max1 = n;
                ind3 = ind2;
                ind2 = ind1;
                ind1 = i;
            } else if (n > max2) {
                max3 = max2;
                max2 = n;
                ind3 = ind2;
                ind2 = i;
            } else if (n > max3) {
                max3 = n;
                ind3 = i;
            }
        }

        if (max2 < max1 / 10) {
            ind2 = -1;
            ind3 = -1;
        } else if (max3 < max1 / 10) {
            ind3 = -1;
        }
    }

} // mono_orb_slam3