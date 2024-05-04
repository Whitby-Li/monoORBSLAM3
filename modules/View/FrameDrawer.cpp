//
// Created by whitby on 8/25/23.
//

#include <opencv2/imgproc.hpp>
#include "FrameDrawer.h"

using namespace std;

namespace mono_orb_slam3 {

    FrameDrawer::FrameDrawer(int w, int h) {
        state = Tracking::State::NO_IMAGE_YET;
        img = cv::Mat(h, w, CV_8UC3, cv::Scalar(0, 0, 0));
    }

    cv::Mat FrameDrawer::DrawFrame() {
        cv::Mat drawImg;
        vector<int> matches12;
        vector<cv::KeyPoint> keyPoints, initialKeyPoints;
        vector<bool> vecBeTracked;
        int trackingState;

        {
            lock_guard<mutex> lock(frame_drawer_mutex);
            trackingState = state;
            img.copyTo(drawImg);
            keyPoints = key_points;

            switch (state) {
                case Tracking::NOT_INITIALIZE: {
                    initialKeyPoints = initial_key_points;
                    matches12 = initial_matches;
                    break;
                }
                case Tracking::OK:
                    vecBeTracked = vec_is_tracked;
                    break;
                default:
                    break;
            }
        }

        if (drawImg.channels() < 3)
            cv::cvtColor(drawImg, drawImg, cv::COLOR_GRAY2BGR);

        switch (trackingState) {
            case Tracking::NOT_INITIALIZE: {
                for (unsigned int i = 0, end = matches12.size(); i < end; ++i) {
                    if (matches12[i] >= 0)
                        cv::line(drawImg, initialKeyPoints[i].pt, keyPoints[matches12[i]].pt, cv::Scalar(0, 255, 0));
                }
                break;
            }
            case Tracking::OK: {
                num_tracked = 0;
                const float r = 5;
                const int n = (int) keyPoints.size();
                for (int i = 0; i < n; ++i) {
                    if (vecBeTracked[i]) {
                        const cv::KeyPoint &kp = keyPoints[i];
                        cv::Point2f pt1(kp.pt.x - r, kp.pt.y - r);
                        cv::Point2f pt2(kp.pt.x + r, kp.pt.y + r);

                        cv::rectangle(drawImg, pt1, pt2, cv::Scalar(0, 255, 0));
                        cv::circle(drawImg, kp.pt, 2, cv::Scalar(0, 255, 0));
                        num_tracked++;
                    }
                }
                break;
            }
            default:
                break;
        }

        DrawTextInfo(drawImg, trackingState);

        return drawImg;
    }

    void FrameDrawer::DrawTextInfo(cv::Mat &image, int curState) const {
        stringstream ss;
        switch (curState) {
            case Tracking::NOT_INITIALIZE: {
                ss << " TRYING TO INITIALIZE ";
                break;
            }
            case Tracking::OK: {
                ss << " SLAM MODE | " << "Matches: " << num_tracked;
                break;
            }
            case Tracking::LOST: {
                ss << " TRACK LOST ";
                break;
            }
            default:
                break;
        }

        int baseline = 0;
        cv::Size textSize = cv::getTextSize(ss.str(), cv::FONT_HERSHEY_PLAIN, 1, 1, &baseline);

        image.rowRange(image.rows - textSize.height - 10, image.rows) = cv::Mat::zeros(textSize.height + 10, image.cols, image.type());
        cv::putText(image, ss.str(), cv::Point(5, image.rows - 5), cv::FONT_HERSHEY_PLAIN, 1,
                    cv::Scalar(255, 255, 255), 1, 8);
    }

    void FrameDrawer::Update(Tracking *tracker) {
        lock_guard<mutex> lock(frame_drawer_mutex);
        const shared_ptr<Frame> currentFrame = tracker->current_frame;
        currentFrame->img.copyTo(img);
        key_points.clear();
        key_points = currentFrame->raw_key_points;
        num_kps = currentFrame->num_kps;
        vec_is_tracked = vector<bool>(num_kps, false);
        state = tracker->state;

        switch (state) {
            case Tracking::NOT_INITIALIZE: {
                initial_key_points = tracker->last_frame->raw_key_points;
                initial_matches = tracker->initial_matches;
                break;
            }
            case Tracking::OK: {
                for (int i = 0; i < num_kps; ++i) {
                    const shared_ptr<MapPoint> mp = currentFrame->map_points[i];
                    if (mp && !mp->isBad()) {
                        vec_is_tracked[i] = true;
                    }
                }
                break;
            }
            default:
                break;
        }
    }

} // mono_orb_slam3