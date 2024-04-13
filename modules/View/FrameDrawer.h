//
// Created by whitby on 8/25/23.
//

#ifndef MONO_ORB_SLAM3_FRAMEDRAWER_H
#define MONO_ORB_SLAM3_FRAMEDRAWER_H

#include "Frontend/Tracking.h"

#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <mutex>

namespace mono_orb_slam3 {

    class Tracking;

    class FrameDrawer {
    public:
        FrameDrawer(int w, int h);

        // update info from the last processed frame
        void Update(Tracking *tracker);

        // draw last processed frame
        cv::Mat DrawFrame();

    protected:
        void DrawTextInfo(cv::Mat &image, int curState, cv::Mat &imgText) const;

        // info of the frame to be drawn
        cv::Mat img;
        int num_kps = 0;
        std::vector<cv::KeyPoint> key_points;
        std::vector<bool> vec_is_tracked;
        int num_tracked = 0;
        std::vector<cv::KeyPoint> initial_key_points;
        std::vector<int> initial_matches;
        int state;

        std::mutex frame_drawer_mutex;
    };

} // mono_orb_slam3

#endif //MONO_ORB_SLAM3_FRAMEDRAWER_H
