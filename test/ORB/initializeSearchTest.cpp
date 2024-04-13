//
// Created by whitby on 8/23/23.
//

#include "BasicObject/Frame.h"
#include "ORB/ORBMatcher.h"
#include "Data.h"

#include <opencv2/opencv.hpp>
#include <iostream>

using namespace std;
using namespace mono_orb_slam3;

int main(int argc, char *argv[]) {
    if (argc != 2) {
        cout << "usage: ./initialize_search_test data_folder" << endl;
        return -1;
    }

    // 1. load camera data
    const string data_folder = argv[1];
    vector<double> timestamps;
    loadCameraData(data_folder + "/camera.txt", timestamps);
    int numCamera = (int) timestamps.size();
    cout << "load " << numCamera << " camera data" << endl;

    // 2. create ORB extractor
    ORBExtractor extractor(8000, 1.2, 8, 20, 7);

    // 3. main loop
    ORBMatcher matcher(0.9, true);
    shared_ptr<Frame> lastFrame, currentFrame;
    vector<cv::Point2f> predictMatch;
    vector<int> matches12;


    int idx = 0;
    while (idx < 10) {
        cout << "iter: " << idx << endl;

        cv::Mat img = cv::imread(data_folder + cv::format("/images/%08d.png", idx), cv::IMREAD_GRAYSCALE);
        currentFrame = make_shared<Frame>(img, timestamps[idx], &extractor, Bias());
        if (lastFrame == nullptr) {
            predictMatch.resize(currentFrame->num_kps);
            for (int i = 0; i < currentFrame->num_kps; ++i) {
                predictMatch[i] = currentFrame->key_points[i].pt;
            }
            lastFrame = currentFrame;
        } else {
            int numMatch = matcher.SearchForInitialization(lastFrame, currentFrame, predictMatch, matches12);
            cout << "second frame (id " << currentFrame->id << "), match " << numMatch << endl;

            // draw matches
            vector<cv::DMatch> goodMatches;
            for (int i = 0; i < lastFrame->num_kps; ++i) {
                if (matches12[i] > 0) {
                    goodMatches.emplace_back(i, matches12[i], 50);
                }
            }
            cv::Mat imgMatch;
            cv::drawMatches(lastFrame->img, lastFrame->key_points, currentFrame->img, currentFrame->key_points, goodMatches, imgMatch);
            cv::imshow("matches", imgMatch);
            cv::waitKey();
        }
        idx++;
    }

    return 0;
}