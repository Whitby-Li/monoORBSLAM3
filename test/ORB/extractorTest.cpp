//
// Created by whitby on 8/22/23.
//

#include "ORB/ORBExtractor.h"

#include <opencv2/opencv.hpp>
#include <iostream>

using namespace std;
using namespace mono_orb_slam3;

int main(int argc, char *argv[]) {
    if (argc != 3) {
        cout << "Usage: ./extractor_test orb.yaml image_folder" << endl;
        return -1;
    }

    // 1. load orb.yaml
    cv::FileStorage fs(argv[1], cv::FileStorage::READ);
    if (!fs.isOpened()) {
        cerr << "Fail to load " << argv[1] << endl;
        return -1;
    }
    cv::FileNode orbNode = fs["ORB"];
    if (orbNode.empty()) {
        cerr << "there are not 'Pyramid' node" << endl;
        return -1;
    }

    // 2. create ORB extractor
    ORBExtractor orbExtractor(orbNode["Features"], orbNode["ScaleFactor"], orbNode["Levels"], orbNode["IniThFAST"], orbNode["MinThFAST"]);

    // 3. main loop
    const string imgFolder = argv[2];
    int idx = 0;
    while (true) {
        // load image
        cv::Mat img = cv::imread(imgFolder + cv::format("/%08d.png", idx), cv::IMREAD_GRAYSCALE);
        if (img.empty()) {
            cout << "fail to load image" << endl;
            break;
        }

        // extractor key-points
        vector<cv::KeyPoint> keyPoints;
        cv::Mat descriptors;
        orbExtractor(img, keyPoints, descriptors);
        cout << "iter: " << idx << ", extractor " << keyPoints.size() << " key-points" << endl;

        // draw key-points and show
        cv::Mat displayImage;
        cv::drawKeypoints(img, keyPoints, displayImage, cv::Scalar(0, 255, 0));
        cv::imshow("orb features", displayImage);
        if (cv::waitKey(1) == 27){
            break;
        }

        idx++;
    }

    return 0;
}