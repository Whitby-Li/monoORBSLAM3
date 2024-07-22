//
// Created by whitby on 8/22/23.
//

#include "System.h"
#include "Sensor/Camera.h"
#include "Sensor/Imu.h"
#include "ORB/ORBVocabulary.h"

#include <unistd.h>
#include <iostream>
#include <iomanip>
#include <utility>
#include <filesystem>

using namespace std;

namespace mono_orb_slam3 {

    System::System(const std::string &settingYaml, const std::string &vocabularyFile, bool useViewer, bool recordViewer)
            : be_reset(false) {
        cv::FileStorage fs(settingYaml, cv::FileStorage::READ);
        if (!fs.isOpened()) {
            cout << "fail to load " << settingYaml << endl;
            terminate();
        }

        // camera
        Camera::create(fs["Camera"]);
        const Camera *camera_ptr = Camera::getCamera();
        camera_ptr->print();

        // imu
        ImuCalib::create(fs["IMU"]);
        const ImuCalib *imu_ptr = ImuCalib::getImuCalib();
        imu_ptr->print();

        // orb vocabulary
        cout << endl << "Loading ORB Vocabulary. This could take a while..." << endl;
        bool loaded = ORBVocabulary::createORBVocabulary(vocabularyFile);
        if (!loaded) {
            cerr << "Wrong path to vocabulary." << endl;
            cerr << "Fail to open at " << vocabularyFile << endl;
            exit(-1);
        }
        cout << "Vocabulary loaded!" << endl << endl;

        // map
        point_map = new Map();

        // threads
        tracker = new Tracking(fs["ORB"], point_map, this);
        tracking_state = tracker->state;

        local_mapper = new LocalMapping(point_map);
        local_mapping = new thread(&LocalMapping::Run, local_mapper);

        tracker->setLocalMapper(local_mapper);
        local_mapper->setTracker(tracker);

        if (useViewer) {
            cv::FileNode viewNode = fs["View"];
            frame_drawer = new FrameDrawer(camera_ptr->width, camera_ptr->height);
            map_drawer = new MapDrawer(point_map, viewNode);

            viewer = new Viewer(this, frame_drawer, map_drawer, viewNode, recordViewer);
            viewing = new thread(&Viewer::Run, viewer);
            tracker->setViewer(viewer);
        }
    }

    System::~System() {
        local_mapping->join();
        if (viewing != nullptr) viewing->join();
        delete local_mapping;
        delete viewing;
        delete tracker;
        delete local_mapper;
        delete point_map;
        delete viewer;
        delete frame_drawer;
        delete map_drawer;
        ImuCalib::destroy();
        Camera::destroy();
    }

    void System::Track(double timeStamp, cv::Mat grayImg, const vector<ImuData> &imus) {
        // check reset
        {
            lock_guard<mutex> lock(reset_mutex);
            if (be_reset) {
                tracker->reset();
                be_reset = false;

                if (trajectory_out.is_open()) trajectory_out.close();
                trajectory_out.open(save_folder + "/trajectory.txt");
                trajectory_out << setiosflags(ios::fixed) << setprecision(6);
            }
        }

        tracker->Track(timeStamp, std::move(grayImg), imus);
        {
            lock_guard<mutex> lock(state_mutex);
            tracking_state = tracker->state;
        }

        // recorder trajectory
        if (local_mapper->finishImuInit() && tracker->current_frame) {
            auto curFrame = tracker->current_frame;
            const Pose Twb = curFrame->T_wb;
            const Eigen::Vector3f twb = Twb.t;
            const Eigen::Quaternionf q(Twb.R.transpose());
            const Eigen::Vector3f velo = curFrame->v_w;

            trajectory_out << timeStamp << " " << twb.x() << " " << twb.y() << " " << twb.z()
                           << " " << q.w() << " " << q.x() << " " << q.y() << " " << q.z()
                           << " " << velo.x() << " " << velo.y() << " " << velo.z() << endl;
            trajectory_out.flush();
        }
    }

    void System::Reset() {
        lock_guard<mutex> lock(reset_mutex);
        be_reset = true;
    }

    void System::ShutDown() {
        // save frame trajectory
        if (trajectory_out.is_open()) trajectory_out.close();

        local_mapper->requestFinish();

        if (viewer != nullptr) {
            viewer->requestFinish();
            while (!viewer->isFinished())
                usleep(5000);
        }

        while (!local_mapper->isFinish()) {
            usleep(5000);
        }

        if (viewer != nullptr)
            pangolin::BindToContext("mono-orb-slam3: map viewer");
    }

    void System::saveKeyFrameTrajectory() {
        const string fileName = save_folder + "/kf_trajectory.txt";
        cout << endl << "saving keyframe trajectory to " << fileName << "..." << endl;

        vector<shared_ptr<KeyFrame>> keyFrames = point_map->getAllKeyFrames();
        sort(keyFrames.begin(), keyFrames.end(),
             [](const shared_ptr<KeyFrame> &kf1, const shared_ptr<KeyFrame> &kf2) { return kf1->id < kf2->id; });

        ofstream outFile(fileName);
        outFile << setiosflags(ios::fixed) << setprecision(6);
        for (const auto &kf: keyFrames) {
            const Pose Twb = kf->getImuPose();
            const Eigen::Vector3f twb = Twb.t;
            const Eigen::Quaternionf q(Twb.R.transpose());
            const Eigen::Vector3f velo = kf->getVelocity();
            const Bias &bias = kf->pre_integrator->updated_bias;

            outFile << kf->timestamp << " " << twb.x() << " " << twb.y() << " " << twb.z()
                    << " " << q.w() << " " << q.x() << " " << q.y() << " " << q.z()
                    << " " << velo.x() << " " << velo.y() << " " << velo.z() << " "
                    << bias.bg.x() << " " << bias.bg.y() << " " << bias.bg.z() << " "
                    << bias.ba.x() << " " << bias.ba.y() << " " << bias.ba.z() << endl;
        }
        outFile.close();
        cout << "trajectory saved!" << endl;
    }

    void System::savePointCloudMap() {
        const string fileName = save_folder + "/map.pcd";
        cout << endl << "save point cloud map to " << fileName << " ... ";

        vector<shared_ptr<MapPoint>> mapPoints = point_map->getAllMapPoints();

        int nPoints = (int) mapPoints.size();
        ofstream fout(fileName);
        fout << fixed << setprecision(2);

        fout << "# .PCD v0.7 - Point Cloud Data file format" << endl;
        fout << "VERSION 0.7" << endl;
        fout << "FIELDS x y z" << std::endl;
        fout << "SIZE 4 4 4" << std::endl;
        fout << "TYPE F F F" << std::endl;
        fout << "COUNT 1 1 1" << std::endl;
        fout << "WIDTH " << nPoints << std::endl;
        fout << "HEIGHT 1" << std::endl;
        fout << "VIEWPOINT 0 0 0 1 0 0 0" << std::endl;
        fout << "POINTS " << nPoints << std::endl;
        fout << "DATA ascii" << std::endl;

        for (const auto &mp: mapPoints) {
            const Eigen::Vector3f &pos = mp->getPos();
            fout << pos.x() << " " << pos.y() << " " << pos.z() << endl;
        }

        fout.close();
    }

    void System::saveKeyFrameDepth() {
        const string fileName = save_folder + "/kf_depth.txt";
        cout << endl << "save keyframe's depth to " << fileName << " ... ";

        vector<shared_ptr<KeyFrame>> keyFrames = point_map->getAllKeyFrames();
        ofstream fout(fileName);
        fout << fixed << setprecision(2);

        for (const auto &kf: keyFrames) {
            const Pose &Tcw = kf->getPose();
            const vector<cv::KeyPoint> &keyPoints = kf->raw_key_points;
            const vector<shared_ptr<MapPoint>> &mapPoints = kf->getMapPoints();

            fout << kf->timestamp << " ";

            for (int i = 0; i < kf->num_kps; ++i) {
                const auto &mp = mapPoints[i];
                if (mp && !mp->isBad()) {
                    float depth = Tcw.map(mp->getPos()).z();
                    fout << keyPoints[i].pt.x << " " << keyPoints[i].pt.y << " " << depth << " ";
                }
            }

            fout << endl;
        }

        fout.close();
    }

    int System::getTrackingState() {
        lock_guard<mutex> lock(state_mutex);
        return tracking_state;
    }

    void System::setSaveFolder(const std::string &path) {
        if (!filesystem::exists(path)) {
            cout << "create save folder: " << path << endl;
            filesystem::create_directories(path);
        }
        save_folder = path;

        if (trajectory_out.is_open()) trajectory_out.close();
        trajectory_out.open(save_folder + "/trajectory.txt");
        trajectory_out << setiosflags(ios::fixed) << setprecision(6);
    }

} // mono_orb_slam3