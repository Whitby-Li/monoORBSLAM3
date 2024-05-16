//
// Created by whitby on 8/26/23.
//

#include "Viewer.h"
#include <pangolin/pangolin.h>
#include <opencv2/opencv.hpp>

using namespace std;

namespace mono_orb_slam3 {

    Viewer::Viewer(System *systemPtr, FrameDrawer *frameDrawer, MapDrawer *mapDrawer,
                   const cv::FileNode &viewNode, bool bRecord)
            : system(systemPtr), frame_drawer(frameDrawer), map_drawer(mapDrawer), finish_requested(false),
              be_finished(false), be_stopped(true), stop_requested(false) {
        fps = Camera::getCamera()->fps;
        delta_ms = 1000 / fps;
        record = bRecord;

        width = viewNode["Width"];
        height = viewNode["Height"];
        view_point_x = viewNode["ViewPointX"];
        view_point_y = viewNode["ViewPointY"];
        view_point_z = viewNode["ViewPointZ"];
        view_point_f = viewNode["ViewPointF"];
    }

    void Viewer::Run() {
        be_finished = false;
        be_stopped = false;

        pangolin::CreateWindowAndBind("mono-orb-slam3: map viewer", width + 174, height);

        // 3D mouse handler requires depth testing to be enabled
        glEnable(GL_DEPTH_TEST);

        // issue specific OpenGl we might need
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

        pangolin::CreatePanel("menu").SetBounds(0.0, 1.0, 0.0, pangolin::Attach::Pix(174));
        pangolin::Var<bool> menuFollowCamera("menu.Follow Camera", true, true);
        pangolin::Var<bool> menuShowPoints("menu.Show Points", true, true);
        pangolin::Var<bool> menuShowKeyFrames("menu.Show KeyFrames", true, true);
        pangolin::Var<bool> menuShowGraph("menu.Show Graph", true, true);
        pangolin::Var<bool> menuReset("menu.Reset", false, false);

        pangolin::OpenGlRenderState s_cam(
                pangolin::ProjectionMatrix(width, height, view_point_f, view_point_f, width * 0.5, height * 0.5, 0.05,
                                           5000),
                pangolin::ModelViewLookAt(view_point_x, view_point_y, view_point_z, 0, 0, 0, pangolin::AxisNegY)
        );

        // add named OpenGL viewport to window and provide 3D handler
        pangolin::View &d_cam = pangolin::CreateDisplay()
                .SetBounds(0.0, 1.0, pangolin::Attach::Pix(174), 1.0, 1.f)
                .SetHandler(new pangolin::Handler3D(s_cam));

        pangolin::OpenGlMatrix Twc;
        Twc.SetIdentity();

        /*cv::namedWindow("mono-orb-slam3: current frame");
        cv::moveWindow("mono-orb-slam3: current frame", width + 174, 50);*/

        bool beFollow = true;
        cv::VideoWriter *frameVideo;
        if (record) {
            const string recordURI = cv::format("ffmpeg:[fps=%d,bps=2388608,overwrite]//%s/map.avi", 30,
                                                system->getSaveFolder().c_str());
            pangolin::DisplayBase().RecordOnRender(recordURI);

            frameVideo = new cv::VideoWriter(system->getSaveFolder() + "/frame.avi",
                                       cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), 30,
                                       {Camera::getCamera()->width, Camera::getCamera()->height});
        }

        cv::namedWindow("mono_orb_slam3: frame");
        cv::moveWindow("mono_orb_slam3: frame", width + 174, 0);

        while (true) {
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

            map_drawer->getCurrentOpenGLCameraMatrix(Twc);

            if (menuFollowCamera && beFollow) {
                s_cam.Follow(Twc);
            } else if (menuFollowCamera && !beFollow) {
                s_cam.SetModelViewMatrix(
                        pangolin::ModelViewLookAt(view_point_x, view_point_y, view_point_z, 0, 0, 0, 0.0, -1.0, 0.0));
                s_cam.Follow(Twc);
                beFollow = true;
            } else if (!menuFollowCamera && beFollow) {
                beFollow = false;
            }
            glClearColor(1.f, 1.f, 1.f, 1.f);

            d_cam.Activate(s_cam);
            map_drawer->DrawCurrentCamera(Twc);
            if (menuShowKeyFrames || menuShowGraph)
                map_drawer->DrawKeyFrames(menuShowKeyFrames, menuShowGraph);
            if (menuShowPoints)
                map_drawer->DrawMapPoints();

            pangolin::FinishFrame();

            cv::Mat img = frame_drawer->DrawFrame();
            if (record)
                frameVideo->write(img);
            cv::imshow("mono_orb_slam3: frame", img);
            cv::waitKey(delta_ms);

            if (menuReset) {
                menuShowGraph = true;
                menuShowKeyFrames = true;
                menuShowPoints = true;
                beFollow = true;
                menuFollowCamera = true;
                system->Reset();
                menuReset = false;
            }

            if (stop()) {
                while (isStopped()) {
                    usleep(3000);
                }
            }

            if (checkFinish()) break;
        }

        if (record) {
            frameVideo->release();
            delete frameVideo;
        }
        setFinish();
    }

    void Viewer::requestFinish() {
        lock_guard<mutex> lock(finish_mutex);
        finish_requested = true;
    }

    bool Viewer::checkFinish() {
        lock_guard<mutex> lock(finish_mutex);
        return finish_requested;
    }

    void Viewer::setFinish() {
        lock_guard<mutex> lock(finish_mutex);
        be_finished = true;
    }

    bool Viewer::isFinished() {
        lock_guard<mutex> lock(finish_mutex);
        return be_finished;
    }

    void Viewer::requestStop() {
        lock_guard<mutex> lock(stop_mutex);
        if (!be_stopped) stop_requested = true;
    }

    bool Viewer::isStopped() {
        lock_guard<mutex> lock(stop_mutex);
        return be_stopped;
    }

    bool Viewer::stop() {
        {
            lock_guard<mutex> lock(finish_mutex);
            if (finish_requested) return false;
        }
        {
            lock_guard<mutex> lock(stop_mutex);
            if (stop_requested) {
                be_stopped = true;
                stop_requested = false;
                return true;
            }
        }

        return false;
    }

    void Viewer::release() {
        lock_guard<mutex> lock(stop_mutex);
        be_stopped = false;
    }

} // mono_orb_slam3