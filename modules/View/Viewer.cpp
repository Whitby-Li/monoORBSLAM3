//
// Created by whitby on 8/26/23.
//

#include "Viewer.h"
#include <pangolin/pangolin.h>
#include <opencv2/highgui.hpp>

using namespace std;

namespace mono_orb_slam3 {

    Viewer::Viewer(System *systemPtr, FrameDrawer *frameDrawer, MapDrawer *mapDrawer, const cv::FileNode &cameraNode,
                   const cv::FileNode &viewNode)
            : system(systemPtr), frame_drawer(frameDrawer), map_drawer(mapDrawer), finish_requested(false),
              be_finished(false), be_stopped(true), stop_requested(false) {
        int fps = cameraNode["fps"];
        delta_ms = 1000 / fps;

        view_point_x = viewNode["ViewPointX"];
        view_point_y = viewNode["ViewPointY"];
        view_point_z = viewNode["ViewPointZ"];
        view_point_f = viewNode["ViewPointF"];
    }

    void Viewer::Run() {
        be_finished = false;
        be_stopped = false;

        pangolin::CreateWindowAndBind("mono-orb-slam3: map viewer", 1000, 500);
        pangolin::GlRenderBuffer renderBuffer(1000, 500);
        pangolin::GlTexture texture(1000, 500);
        pangolin::GlFramebuffer frameBuffer(texture, renderBuffer);

        cv::VideoWriter video("output.avi", cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), 20, {1220, 1046});

        // 3D mouse handler requires depth testing to be enabled
        glEnable(GL_DEPTH_TEST);

        // issue specific OpenGl we might need
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

        /*pangolin::CreatePanel("menu").SetBounds(0.0, 1.0, 0.0, pangolin::Attach::Pix(174));
        pangolin::Var<bool> menuFollowCamera("menu.Follow Camera", true, true);
        pangolin::Var<bool> menuShowPoints("menu.Show Points", true, true);
        pangolin::Var<bool> menuShowKeyFrames("menu.Show KeyFrames", true, true);
        pangolin::Var<bool> menuShowGraph("menu.Show Graph", true, true);
        pangolin::Var<bool> menuReset("menu.Reset", false, false);*/

        // define camera render object (for view / scene browsing)
        pangolin::OpenGlRenderState s_cam1(
                pangolin::ProjectionMatrix(500, 500, 0.4 * view_point_f, 0.4 * view_point_f, 250,
                                           250, 0.05, 5000),
                pangolin::ModelViewLookAt(0, view_point_y, 0, 0, 0, 0, pangolin::AxisZ)
        );

        pangolin::OpenGlRenderState s_cam2(
                pangolin::ProjectionMatrix(500, 500, view_point_f, view_point_f, 250, 250, 0.05, 2000),
                pangolin::ModelViewLookAt(view_point_x, view_point_y, view_point_z, 0, 0, 0, pangolin::AxisNegY)
        );

        // add named OpenGL viewport to window and provide 3D handler
        pangolin::View &d_cam1 = pangolin::Display("cam1").SetAspect(1.f);
        pangolin::View &d_cam2 = pangolin::Display("cam2").SetAspect(1.f);

        auto &container = pangolin::Display("multi")
            .SetBounds(0.0, 1.0, 0.0, 1.0)
            .SetLayout(pangolin::LayoutEqual)
            .AddDisplay(d_cam1)
            .AddDisplay(d_cam2);

        pangolin::OpenGlMatrix Twc;
        Twc.SetIdentity();

        cv::namedWindow("mono-orb-slam3: current frame");
        cv::moveWindow("mono-orb-slam3: current frame", 1450, 0);

        bool beFollow = true;

        while (true) {
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
            // frameBuffer.Bind();

            map_drawer->getCurrentOpenGLCameraMatrix(Twc);

            s_cam1.Follow(Twc);
            s_cam2.Follow(Twc);
            /*if (menuFollowCamera && beFollow) {
                s_cam.Follow(Twc);
            } else if (menuFollowCamera && !beFollow) {
                s_cam.SetModelViewMatrix(
                        pangolin::ModelViewLookAt(view_point_x, view_point_y, view_point_z, 0, 0, 0, 0.0, -1.0, 0.0));
                s_cam.Follow(Twc);
                beFollow = true;
            } else if (!menuFollowCamera && beFollow) {
                beFollow = false;
            }*/
            glClearColor(1.f, 1.f, 1.f, 1.f);
            d_cam1.Activate(s_cam1);
            map_drawer->DrawCurrentCamera(Twc);
            map_drawer->DrawKeyFrames(true, true);

            d_cam2.Activate(s_cam2);
            map_drawer->DrawCurrentCamera(Twc);
            map_drawer->DrawKeyFrames(true, true);
            map_drawer->DrawMapPoints();

            pangolin::FinishFrame();

            /*glReadBuffer(GL_COLOR_ATTACHMENT0);
            cv::Mat displayImg(1046, 1220, CV_8UC3);
            glReadPixels(0, 0, 1220, 1046, GL_BGR, GL_UNSIGNED_BYTE, displayImg.data);
            cv::flip(displayImg, displayImg, 0);

            cv::imshow("map", displayImg);
            video.write(displayImg);*/

            cv::Mat img = frame_drawer->DrawFrame();
            cv::imshow("mono-orb-slam3: current frame", img);
            cv::waitKey(delta_ms);

            /*if (menuReset) {
                menuShowGraph = true;
                menuShowKeyFrames = true;
                menuShowPoints = true;
                beFollow = true;
                menuFollowCamera = true;
                system->Reset();
                menuReset = false;
            }*/

            if (stop()) {
                while (isStopped()) {
                    usleep(3000);
                }
            }

            if (checkFinish()) break;
        }

        // video.release();
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