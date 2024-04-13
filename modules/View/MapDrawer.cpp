//
// Created by whitby on 8/25/23.
//

#include "MapDrawer.h"

using namespace std;

namespace mono_orb_slam3 {

    MapDrawer::MapDrawer(Map *pointMap, const cv::FileNode &viewNode) : point_map(pointMap) {
        kf_size = viewNode["KeyFrameSize"];
        kf_line_width = viewNode["KeyFrameLineWidth"];
        graph_line_width = viewNode["GraphLineWidth"];
        point_size = viewNode["PointSize"];
        camera_size = viewNode["CameraSize"];
        camera_line_width = viewNode["CameraLineWidth"];
    }

    void MapDrawer::DrawMapPoints() const {
        const vector<shared_ptr<MapPoint>> &mapPoints = point_map->getAllMapPoints();
        const set<shared_ptr<MapPoint>> &referenceMapPoints = point_map->getReferenceMapPoints();

        if (mapPoints.empty()) return;

        glPointSize(point_size);
        glBegin(GL_POINTS);
        glColor3f(0.0, 0.0, 0.0);

        for (const auto &mp: mapPoints) {
            if (!mp || mp->isBad() || referenceMapPoints.count(mp)) continue;
            Eigen::Vector3f pos = mp->getPos();
            glVertex3f(pos[0], pos[1], pos[2]);
        }
        glEnd();

        glPointSize(point_size);
        glBegin(GL_POINTS);
        glColor3f(1.0, 0.0, 0.0);
        for (const auto &mp: referenceMapPoints) {
            if (mp->isBad()) continue;
            Eigen::Vector3f pos = mp->getPos();
            glVertex3f(pos[0], pos[1], pos[2]);
        }
        glEnd();
    }

    void MapDrawer::DrawKeyFrames(bool drawKF, bool drawGraph) {
        const float &w = kf_size;
        const float h = 0.75f * w;
        const float z = 0.6f * w;

        const vector<shared_ptr<KeyFrame>> keyFrames = point_map->getAllKeyFrames();

        if (drawKF) {
            if (reference_kf) {
                const Pose Twc = reference_kf->getInversePose();
                float data[] = {Twc.R(0, 0), Twc.R(1, 0), Twc.R(2, 0), 0,
                                Twc.R(0, 1), Twc.R(1, 1), Twc.R(2, 1), 0,
                                Twc.R(0, 2), Twc.R(1, 2), Twc.R(2, 2), 0,
                                Twc.t(0), Twc.t(1), Twc.t(2), 1};
                glPushMatrix();
                glMultMatrixf(data);
                glLineWidth(kf_line_width);
                glColor3f(1.f, 0.f, 0.f);
                glBegin(GL_LINES);
                glVertex3f(0, 0, 0);
                glVertex3f(w, h, z);
                glVertex3f(0, 0, 0);
                glVertex3f(w, -h, z);
                glVertex3f(0, 0, 0);
                glVertex3f(-w, -h, z);
                glVertex3f(0, 0, 0);
                glVertex3f(-w, h, z);

                glVertex3f(w, h, z);
                glVertex3f(w, -h, z);

                glVertex3f(-w, h, z);
                glVertex3f(-w, -h, z);

                glVertex3f(-w, h, z);
                glVertex3f(w, h, z);

                glVertex3f(-w, -h, z);
                glVertex3f(w, -h, z);
                glEnd();

                glPopMatrix();
            }

            for (const auto &kf: keyFrames) {
                const Pose Twc = kf->getInversePose();
                float data[] = {Twc.R(0, 0), Twc.R(1, 0), Twc.R(2, 0), 0,
                                Twc.R(0, 1), Twc.R(1, 1), Twc.R(2, 1), 0,
                                Twc.R(0, 2), Twc.R(1, 2), Twc.R(2, 2), 0,
                                Twc.t(0), Twc.t(1), Twc.t(2), 1};

                glPushMatrix();

                glMultMatrixf(data);

                glLineWidth(kf_line_width);
                glColor3f(0.0f, 0.0f, 1.0f);
                glBegin(GL_LINES);
                glVertex3f(0, 0, 0);
                glVertex3f(w, h, z);
                glVertex3f(0, 0, 0);
                glVertex3f(w, -h, z);
                glVertex3f(0, 0, 0);
                glVertex3f(-w, -h, z);
                glVertex3f(0, 0, 0);
                glVertex3f(-w, h, z);

                glVertex3f(w, h, z);
                glVertex3f(w, -h, z);

                glVertex3f(-w, h, z);
                glVertex3f(-w, -h, z);

                glVertex3f(-w, h, z);
                glVertex3f(w, h, z);

                glVertex3f(-w, -h, z);
                glVertex3f(w, -h, z);
                glEnd();

                glPopMatrix();
            }
        }

        if (drawGraph) {
            glLineWidth(graph_line_width);
            glColor4f(0.0f, 1.0f, 0.0f, 0.6f);
            glBegin(GL_LINES);

            for (const auto &kf: keyFrames) {
                // covisibility graph
                const vector<shared_ptr<KeyFrame>> neighKFs = kf->getBestCovisibleKFs(10);
                Eigen::Vector3f Ow = kf->getCameraCenter();
                if (!neighKFs.empty()) {
                    for (const auto &neigh_kf: neighKFs) {
                        if (neigh_kf->id < kf->id) continue;
                        Eigen::Vector3f O2 = neigh_kf->getCameraCenter();
                        glVertex3f(Ow[0], Ow[1], Ow[2]);
                        glVertex3f(O2[0], O2[1], O2[2]);
                    }
                }

                // spanning tree
                shared_ptr<KeyFrame> parent = kf->getParent();
                if (parent) {
                    Eigen::Vector3f Op = parent->getCameraCenter();
                    glVertex3f(Ow[0], Ow[1], Ow[2]);
                    glVertex3f(Op[0], Op[1], Op[2]);
                }
            }

            glEnd();
        }
    }

    void MapDrawer::DrawCurrentCamera(pangolin::OpenGlMatrix &Twc) const {
        const float &w = camera_size;
        const float h = 0.75f * w;
        const float z = 0.6f * w;

        glPushMatrix();
        glMultMatrixd(Twc.m);

        glLineWidth(camera_line_width);
        glColor3f(0.0f, 1.0f, 0.0f);
        glBegin(GL_LINES);
        glVertex3f(0, 0, 0);
        glVertex3f(w, h, z);
        glVertex3f(0, 0, 0);
        glVertex3f(w, -h, z);
        glVertex3f(0, 0, 0);
        glVertex3f(-w, -h, z);
        glVertex3f(0, 0, 0);
        glVertex3f(-w, h, z);

        glVertex3f(w, h, z);
        glVertex3f(w, -h, z);

        glVertex3f(-w, h, z);
        glVertex3f(-w, -h, z);

        glVertex3f(-w, h, z);
        glVertex3f(w, h, z);

        glVertex3f(-w, -h, z);
        glVertex3f(w, -h, z);
        glEnd();

        glPopMatrix();
    }

    void MapDrawer::SetCurrentCameraPose(const Pose &Twc) {
        lock_guard<mutex> lock(map_drawer_mutex);
        camera_pose = Twc;
    }

    void MapDrawer::SetReferenceKeyFrame(const std::shared_ptr<KeyFrame> &keyFrame) {
        reference_kf = keyFrame;
    }

    void MapDrawer::getCurrentOpenGLCameraMatrix(pangolin::OpenGlMatrix &matrix) {
        Pose Twc;
        {
            lock_guard<mutex> lock(map_drawer_mutex);
            Twc = camera_pose;
        }

        matrix.m[0] = Twc.R(0, 0);
        matrix.m[1] = Twc.R(1, 0);
        matrix.m[2] = Twc.R(2, 0);
        matrix.m[3] = 0;

        matrix.m[4] = Twc.R(0, 1);
        matrix.m[5] = Twc.R(1, 1);
        matrix.m[6] = Twc.R(2, 1);
        matrix.m[7] = 0;

        matrix.m[8] = Twc.R(0, 2);
        matrix.m[9] = Twc.R(1, 2);
        matrix.m[10] = Twc.R(2, 2);
        matrix.m[11] = 0;

        matrix.m[12] = Twc.t(0);
        matrix.m[13] = Twc.t(1);
        matrix.m[14] = Twc.t(2);
        matrix.m[15] = 1;
    }

} // mono_orb_slam3