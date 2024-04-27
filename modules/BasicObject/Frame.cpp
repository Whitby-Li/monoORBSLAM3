//
// Created by whitby on 8/22/23.
//

#include "Frame.h"
#include "ORB/ORBVocabulary.h"
#include "Sensor/Camera.h"

using namespace std;

namespace mono_orb_slam3 {
    long unsigned int Frame::next_id = 0;
    int Frame::GRID_COLS, Frame::GRID_ROWS;
    bool Frame::grid_size_computed = false;

    Frame::Frame(cv::Mat &grayImg, const double &timeStamp, ORBExtractor *orbExtractor, const Bias &bias)
            : img(grayImg.clone()), timestamp(timeStamp), orb_extractor(orbExtractor) {
        id = next_id++;

        (*orb_extractor)(img, raw_key_points, descriptors);
        num_kps = (int) raw_key_points.size();
        const Camera *camera_ptr = Camera::getCamera();

        for (auto &kp : raw_key_points) {
            kp.size *= camera_ptr->uncertainty(kp.pt);
        }

        camera_ptr->undistortKeyPoints(raw_key_points, key_points);

        map_points = vector<shared_ptr<MapPoint>>(num_kps, nullptr);

        if (!grid_size_computed) {
            int width = img.cols, height = img.rows;
            if (width % GRID_SIZE == 0)
                GRID_COLS = width / GRID_SIZE;
            else GRID_COLS = width / GRID_SIZE + 1;
            if (height % GRID_SIZE == 0)
                GRID_ROWS = height / GRID_SIZE;
            else GRID_ROWS = height / GRID_SIZE + 1;
            grid_size_computed = true;
        }

        grid = vector<vector<vector<size_t>>>(GRID_COLS, vector<vector<size_t >>(GRID_ROWS));

        // assign features to grid for speed up feature matching
        for (int i = 0; i < num_kps; ++i) {
            int x, y;
            if (PosInGrid(key_points[i], x, y)) {
                grid[x][y].push_back(i);
            }
        }

        pre_integrator = make_shared<PreIntegrator>(bias);
        v_w.setZero();
    }

    void Frame::setPose(const Pose &Tcw) {
        T_cw = Tcw;
        O_w = -T_cw.R.transpose() * T_cw.t;

        const ImuCalib *imu_ptr = ImuCalib::getImuCalib();
        T_wb = T_cw.inverse() * imu_ptr->T_cb;
    }

    void Frame::setImuPoseAndVelocity(const Pose &Twb, const Eigen::Vector3f &Vw) {
        T_wb = Twb;
        v_w = Vw;

        const ImuCalib *imu_ptr = ImuCalib::getImuCalib();
        T_cw = imu_ptr->T_cb * T_wb.inverse();
    }

    void Frame::computePreIntegration(const std::vector<ImuData> &imus, double endTime) const {
        double startTime = timestamp + pre_integrator->delta_t;
        if (imus.size() == 1) {
            pre_integrator->IntegrateNewMeasurement(imus[0].w, imus[0].a, endTime - startTime);
            return;
        }

        for (int i = 0, end = (int) imus.size(); i < end; ++i) {
            if (i == 0)
                pre_integrator->IntegrateNewMeasurement(imus[i].w, imus[i].a, imus[i + 1].t - startTime);
            else if (i == end - 1)
                pre_integrator->IntegrateNewMeasurement(imus[i].w, imus[i].a, endTime - imus[i].t);
            else
                pre_integrator->IntegrateNewMeasurement(imus[i].w, imus[i].a, imus[i + 1].t - imus[i].t);
        }
    }

    bool Frame::PosInGrid(const cv::KeyPoint &kp, int &gridX, int &gridY) const {
        int x = cvFloor(kp.pt.x), y = cvFloor(kp.pt.y);
        if (x < 0 || x >= img.cols || y < 0 || y >= img.rows) return false;
        gridX = x / GRID_SIZE, gridY = y / GRID_SIZE;
        return true;
    }

    vector <size_t> Frame::getFeaturesInArea(const float &x, const float &y, const float &r, int minLevel,
                                             int maxLevel) const {
        const int minCellX = max(0, cvFloor(x - r) / GRID_SIZE);
        const int maxCellX = min(GRID_COLS - 1, cvFloor(x + r) / GRID_SIZE);
        if (minCellX > maxCellX) return {};
        const int minCellY = max(0, cvFloor(y - r) / GRID_SIZE);
        const int maxCellY = min(GRID_ROWS - 1, cvFloor(y + r) / GRID_SIZE);
        if (minCellY > maxCellY) return {};

        vector<size_t> indices;
        const bool beCheckLevel = minLevel > 0 || maxLevel >= 0;
        for (int cx = minCellX; cx <= maxCellX; ++cx) {
            for (int cy = minCellY; cy <= maxCellY; ++cy) {
                const vector<size_t> cellIndices = grid[cx][cy];
                if (cellIndices.empty()) continue;

                for (auto idx: cellIndices) {
                    const cv::KeyPoint &kp = key_points[idx];
                    if (beCheckLevel) {
                        if (kp.octave < minLevel) continue;
                        if (maxLevel >= 0 && kp.octave > maxLevel) continue;
                    }

                    if (abs(kp.pt.x - x) <= r && abs(kp.pt.y - y) <= r)
                        indices.push_back(idx);
                }
            }
        }

        return indices;
    }

    bool Frame::isInFrustum(const std::shared_ptr<MapPoint> &mapPoint, float viewCosLimit) const {
        mapPoint->track_in_view = false;
        Eigen::Vector3f Pw = mapPoint->getPos();
        const Camera *camera_ptr = Camera::getCamera();

        const Eigen::Vector3f Pc = T_cw.map(Pw);

        // check positive depth
        if (Pc.z() < 0) return false;

        const cv::Point2f project_p = camera_ptr->project(Pc);
        if (!camera_ptr->isInImage(project_p)) return false;

        // check distance is in the scale invariance region of the map-point
        const float maxDistance = mapPoint->getMaxDistanceInvariance();
        const float minDistance = mapPoint->getMinDistanceInvariance();
        const Eigen::Vector3f OP = Pw - O_w;
        const float dist = OP.norm();

        if (dist < minDistance || dist > maxDistance) return false;

        // check viewing angle
        Eigen::Vector3f Pn = mapPoint->getAverageDirection();
        const float viewCos = OP.dot(Pn) / dist;
        if (viewCos < viewCosLimit) return false;

        // predict scale in image
        const int predictLevel = mapPoint->predictScaleLevel(dist);

        // data used by the tracking
        mapPoint->track_in_view = true;
        mapPoint->track_proj_x = project_p.x;
        mapPoint->track_proj_y = project_p.y;
        mapPoint->track_scale_level = predictLevel;
        mapPoint->track_view_cos = viewCos;

        return true;
    }

    void Frame::computeBow() {
        if (bow_vector.empty() || feature_vector.empty()) {
            vector<cv::Mat> vecDescriptor(num_kps);
            for (int i = 0; i < num_kps; ++i) {
                vecDescriptor[i] = descriptors.row(i);
            }

            const Vocabulary *vocabulary = ORBVocabulary::getORBVocabulary();
            vocabulary->transform(vecDescriptor, bow_vector, feature_vector, 4);
        }
    }
} // mono_orb_slam3