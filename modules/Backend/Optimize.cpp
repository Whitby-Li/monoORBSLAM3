//
// Created by whitby on 8/22/23.
//

#include "Optimize.h"
#include "G2oTypes.h"
#include "Sensor/Camera.h"
#include "Log/Logger.h"

#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/core/robust_kernel_impl.h>

using namespace std;

namespace mono_orb_slam3 {

    void Optimize::initialOptimize(const std::shared_ptr<KeyFrame> &lastKeyFrame,
                                   const std::shared_ptr<KeyFrame> &curKeyFrame) {
        int iteration = 20;

        // 1. set-up optimizer
        g2o::SparseOptimizer optimizer;
        auto *solver = new g2o::OptimizationAlgorithmLevenberg(
                g2o::make_unique<g2o::BlockSolver_6_3>(g2o::make_unique<LinearSolverEigen_6_3>())
        );
        optimizer.setAlgorithm(solver);

        // 2. set vertices and edges
        Pose T1w = lastKeyFrame->getPose();
        auto *vPose1 = new VertexSE3(T1w.R, T1w.t);
        vPose1->setId(0);
        vPose1->setFixed(true);
        optimizer.addVertex(vPose1);

        Pose T2w = curKeyFrame->getPose();
        auto *vPose2 = new VertexSE3(T2w.R, T2w.t);
        vPose2->setId(1);
        optimizer.addVertex(vPose2);

        int mpId = 2;
        vector<shared_ptr<MapPoint>> mapPoints = curKeyFrame->getTrackedMapPoints();
        for (const auto &mp: mapPoints) {
            auto *vPoint = new Vertex3D(mp->getPos());
            vPoint->setId(mpId++);
            vPoint->setMarginalized(true);
            optimizer.addVertex(vPoint);

            unsigned int idx1 = mp->getFeatureId(lastKeyFrame);
            const cv::KeyPoint &kp1 = lastKeyFrame->key_points[idx1];
            auto *eVisual1 = new EdgeSE3Project3D();
            eVisual1->setVertex(0, vPoint);
            eVisual1->setVertex(1, vPose1);
            eVisual1->setMeasurement({kp1.pt.x, kp1.pt.y});
            eVisual1->setInformation(Eigen::Matrix2d::Identity());

            auto *rk1 = new g2o::RobustKernelHuber;
            eVisual1->setRobustKernel(rk1);
            rk1->setDelta(5.991);

            optimizer.addEdge(eVisual1);

            unsigned int idx2 = mp->getFeatureId(curKeyFrame);
            const cv::KeyPoint &kp2 = curKeyFrame->key_points[idx2];
            auto *eVisual2 = new EdgeSE3Project3D();
            eVisual2->setVertex(0, vPoint);
            eVisual2->setVertex(1, vPose2);
            eVisual2->setMeasurement({kp2.pt.x, kp2.pt.y});
            eVisual2->setInformation(Eigen::Matrix2d::Identity());

            auto *rk2 = new g2o::RobustKernelHuber;
            eVisual2->setRobustKernel(rk2);
            rk2->setDelta(5.991);

            optimizer.addEdge(eVisual2);
        }

        // 3. optimize
        optimizer.initializeOptimization();
        optimizer.optimize(iteration);

        // 4. recover
        auto pose2 = vPose2->estimate();
        curKeyFrame->setPose(pose2.rotation().toRotationMatrix().cast<float>(), pose2.translation().cast<float>());

        mpId = 2;
        for (auto &mp: mapPoints) {
            auto *vPoint = dynamic_cast<Vertex3D *>(optimizer.vertex(mpId++));
            mp->setPos(vPoint->estimate().cast<float>());
            mp->update();
        }
    }

    void Optimize::inertialOptimize(Map *pointMap, Eigen::Matrix3d &Rwg, double &scale, float prioriG, float prioriA,
                                    bool beFirst) {
        int iteration = 200;
        const vector<shared_ptr<KeyFrame>> keyFrames = pointMap->getAllKeyFrames();

        // 1. setup optimizer
        g2o::SparseOptimizer optimizer;
        auto *solver = new g2o::OptimizationAlgorithmLevenberg(
                g2o::make_unique<g2o::BlockSolverX>(g2o::make_unique<LinearSolverEigenX>())
        );

        if (prioriG != 0.f)
            solver->setUserLambdaInit(1e3);
        optimizer.setAlgorithm(solver);

        // 2. set vertices and edges
        // bias, gravity and scale
        int numKF = (int) keyFrames.size();
        const Bias initialBias = ImuCalib::getImuCalib()->initial_bias;
        const shared_ptr<PreIntegrator> &backPreIntegrator = keyFrames.back()->pre_integrator;
        auto *vGyroBias = new Vertex3D(initialBias.bg);
        vGyroBias->setId(numKF + 1);
        optimizer.addVertex(vGyroBias);

        auto *vAccBias = new Vertex3D(initialBias.ba);
        vAccBias->setId(numKF + 2);
        optimizer.addVertex(vAccBias);

        auto *ePrioriGyro = new EdgePriori3D(initialBias.bg);
        ePrioriGyro->setVertex(0, vGyroBias);
        ePrioriGyro->setInformation(prioriG * Eigen::Matrix3d::Identity());
        optimizer.addEdge(ePrioriGyro);

        auto *ePrioriAcc = new EdgePriori3D(initialBias.ba);
        ePrioriAcc->setVertex(0, vAccBias);
        ePrioriAcc->setInformation(prioriA * Eigen::Matrix3d::Identity());
        optimizer.addEdge(ePrioriAcc);

        auto *vGravityDir = new VertexGravity(Rwg);
        vGravityDir->setId(numKF + 3);
        optimizer.addVertex(vGravityDir);

        g2o::BaseVertex<1, double> *vScale;
        if (beFirst) {
            vScale = new VertexScale(scale);
            vScale->setId(numKF + 4);
            optimizer.addVertex(vScale);
        }

        // velocity vertices and inertial edge
        vector<Vertex3D *> veloVertices;
        veloVertices.reserve(numKF);
        shared_ptr<KeyFrame> last_kf;
        CameraImuPose lastPose;
        for (int i = 0; i < numKF; ++i) {
            const shared_ptr<KeyFrame> &kf = keyFrames[i];
            CameraImuPose curPose(kf->getPose(), kf->getImuPose());

            auto *vVelo = new Vertex3D(kf->getVelocity());
            vVelo->setId(i);
            optimizer.addVertex(vVelo);
            veloVertices.push_back(vVelo);

            if (last_kf) {
                if (beFirst) {
                    auto *eInertial = new EdgeInertialGS(lastPose, curPose, last_kf->pre_integrator);
                    eInertial->setVertex(0, veloVertices[i - 1]);
                    eInertial->setVertex(1, vGyroBias);
                    eInertial->setVertex(2, vAccBias);
                    eInertial->setVertex(3, veloVertices[i]);
                    eInertial->setVertex(4, vGravityDir);
                    eInertial->setVertex(5, vScale);
                    optimizer.addEdge(eInertial);
                } else {
                    auto *eInertial = new EdgeInertialG(lastPose, curPose, last_kf->pre_integrator);
                    eInertial->setVertex(0, veloVertices[i - 1]);
                    eInertial->setVertex(1, vGyroBias);
                    eInertial->setVertex(2, vAccBias);
                    eInertial->setVertex(3, veloVertices[i]);
                    eInertial->setVertex(4, vGravityDir);
                    optimizer.addEdge(eInertial);
                }
            }

            last_kf = kf;
            lastPose = curPose;
        }

        // 3. optimize
        optimizer.setVerbose(false);
        optimizer.initializeOptimization();
        optimizer.optimize(iteration);

        // 4. recover
        Rwg = vGravityDir->estimate().R_wg;
        if (beFirst) scale = vScale->estimate();
        Eigen::Vector3f bg = vGyroBias->estimate().cast<float>();
        Eigen::Vector3f ba = vAccBias->estimate().cast<float>();
        Bias newBias(bg, ba);

        lock_guard<mutex> lock(pointMap->map_update_mutex);
        for (int i = 0; i < numKF; ++i) {
            const shared_ptr<KeyFrame> &kf = keyFrames[i];

            kf->setVelocity(veloVertices[i]->estimate().cast<float>());
            kf->setImuBias(newBias);

            if (i > 0) {
                kf->bias_priori = newBias;
                kf->setPrioriInformation(keyFrames[i - 1]->pre_integrator->C);
            }
        }
    }

    void Optimize::gravityOptimize(Map *pointMap, Eigen::Matrix3d &Rwg) {
        int iteration = 20;
        const vector<shared_ptr<KeyFrame>> keyFrames = pointMap->getAllKeyFrames();
        const int numKF = (int) keyFrames.size();

        // 1. setup optimizer
        g2o::SparseOptimizer optimizer;
        auto *solver = new g2o::OptimizationAlgorithmLevenberg(
                g2o::make_unique<BlockSolver_6_2>(g2o::make_unique<LinearSolverEigen_6_2>())
        );
        optimizer.setAlgorithm(solver);

        // 2. set vertex and edges
        auto *vGravityDir = new VertexGravity(Rwg);
        vGravityDir->setId(0);
        optimizer.addVertex(vGravityDir);

        for (int i = 0; i < numKF - 1; ++i) {
            auto *eGravity = new EdgeGravity(keyFrames[i], keyFrames[i + 1]);
            eGravity->setVertex(0, vGravityDir);
            optimizer.addEdge(eGravity);
        }

        // 3. optimize
        optimizer.setVerbose(false);
        optimizer.initializeOptimization();
        optimizer.optimize(iteration);

        // 4. recover
        Rwg = vGravityDir->estimate().R_wg;
    }

    void Optimize::fullInertialOptimize(Map *pointMap, int iteration, bool beInit, bool fixedMP, float prioriG,
                                        float prioriA) {
        const vector<shared_ptr<KeyFrame>> keyFrames = pointMap->getAllKeyFrames();
        const vector<shared_ptr<MapPoint>> mapPoints = pointMap->getAllMapPoints();

        // 1. setup optimizer
        g2o::SparseOptimizer optimizer;
        auto *solver = new g2o::OptimizationAlgorithmLevenberg(
                g2o::make_unique<g2o::BlockSolverX>(g2o::make_unique<LinearSolverEigenX>())
        );

        solver->setUserLambdaInit(1e-5);
        optimizer.setAlgorithm(solver);
        optimizer.setVerbose(false);

        // 2. set vertices and edges
        int maxKFId = keyFrames.back()->id;
        int numKF = (int) keyFrames.size();
        for (const auto &kf: keyFrames) {
            auto *vPose = new VertexPose({kf->getPose(), kf->getImuPose()});
            vPose->setId(kf->id);
            vPose->setFixed(kf->id == 0);
            optimizer.addVertex(vPose);

            auto *vVelo = new Vertex3D(kf->getVelocity());
            vVelo->setId(maxKFId + 3 * kf->id + 1);
            vVelo->setFixed(kf->id == 0);
            optimizer.addVertex(vVelo);

            if (!beInit) {
                const Bias bias = kf->pre_integrator->updated_bias;
                auto *vGyroBias = new Vertex3D(bias.bg);
                vGyroBias->setId(maxKFId + 3 * kf->id + 2);
                vGyroBias->setFixed(kf->id == 0);
                optimizer.addVertex(vGyroBias);

                auto *vAccBias = new Vertex3D(bias.ba);
                vAccBias->setId(maxKFId + 3 * kf->id + 3);
                vAccBias->setFixed(kf->id == 0);
                optimizer.addVertex(vAccBias);
            }
        }

        if (beInit) {
            const Bias bias = keyFrames.back()->pre_integrator->updated_bias;
            auto *vGyroBias = new Vertex3D(bias.bg);
            vGyroBias->setId(4 * maxKFId + 2);
            optimizer.addVertex(vGyroBias);

            auto *vAccBias = new Vertex3D(bias.ba);
            vAccBias->setId(4 * maxKFId + 3);
            optimizer.addVertex(vAccBias);
        }

        // imu links
        g2o::HyperGraph::Vertex *vPose1, *vVelo1, *vGyroBias1, *vAccBias1;
        g2o::HyperGraph::Vertex *vPose2, *vVelo2;
        for (int i = 0; i < numKF; ++i) {
            int kfId = keyFrames[i]->id;
            if (i == 0) {
                vPose1 = optimizer.vertex(kfId);
                vVelo1 = optimizer.vertex(maxKFId + 3 * kfId + 1);
                if (!beInit) {
                    vGyroBias1 = optimizer.vertex(maxKFId + 3 * kfId + 2);
                    vAccBias1 = optimizer.vertex(maxKFId + 3 * kfId + 3);
                } else {
                    vGyroBias1 = optimizer.vertex(4 * maxKFId + 2);
                    vAccBias1 = optimizer.vertex(4 * maxKFId + 3);
                }
            } else {
                vPose2 = optimizer.vertex(kfId);
                vVelo2 = optimizer.vertex(maxKFId + 3 * kfId + 1);

                shared_ptr<PreIntegrator> preIntegrator = keyFrames[i - 1]->pre_integrator;

                auto *eInertial = new EdgeInertial(preIntegrator);
                eInertial->setVertex(0, vPose1);
                eInertial->setVertex(1, vVelo1);
                eInertial->setVertex(2, vGyroBias1);
                eInertial->setVertex(3, vAccBias1);
                eInertial->setVertex(4, vPose2);
                eInertial->setVertex(5, vVelo2);

                auto *rk = new g2o::RobustKernelHuber;
                eInertial->setRobustKernel(rk);
                rk->setDelta(4.1134);

                optimizer.addEdge(eInertial);

                vPose1 = vPose2;
                vVelo1 = vVelo2;

                if (!beInit) {
                    g2o::HyperGraph::Vertex *vGyroBias2 = optimizer.vertex(maxKFId + 3 * kfId + 2);
                    g2o::HyperGraph::Vertex *vAccBias2 = optimizer.vertex(maxKFId + 3 * kfId + 3);

                    auto *eGyroWalk = new EdgeBiasWalk();
                    eGyroWalk->setVertex(0, vGyroBias1);
                    eGyroWalk->setVertex(1, vGyroBias2);
                    eGyroWalk->setInformation(preIntegrator->C.block<3, 3>(9, 9).cast<double>().inverse());
                    optimizer.addEdge(eGyroWalk);

                    auto *eAccWalk = new EdgeBiasWalk();
                    eAccWalk->setVertex(0, vAccBias1);
                    eAccWalk->setVertex(1, vAccBias2);
                    eAccWalk->setInformation(preIntegrator->C.block<3, 3>(12, 12).cast<double>().inverse());
                    optimizer.addEdge(eAccWalk);

                    vGyroBias1 = vGyroBias2;
                    vAccBias1 = vAccBias2;
                }
            }
        }

        if (beInit) {
            const Bias bias = keyFrames.back()->pre_integrator->updated_bias;
            auto *ePrioriGyro = new EdgePriori3D(bias.bg);
            ePrioriGyro->setVertex(0, vGyroBias1);
            ePrioriGyro->setInformation(prioriG * Eigen::Matrix3d::Identity());
            optimizer.addEdge(ePrioriGyro);

            auto *ePrioriAcc = new EdgePriori3D(bias.ba);
            ePrioriAcc->setVertex(0, vAccBias1);
            ePrioriAcc->setInformation(prioriA * Eigen::Matrix3d::Identity());
            optimizer.addEdge(ePrioriAcc);
        }

        const float thHuber = sqrtf(5.991);
        const int mpIdStart = maxKFId * 5;

        for (const auto &mp: mapPoints) {
            auto *vPoint = new Vertex3D(mp->getPos());
            vPoint->setId(mpIdStart + mp->id + 1);
            vPoint->setMarginalized(true);
            optimizer.addVertex(vPoint);

            const map<shared_ptr<KeyFrame>, size_t> observations = mp->getObservations();
            for (const auto &obsPair: observations) {
                const shared_ptr<KeyFrame> &kf = obsPair.first;

                if (kf->id > maxKFId || kf->isBad()) continue;

                const cv::KeyPoint &kp = kf->key_points[obsPair.second];

                auto *eVisual = new EdgeMono(kp.pt);
                eVisual->setVertex(0, vPoint);
                eVisual->setVertex(1, optimizer.vertex(kf->id));
                eVisual->setInformation(Eigen::Matrix2d::Identity() * ORBExtractor::getInvSquareSigma(kp.octave));

                auto *rk = new g2o::RobustKernelHuber;
                eVisual->setRobustKernel(rk);
                rk->setDelta(thHuber);

                optimizer.addEdge(eVisual);
            }
        }

        // 3. optimize
        optimizer.initializeOptimization();
        optimizer.optimize(iteration);

        // 4. recover
        lock_guard<mutex> lock(pointMap->map_update_mutex);
        // bias
        Bias newBias;
        if (beInit) {
            newBias.bg = dynamic_cast<Vertex3D *>(optimizer.vertex(4 * maxKFId + 2))->estimate().cast<float>();
            newBias.ba = dynamic_cast<Vertex3D *>(optimizer.vertex(4 * maxKFId + 3))->estimate().cast<float>();
        }

        // keyframes
        Bias lastBias;
        for (int i = 0; i < numKF; ++i) {
            const shared_ptr<KeyFrame> &kf = keyFrames[i];
            int curKFid = kf->id;

            auto pose = dynamic_cast<VertexPose *>(optimizer.vertex(curKFid))->estimate();
            kf->setPose(pose.R_cw.cast<float>(), pose.t_cw.cast<float>());

            Eigen::Vector3d velo = dynamic_cast<Vertex3D *>(optimizer.vertex(maxKFId + 3 * curKFid + 1))->estimate();
            kf->setVelocity(velo.cast<float>());

            if (!beInit) {
                newBias.bg = dynamic_cast<Vertex3D *>(optimizer.vertex(
                        maxKFId + 3 * curKFid + 2))->estimate().cast<float>();
                newBias.ba = dynamic_cast<Vertex3D *>(optimizer.vertex(
                        maxKFId + 3 * curKFid + 3))->estimate().cast<float>();
            }

            kf->setImuBias(newBias);
            if (i > 0) {
                kf->bias_priori = lastBias;
                kf->setPrioriInformation(keyFrames[i - 1]->pre_integrator->C);
            }
            lastBias = newBias;
        }

        // map-points
        for (const auto &mp: mapPoints) {
            auto *vPoint = dynamic_cast<Vertex3D *>(optimizer.vertex(mpIdStart + mp->id + 1));
            mp->setPos(vPoint->estimate().cast<float>());
            mp->update();
        }
    }

    int Optimize::poseOptimize(const std::shared_ptr<Frame> &frame) {
        // 1. set optimizer
        g2o::SparseOptimizer optimizer;
        auto *solver = new g2o::OptimizationAlgorithmLevenberg(
                g2o::make_unique<g2o::BlockSolver_6_3>(g2o::make_unique<LinearSolverEigen_6_3>())
        );
        optimizer.setAlgorithm(solver);

        int numInitialCorrespondences = 0;

        // 2. set vertex and edges
        const Pose Tcw = frame->T_cw;
        auto *vPose = new VertexSE3(Tcw.R, Tcw.t);
        vPose->setId(0);
        optimizer.addVertex(vPose);

        // set map-point vertices
        const int n = frame->num_kps;
        vector<EdgeSE3Project3DOnlyPose *> vecEdge;
        vector<size_t> vecIndices;
        vecEdge.reserve(n), vecIndices.reserve(n);

        const float deltaMono = sqrtf(5.991);

        {
            lock_guard<mutex> lock(MapPoint::global_mutex);
            for (int i = 0; i < n; ++i) {
                const shared_ptr<MapPoint> &mp = frame->map_points[i];
                if (mp != nullptr && !mp->isBad()) {
                    numInitialCorrespondences++;
                    const cv::KeyPoint &kp = frame->key_points[i];

                    auto *eVisual = new EdgeSE3Project3DOnlyPose(mp->getPos());
                    eVisual->setVertex(0, vPose);
                    eVisual->setMeasurement({kp.pt.x, kp.pt.y});
                    const float invSigma2 = ORBExtractor::getInvSquareSigma(kp.octave);
                    eVisual->setInformation(invSigma2 * Eigen::Matrix2d::Identity());

                    auto *rk = new g2o::RobustKernelHuber;
                    eVisual->setRobustKernel(rk);
                    rk->setDelta(deltaMono);

                    optimizer.addEdge(eVisual);
                    vecEdge.push_back(eVisual);
                    vecIndices.push_back(i);
                }
            }
        }

        if (numInitialCorrespondences < 3) return 0;

        // 3. optimize
        // we perform 4 optimizations, after each optimization we classify observation as inlier/outlier
        // at the next optimization, outliers are not included, but they can be optimized again at the end.
        const int iters[4] = {10, 10, 10, 10};

        vector<bool> vecBeInlier(numInitialCorrespondences, true);
        for (auto iter: iters) {
            vPose->setEstimate({Tcw.R.cast<double>(), Tcw.t.cast<double>()});

            optimizer.initializeOptimization(0);
            optimizer.optimize(iter);

            for (unsigned i = 0; i < numInitialCorrespondences; ++i) {
                auto *e = vecEdge[i];

                if (!vecBeInlier[i]) e->computeError();

                const double chi2 = e->chi2();
                if (chi2 > 5.991) {
                    vecBeInlier[i] = false;
                    e->setLevel(1);
                } else {
                    vecBeInlier[i] = true;
                    e->setLevel(0);
                }

                if (iter == 2)
                    e->setRobustKernel(nullptr);
            }
        }

        // 4. recover
        auto optimizedPose = vPose->estimate();
        frame->setPose(
                {optimizedPose.rotation().toRotationMatrix().cast<float>(), optimizedPose.translation().cast<float>()});

        // discard outlier map-points
        int numInlier = 0;
        for (int i = 0; i < numInitialCorrespondences; ++i) {
            if (!vecBeInlier[i]) {
                shared_ptr<MapPoint> &mp = frame->map_points[vecIndices[i]];
                mp->track_in_view = false;
                mp->last_frame_seen = frame->id;
                mp = nullptr;
            } else {
                numInlier++;
            }
        }

        return numInlier;
    }

    void Optimize::poseInertialOptimize(const std::shared_ptr<KeyFrame> &lastKF, const std::shared_ptr<Frame> &frame) {
        // 1. set optimizer
        g2o::SparseOptimizer optimizer;
        auto *solver = new g2o::OptimizationAlgorithmLevenberg(
                g2o::make_unique<g2o::BlockSolverX>(g2o::make_unique<LinearSolverEigenX>())
        );
        optimizer.setVerbose(false);
        optimizer.setAlgorithm(solver);

        // 2. add vertices
        auto *vVelo1 = new Vertex3D(lastKF->getVelocity());
        vVelo1->setId(0);
        optimizer.addVertex(vVelo1);

        const shared_ptr<PreIntegrator> &preIntegrator = lastKF->pre_integrator;
        const Bias &bias1 = preIntegrator->updated_bias;
        auto *vGyroBias1 = new Vertex3D(bias1.bg);
        vGyroBias1->setId(1);
        optimizer.addVertex(vGyroBias1);

        auto *vAccBias1 = new Vertex3D(bias1.ba);
        vAccBias1->setId(2);
        optimizer.addVertex(vAccBias1);

        auto *vVelo2 = new Vertex3D(frame->v_w);
        vVelo2->setId(3);
        optimizer.addVertex(vVelo2);

        // 3. add edges
        auto *eInertialOnly = new EdgeInertialOnly(lastKF->getImuPose(), frame->T_wb, preIntegrator);
        eInertialOnly->setVertex(0, vVelo1);
        eInertialOnly->setVertex(1, vGyroBias1);
        eInertialOnly->setVertex(2, vAccBias1);
        eInertialOnly->setVertex(3, vVelo2);

        auto *eVeloPriori = new EdgePriori3D(lastKF->velo_priori);
        eVeloPriori->setVertex(0, vVelo1);
        eVeloPriori->setInformation(lastKF->velo_info.cast<double>());
        optimizer.addEdge(eVeloPriori);

        auto *eGyroBiasPriori = new EdgePriori3D(lastKF->bias_priori.bg);
        eGyroBiasPriori->setVertex(0, vGyroBias1);
        eGyroBiasPriori->setInformation(lastKF->gyro_info.cast<double>());
        optimizer.addEdge(eGyroBiasPriori);

        auto *eAccBiasPriori = new EdgePriori3D(lastKF->bias_priori.ba);
        eAccBiasPriori->setVertex(0, vAccBias1);
        eAccBiasPriori->setInformation(lastKF->acc_info.cast<double>());
        optimizer.addEdge(eAccBiasPriori);

        // 4. optimize
        optimizer.initializeOptimization();
        optimizer.optimize(10);

        // 5. recover
        Bias newBias(vGyroBias1->estimate().cast<float>(), vAccBias1->estimate().cast<float>());
        lastKF->setImuBias(newBias);
        lastKF->setVelocity(vVelo1->estimate().cast<float>());

        frame->pre_integrator->setNewBias(newBias);
        frame->v_w = vVelo2->estimate().cast<float>();
    }

    int Optimize::poseFullOptimize(const std::shared_ptr<KeyFrame> &lastKF, const std::shared_ptr<Frame> &frame) {
        // 1. set optimizer
        g2o::SparseOptimizer optimizer;
        auto *solver = new g2o::OptimizationAlgorithmLevenberg(
                g2o::make_unique<g2o::BlockSolverX>(g2o::make_unique<LinearSolverEigenX>())
        );
        optimizer.setVerbose(false);
        optimizer.setAlgorithm(solver);

        int numCorrespondences = 0;

        // 2. set vertices and edges
        // last keyframe vertices
        auto *vPose1 = new VertexPose({lastKF->getPose(), lastKF->getImuPose()});
        vPose1->setId(0);
        vPose1->setFixed(true);
        optimizer.addVertex(vPose1);

        auto *vVelo1 = new Vertex3D(lastKF->getVelocity());
        vVelo1->setId(1);
        optimizer.addVertex(vVelo1);

        const shared_ptr<PreIntegrator> &preIntegrator = lastKF->pre_integrator;
        const Bias &bias1 = preIntegrator->updated_bias;
        auto *vGyroBias1 = new Vertex3D(bias1.bg);
        vGyroBias1->setId(2);
        optimizer.addVertex(vGyroBias1);

        auto *vAccBias1 = new Vertex3D(bias1.ba);
        vAccBias1->setId(3);
        optimizer.addVertex(vAccBias1);

        // frame vertices
        CameraImuPose initialPose(frame->T_cw, frame->T_wb);
        auto *vPose2 = new VertexPose(initialPose);

        vPose2->setId(4);
        optimizer.addVertex(vPose2);

        auto *vVelo2 = new Vertex3D(frame->v_w);
        vVelo2->setId(5);
        optimizer.addVertex(vVelo2);

        auto *eInertial = new EdgeInertial(preIntegrator);
        eInertial->setVertex(0, vPose1);
        eInertial->setVertex(1, vVelo1);
        eInertial->setVertex(2, vGyroBias1);
        eInertial->setVertex(3, vAccBias1);
        eInertial->setVertex(4, vPose2);
        eInertial->setVertex(5, vVelo2);
        optimizer.addEdge(eInertial);

        auto *eVeloPriori = new EdgePriori3D(lastKF->velo_priori);
        eVeloPriori->setVertex(0, vVelo1);
        eVeloPriori->setInformation(lastKF->velo_info.cast<double>());
        optimizer.addEdge(eVeloPriori);

        auto *eGyroBiasPriori = new EdgePriori3D(lastKF->bias_priori.bg);
        eGyroBiasPriori->setVertex(0, vGyroBias1);
        eGyroBiasPriori->setInformation(lastKF->gyro_info.cast<double>());
        optimizer.addEdge(eGyroBiasPriori);

        auto *eAccBiasPriori = new EdgePriori3D(lastKF->bias_priori.ba);
        eAccBiasPriori->setVertex(0, vAccBias1);
        eAccBiasPriori->setInformation(lastKF->acc_info.cast<double>());
        optimizer.addEdge(eAccBiasPriori);

        // map-point vertices and edges
        const int n = frame->num_kps;
        vector<EdgeMonoOnlyPose *> vecEdgeMonoPose;
        vector<int> vecIndices;
        vecEdgeMonoPose.reserve(n / 2);
        vecIndices.reserve(n / 2);
        const float thHuber = sqrtf(5.991);

        {
            lock_guard<mutex> lock(MapPoint::global_mutex);
            for (int i = 0; i < n; ++i) {
                const shared_ptr<MapPoint> mp = frame->map_points[i];
                if (mp && !mp->isBad()) {
                    numCorrespondences++;
                    const cv::KeyPoint &kp = frame->key_points[i];

                    auto *eVisual = new EdgeMonoOnlyPose(mp->getPos());
                    eVisual->setVertex(0, vPose2);
                    eVisual->setMeasurement({kp.pt.x, kp.pt.y});

                    const float invSigma2 = ORBExtractor::getInvSquareSigma(kp.octave);
                    eVisual->setInformation(Eigen::Matrix2d::Identity() * invSigma2);

                    auto *rk = new g2o::RobustKernelHuber;
                    eVisual->setRobustKernel(rk);
                    rk->setDelta(thHuber);

                    optimizer.addEdge(eVisual);
                    vecEdgeMonoPose.push_back(eVisual);
                    vecIndices.push_back(i);
                }
            }
        }

        // we perform 4 optimizations, after each optimization we classify observation as inlier/outlier
        // at the next optimization, outliers are not included, but at the end they can be classified as inliers again
        const int iters[4] = {10, 10, 10, 10};
        vector<bool> vecBeInlier(numCorrespondences, true);

        for (auto iter : iters) {
            vPose2->setEstimate(initialPose);

            optimizer.initializeOptimization(0);
            optimizer.optimize(iter);

            for (unsigned i = 0; i < numCorrespondences; ++i) {
                auto *e = vecEdgeMonoPose[i];

                if (!vecBeInlier[i]) e->computeError();

                const double chi2 = e->chi2();
                if (chi2 > 5.991) {
                    vecBeInlier[i] = false;
                    e->setLevel(1);
                } else {
                    vecBeInlier[i] = true;
                    e->setLevel(0);
                }

                if (iter == 2)
                    e->setRobustKernel(nullptr);
            }
        }


        // 4. recover
        const CameraImuPose optimizePose = vPose2->estimate();
        Bias newBias(vGyroBias1->estimate().cast<float>(), vAccBias1->estimate().cast<float>());

        frame->setPose({optimizePose.R_cw.cast<float>(), optimizePose.t_cw.cast<float>()});
        frame->v_w = vVelo2->estimate().cast<float>();
        frame->pre_integrator->setNewBias(newBias);

        // discard outlier map-points
        int numInlier = 0;
        for (int i = 0; i < numCorrespondences; ++i) {
            if (!vecBeInlier[i]) {
                shared_ptr<MapPoint> &mp = frame->map_points[vecIndices[i]];
                mp->track_in_view = false;
                mp->last_frame_seen = frame->id;
                mp = nullptr;
            } else {
                numInlier++;
            }
        }

        return numInlier;
    }

    void Optimize::localBundleAdjustment(const std::shared_ptr<KeyFrame> &curKeyFrame, Map *pointMap, bool *stopFlag) {
        // 1. extract local map
        // local keyframes: first breath search from current keyframe
        list<shared_ptr<KeyFrame>> localKeyFrames;
        localKeyFrames.push_back(curKeyFrame);
        unsigned int curKFId = curKeyFrame->id;
        curKeyFrame->BA_local_for_kf = curKFId;

        const vector<shared_ptr<KeyFrame>> neighKeyFrames = curKeyFrame->getConnectedKFs();
        for (const auto &kf: neighKeyFrames) {
            kf->BA_local_for_kf = curKFId;
            if (!kf->isBad()) {
                localKeyFrames.push_back(kf);
            }
        }

        // local map-points seen in local keyframes
        list<shared_ptr<MapPoint>> localMapPoints;
        for (const auto &kf: localKeyFrames) {
            vector<shared_ptr<MapPoint>> mapPoints = kf->getMapPoints();
            for (const auto &mp: mapPoints) {
                if (mp && !mp->isBad() && mp->BA_local_for_kf != curKFId) {
                    localMapPoints.push_back(mp);
                    mp->BA_local_for_kf = curKFId;
                }
            }
        }

        // fixed keyframe (see local map-points but not in local-keyframes)
        list<shared_ptr<KeyFrame>> fixedKeyFrames;
        for (const auto &mp: localMapPoints) {
            map<shared_ptr<KeyFrame>, size_t> observations = mp->getObservations();
            for (const auto &obsPair: observations) {
                const shared_ptr<KeyFrame> &kf = obsPair.first;
                if (!kf->isBad() && kf->BA_local_for_kf != curKFId && kf->BA_fixed_for_kf != curKFId) {
                    kf->BA_fixed_for_kf = curKFId;
                    fixedKeyFrames.push_back(kf);
                }
                if (kf->isBad()) cerr << "map-point observe bad kf" << endl;
            }
        }

        mapper_logger << titles[0] << "there are " << localKeyFrames.size() << " local keyframes, "
                      << fixedKeyFrames.size() << " fixed keyframes, " << localMapPoints.size()
                      << " local map-points\n";

        // 2. set optimizer
        g2o::SparseOptimizer optimizer;
        auto *solver = new g2o::OptimizationAlgorithmLevenberg(
                g2o::make_unique<g2o::BlockSolver_6_3>(g2o::make_unique<LinearSolverEigen_6_3>())
        );

        optimizer.setAlgorithm(solver);

        if (stopFlag != nullptr)
            optimizer.setForceStopFlag(stopFlag);

        // 3. set vertices and edges
        // local keyframe vertices
        unsigned long maxKFId = 0;
        for (const auto &kf: localKeyFrames) {
            const Pose Tcw = kf->getPose();
            auto *vPose = new VertexSE3(Tcw.R, Tcw.t);
            vPose->setId(kf->id);
            vPose->setFixed(kf->id == 0);
            optimizer.addVertex(vPose);

            maxKFId = max(maxKFId, kf->id);
        }

        // fixed keyframe vertices
        for (const auto &kf: fixedKeyFrames) {
            const Pose Tcw = kf->getPose();
            auto *vPose = new VertexSE3(Tcw.R, Tcw.t);
            vPose->setId(kf->id);
            vPose->setFixed(true);
            optimizer.addVertex(vPose);

            maxKFId = max(maxKFId, kf->id);
        }

        unsigned int expectSize = localKeyFrames.size() * localMapPoints.size();
        vector<Vertex3D *> vecPoints;
        vecPoints.reserve(localMapPoints.size());

        vector<EdgeSE3Project3D *> edges;
        edges.reserve(expectSize);

        vector<pair<shared_ptr<KeyFrame>, shared_ptr<MapPoint>>> vecEdgeObs;
        vecEdgeObs.reserve(expectSize);

        const float thHuberMono = sqrtf(5.991);
        int numEdge = 0;
        int mpId = maxKFId + 1;
        for (const auto &mp: localMapPoints) {
            auto *vPoint = new Vertex3D(mp->getPos());
            vPoint->setId(mpId++);
            vPoint->setMarginalized(true);
            optimizer.addVertex(vPoint);
            vecPoints.push_back(vPoint);

            const map<shared_ptr<KeyFrame>, size_t> observation = mp->getObservations();
            for (const auto &obsPair: observation) {
                const shared_ptr<KeyFrame> &kf = obsPair.first;
                if (kf->isBad()) continue;
                const cv::KeyPoint &kp = kf->key_points[obsPair.second];

                auto *eVisual = new EdgeSE3Project3D();
                eVisual->setVertex(0, vPoint);
                eVisual->setVertex(1, optimizer.vertex(kf->id));
                eVisual->setMeasurement({kp.pt.x, kp.pt.y});
                const float invSigma2 = ORBExtractor::getInvSquareSigma(kp.octave);
                eVisual->setInformation(invSigma2 * Eigen::Matrix2d::Identity());

                auto *rk = new g2o::RobustKernelHuber;
                eVisual->setRobustKernel(rk);
                rk->setDelta(thHuberMono);

                optimizer.addEdge(eVisual);
                edges.push_back(eVisual);
                vecEdgeObs.emplace_back(kf, mp);
                numEdge++;
            }
        }

        // 4. optimize
        optimizer.initializeOptimization();
        optimizer.optimize(5);

        bool doMore = true;
        if (stopFlag != nullptr && *stopFlag) doMore = false;

        if (doMore) {
            // 5. check inlier observations
            for (int i = 0; i < numEdge; ++i) {
                auto *e = edges[i];
                if (e->chi2() > 5.991) {
                    e->setLevel(1);
                }

                e->setRobustKernel(nullptr);
            }

            // 6. optimize again without outliers
            optimizer.initializeOptimization(0);
            optimizer.optimize(10);
        }

        // 7. check inlier observations, discard outliers
        vector<pair<shared_ptr<KeyFrame>, shared_ptr<MapPoint>>> vecToErase;
        vecToErase.reserve(numEdge / 2);
        for (int i = 0; i < numEdge; ++i) {
            auto *e = edges[i];
            if (e->chi2() > 5.991) {
                vecToErase.push_back(vecEdgeObs[i]);
            }
        }

        // get map update mutex
        lock_guard<mutex> lock(pointMap->map_update_mutex);

        for (const auto &erasePair: vecToErase) {
            const shared_ptr<KeyFrame> &kf = erasePair.first;
            const shared_ptr<MapPoint> &mp = erasePair.second;

            if (mp->isBad()) continue;
            kf->eraseMapPoint(mp->getFeatureId(kf));
            mp->eraseObservation(kf);
        }

        // 8. recover
        for (const auto &kf: localKeyFrames) {
            auto pose = dynamic_cast<VertexSE3 *>(optimizer.vertex(kf->id))->estimate();
            const Eigen::Matrix4f optimizeTcw = pose.to_homogeneous_matrix().cast<float>();
            kf->setPose(optimizeTcw.block<3, 3>(0, 0), optimizeTcw.block<3, 1>(0, 3));
        }

        mpId = maxKFId + 1;
        for (const auto &mp: localMapPoints) {
            auto *vPoint = dynamic_cast<Vertex3D *>(optimizer.vertex(mpId++));
            if (!mp->isBad()) {
                mp->setPos(vPoint->estimate().cast<float>());
                mp->update();
            }
        }
    }

    void Optimize::localInertialBundleAdjustment(const std::shared_ptr<KeyFrame> &keyFrame, Map *pointMap,
                                                 bool beLarge) {
        mapper_logger << "localInertialBundleAdjustment\n";

        // 1. get optimizable keyframes, fixed keyframes and map-points
        int maxOpt = 10, iterations = 10;
        if (beLarge) {
            maxOpt = 20;
            iterations = 10;
        }

        const int numOpt = min(pointMap->getNumKeyFrames(), maxOpt);
        const unsigned int maxKFId = keyFrame->id;

        vector<shared_ptr<KeyFrame>> optimizeKeyFrames = pointMap->getRecentKeyFrames(numOpt);

        mapper_logger << "Initial Pose\n";
        for (const auto &kf : optimizeKeyFrames) {
            mapper_logger << titles[1] << "keyframe id " << kf->id << ", frame_id " << kf->frame_id << "\n";
            mapper_logger << titles[1] << " - pose: " << kf->getPose() << "\n";
            mapper_logger << titles[1] << " - velo: " << kf->getVelocity() << "\n";
            mapper_logger << titles[1] << " - bias: " << kf->pre_integrator->updated_bias << "\n";
        }

        // 2. set up optimizer
        g2o::SparseOptimizer optimizer;
        auto *solver = new g2o::OptimizationAlgorithmLevenberg(
                g2o::make_unique<g2o::BlockSolverX>(g2o::make_unique<LinearSolverEigenX>())
        );

        optimizer.setAlgorithm(solver);
        optimizer.setVerbose(false);

        // 3. set vertices and edges
        Vertex3D *vVelo2, *vGyroBias2, *vAccBias2;
        Pose Twb2;
        for (int i = numOpt - 1; i >= 0; i--) {
            const shared_ptr<KeyFrame> &kf = optimizeKeyFrames[i];

            auto *vVelo1 = new Vertex3D(kf->getVelocity());
            vVelo1->setId(maxKFId + 3 * kf->id + 1);
            vVelo1->setFixed(i == 0);
            optimizer.addVertex(vVelo1);

            const shared_ptr<PreIntegrator> &preIntegrator = kf->pre_integrator;
            const Bias bias = preIntegrator->updated_bias;
            auto *vGyroBias1 = new Vertex3D(bias.bg);
            vGyroBias1->setId(maxKFId + 3 * kf->id + 2);
            vGyroBias1->setFixed(i == 0);
            optimizer.addVertex(vGyroBias1);

            auto *vAccBias1 = new Vertex3D(bias.ba);
            vAccBias1->setId(maxKFId + 3 * kf->id + 3);
            vAccBias1->setFixed(i == 0);
            optimizer.addVertex(vAccBias1);

            const Pose &Twb1 = kf->getImuPose();

            if (i < numOpt - 1) {
                auto *eInertialOnly = new EdgeInertialOnly(Twb1, Twb2, preIntegrator);
                eInertialOnly->setVertex(0, vVelo1);
                eInertialOnly->setVertex(1, vGyroBias1);
                eInertialOnly->setVertex(2, vAccBias1);
                eInertialOnly->setVertex(3, vVelo2);
                optimizer.addEdge(eInertialOnly);

                auto *eGyroWalk = new EdgeBiasWalk();
                eGyroWalk->setVertex(0, vGyroBias1);
                eGyroWalk->setVertex(1, vGyroBias2);
                eGyroWalk->setInformation(preIntegrator->C.block<3, 3>(9, 9).cast<double>().inverse());
                optimizer.addEdge(eGyroWalk);

                auto *eAccWalk = new EdgeBiasWalk();
                eAccWalk->setVertex(0, vAccBias1);
                eAccWalk->setVertex(1, vAccBias2);
                eAccWalk->setInformation(preIntegrator->C.block<3, 3>(12, 12).cast<double>().inverse());
                optimizer.addEdge(eAccWalk);
            }

            vVelo2 = vVelo1;
            vGyroBias2 = vGyroBias1;
            vAccBias2 = vAccBias1;
            Twb2 = Twb1;
        }

        // 4. optimize
        optimizer.initializeOptimization();
        optimizer.optimize(iterations);

        // 5. recover
        Bias newBias;
        for (int i = 0; i < numOpt; ++i) {
            const shared_ptr<KeyFrame> &kf = optimizeKeyFrames[i];

            Eigen::Vector3d optVelo = dynamic_cast<Vertex3D *>(optimizer.vertex(maxKFId + 3 * kf->id + 1))->estimate();
            kf->setVelocity(optVelo.cast<float>());

            newBias.bg = dynamic_cast<Vertex3D *>(optimizer.vertex(maxKFId + 3 * kf->id + 2))->estimate().cast<float>();
            newBias.ba = dynamic_cast<Vertex3D *>(optimizer.vertex(maxKFId + 3 * kf->id + 3))->estimate().cast<float>();
            kf->setImuBias(newBias);
        }

        mapper_logger << "After inertial optimize\n";
        for (const auto &kf : optimizeKeyFrames) {
            mapper_logger << titles[1] << "keyframe id " << kf->id << ", frame_id " << kf->frame_id << "\n";
            mapper_logger << titles[1] << " - pose: " << kf->getPose() << "\n";
            mapper_logger << titles[1] << " - velo: " << kf->getVelocity() << "\n";
            mapper_logger << titles[1] << " - bias: " << kf->pre_integrator->updated_bias << "\n";
        }
    }

    void Optimize::localFullBundleAdjustment(const std::shared_ptr<KeyFrame> &keyFrame, Map *pointMap, bool beLarge,
                                                 bool *stopFlag) {
        // 1. get optimizable keyframes, fixed keyframes and map-points
        int maxOpt = 10, iterations = 10;
        if (beLarge) {
            maxOpt = 25;
            iterations = 4;
        }

        const int numOpt = min(pointMap->getNumKeyFrames() - 2, maxOpt);
        const unsigned int maxKFId = keyFrame->id;

        vector<shared_ptr<KeyFrame>> optimizeKeyFrames = pointMap->getRecentKeyFrames(numOpt);
        for (const auto &kf: optimizeKeyFrames) {
            kf->BA_local_for_kf = maxKFId;
        }


        // optimizable points seen by temporal optimizable keyframes
        list<shared_ptr<MapPoint>> localMapPoints;
        for (int i = 0; i < numOpt; ++i) {
            vector<shared_ptr<MapPoint>> mapPoints = optimizeKeyFrames[i]->getMapPoints();
            for (const auto &mp: mapPoints) {
                if (mp && !mp->isBad() && mp->BA_local_for_kf != maxKFId) {
                    localMapPoints.push_back(mp);
                    mp->BA_local_for_kf = maxKFId;
                }
            }
        }

        // fixed keyframes
        const int maxNumFixed = 150;
        vector<shared_ptr<KeyFrame>> fixedKeyFrames;
        for (const auto &mp: localMapPoints) {
            map<shared_ptr<KeyFrame>, size_t> observations = mp->getObservations();
            for (const auto &obsPair: observations) {
                const auto &kf = obsPair.first;
                if (!kf->isBad() && kf->BA_local_for_kf != maxKFId && kf->BA_fixed_for_kf != maxKFId) {
                    fixedKeyFrames.push_back(kf);
                    kf->BA_fixed_for_kf = maxKFId;
                }
            }

            if (fixedKeyFrames.size() >= maxNumFixed) break;
        }

        mapper_logger << titles[0] << "there are " << optimizeKeyFrames.size() << " local keyframes, "
                      << fixedKeyFrames.size()
                      << " fixed keyframes, " << localMapPoints.size() << " local map-points\n";

        // 2. set up optimizer
        g2o::SparseOptimizer optimizer;
        auto *solver = new g2o::OptimizationAlgorithmLevenberg(
                g2o::make_unique<g2o::BlockSolverX>(g2o::make_unique<LinearSolverEigenX>())
        );
        if (beLarge) {
            solver->setUserLambdaInit(1e-2);
        } else {
            solver->setUserLambdaInit(1e0);
        }

        optimizer.setAlgorithm(solver);
        if (stopFlag)
            optimizer.setForceStopFlag(stopFlag);

        // 3. set vertices and edges
        // optimizable keyframes vertices
        VertexPose *vPose2;
        Vertex3D *vVelo2, *vGyroBias2, *vAccBias2;
        for (int i = numOpt - 1; i >= 0; i--) {
            const shared_ptr<KeyFrame> &kf = optimizeKeyFrames[i];

            auto *vPose1 = new VertexPose({kf->getPose(), kf->getImuPose()});
            vPose1->setId(kf->id);
            optimizer.addVertex(vPose1);

            auto *vVelo1 = new Vertex3D(kf->getVelocity());
            vVelo1->setId(maxKFId + 3 * kf->id + 1);
            optimizer.addVertex(vVelo1);

            const Bias bias = kf->pre_integrator->updated_bias;
            auto *vGyroBias1 = new Vertex3D(bias.bg);
            vGyroBias1->setId(maxKFId + 3 * kf->id + 2);
            optimizer.addVertex(vGyroBias1);

            auto *vAccBias1 = new Vertex3D(bias.ba);
            vAccBias1->setId(maxKFId + 3 * kf->id + 3);
            optimizer.addVertex(vAccBias1);

            if (i < numOpt - 1) {
                auto *eInertial = new EdgeInertial(kf->pre_integrator);
                eInertial->setVertex(0, vPose1);
                eInertial->setVertex(1, vVelo1);
                eInertial->setVertex(2, vGyroBias1);
                eInertial->setVertex(3, vAccBias1);
                eInertial->setVertex(4, vPose2);
                eInertial->setVertex(5, vVelo2);
                optimizer.addEdge(eInertial);

                auto *eGyroWalk = new EdgeBiasWalk();
                eGyroWalk->setVertex(0, vGyroBias1);
                eGyroWalk->setVertex(1, vGyroBias2);
                eGyroWalk->setInformation(kf->pre_integrator->C.block<3, 3>(9, 9).cast<double>().inverse());
                optimizer.addEdge(eGyroWalk);

                auto *eAccWalk = new EdgeBiasWalk();
                eAccWalk->setVertex(0, vAccBias1);
                eAccWalk->setVertex(1, vAccBias2);
                eAccWalk->setInformation(kf->pre_integrator->C.block<3, 3>(12, 12).cast<double>().inverse());
                optimizer.addEdge(eAccWalk);
            }

            if (i == 0) {
                auto *eVeloPriori = new EdgePriori3D(kf->velo_priori);
                eVeloPriori->setVertex(0, vVelo1);
                eVeloPriori->setInformation(kf->velo_info.cast<double>());
                optimizer.addEdge(eVeloPriori);

                auto *eGyroPriori = new EdgePriori3D(kf->bias_priori.bg);
                eGyroPriori->setVertex(0, vGyroBias1);
                eGyroPriori->setInformation(kf->gyro_info.cast<double>());
                optimizer.addEdge(eGyroPriori);

                auto *eAccPriori = new EdgePriori3D(kf->bias_priori.ba);
                eAccPriori->setVertex(0, vAccBias1);
                eAccPriori->setInformation(kf->gyro_info.cast<double>());
                optimizer.addEdge(eAccPriori);
            }

            vPose2 = vPose1;
            vVelo2 = vVelo1;
            vGyroBias2 = vGyroBias1;
            vAccBias2 = vAccBias1;
        }

        // fixed keyframe vertices
        for (const auto &kf: fixedKeyFrames) {
            auto *vPose = new VertexPose({kf->getPose(), kf->getImuPose()});
            vPose->setId(kf->id);
            vPose->setFixed(true);
            optimizer.addVertex(vPose);
        }

        // map-points vertices and edges
        const int expectedSize = numOpt * localMapPoints.size();
        vector<EdgeMono *> edges;
        edges.reserve(expectedSize);

        vector<pair<shared_ptr<KeyFrame>, shared_ptr<MapPoint>>> vecEdgeObs;
        vecEdgeObs.reserve(expectedSize);

        const float thHuber = sqrtf(5.991);
        const float chi2Mono = 5.991;
        const unsigned int iniMPid = maxKFId * 5 + 1;
        for (const auto &mp: localMapPoints) {
            auto *vPoint = new Vertex3D(mp->getPos());
            vPoint->setId(mp->id + iniMPid);
            vPoint->setMarginalized(true);
            optimizer.addVertex(vPoint);

            const map<shared_ptr<KeyFrame>, size_t> observation = mp->getObservations();
            for (const auto &obsPair: observation) {
                const shared_ptr<KeyFrame> &kf = obsPair.first;
                if (kf->isBad()) continue;
                const cv::KeyPoint &kp = kf->key_points[obsPair.second];

                auto *eVisual = new EdgeMono(kp.pt);
                eVisual->setVertex(0, vPoint);
                eVisual->setVertex(1, optimizer.vertex(kf->id));
                const float invSigma2 = ORBExtractor::getInvSquareSigma(kp.octave);
                eVisual->setInformation(invSigma2 * Eigen::Matrix2d::Identity());

                auto *rk = new g2o::RobustKernelHuber;
                eVisual->setRobustKernel(rk);
                rk->setDelta(thHuber);

                optimizer.addEdge(eVisual);
                edges.push_back(eVisual);
                vecEdgeObs.emplace_back(kf, mp);
            }
        }

        // 4. optimize
        optimizer.initializeOptimization();
        optimizer.optimize(iterations);

        // 5. check inlier observations
        int numEdge = (int) edges.size();
        vector<int> eraseIndices;
        eraseIndices.reserve(numEdge / 2);

        for (int i = 0; i < numEdge; ++i) {
            auto *e = edges[i];

            if (e->chi2() > chi2Mono) {
                eraseIndices.push_back(i);
            }
        }

        // 6. erase outliers
        lock_guard<mutex> lock(pointMap->map_update_mutex);
        for (auto idx: eraseIndices) {
            const shared_ptr<KeyFrame> &kf = vecEdgeObs[idx].first;
            const shared_ptr<MapPoint> &mp = vecEdgeObs[idx].second;

            if (mp->isBad()) continue;
            kf->eraseMapPoint(mp->getFeatureId(kf));
            mp->eraseObservation(kf);
        }

        // 7. recover
        Bias lastBias;
        for (int i = 0; i < numOpt; ++i) {
            const shared_ptr<KeyFrame> &kf = optimizeKeyFrames[i];

            CameraImuPose optimizedPose = dynamic_cast<VertexPose *>(optimizer.vertex(kf->id))->estimate();
            Eigen::Vector3f optimizedVelo = dynamic_cast<Vertex3D *>(optimizer.vertex(
                    maxKFId + 3 * kf->id + 1))->estimate().cast<float>();
            Eigen::Vector3f bg = dynamic_cast<Vertex3D *>(optimizer.vertex(
                    maxKFId + 3 * kf->id + 2))->estimate().cast<float>();
            Eigen::Vector3f ba = dynamic_cast<Vertex3D *>(optimizer.vertex(
                    maxKFId + 3 * kf->id + 3))->estimate().cast<float>();

            kf->setPose(optimizedPose.R_cw.cast<float>(), optimizedPose.t_cw.cast<float>());
            kf->setVelocity(optimizedVelo);

            Bias newBias(bg, ba);
            kf->setImuBias(newBias);

            if (i > 0) {
                kf->bias_priori = lastBias;
                kf->setPrioriInformation(optimizeKeyFrames[i - 1]->pre_integrator->C);
            }

            lastBias = newBias;
        }

        for (const auto &mp: localMapPoints) {
            auto *vPoint = dynamic_cast<Vertex3D *>(optimizer.vertex(iniMPid + mp->id));
            if (!mp->isBad()) {
                mp->setPos(vPoint->estimate().cast<float>());
                mp->update();
            }
        }

        pointMap->increaseChangeIdx();
    }

} // mono_orb_slam3