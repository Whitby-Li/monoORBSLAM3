//
// Created by whitby on 8/22/23.
//

#include "G2oTypes.h"
#include "Utils/LieAlgeBra.h"
#include "Sensor/Camera.h"

namespace mono_orb_slam3 {
    void CameraImuPose::update(const double *pu) {
        Eigen::Vector3d delta_t(pu[3], pu[4], pu[5]);
        t_wb += R_wb * delta_t;
        R_wb = R_wb * lie::ExpSO3(pu[0], pu[1], pu[2]);

        // normalized rotation after 3 updates
        iter++;
        if (iter == 3) {
            R_wb = lie::NormalizeRotation(R_wb);
            iter = 0;
        }

        // update camera pose
        R_cw = imu_ptr->R_cb * R_wb.transpose();
        t_cw = imu_ptr->t_cb - R_cw * t_wb;
    }

    void EdgeSE3Project3DOnlyPose::linearizeOplus() {
        const auto *vSE3 = dynamic_cast<const VertexSE3 *>(_vertices[0]);
        const Eigen::Vector3d Pc = vSE3->estimate().map(P_w);

        Eigen::Matrix<double, 2, 3> projectJacobian = camera->getProjJacobian(Pc);
        _jacobianOplusXi.block<2, 3>(0, 0) = projectJacobian * lie::Hat(Pc);
        _jacobianOplusXi.block<2, 3>(0, 3) = -projectJacobian;
    }

    void EdgeSE3Project3D::linearizeOplus() {
        const auto *vSE3 = dynamic_cast<const VertexSE3 *>(_vertices[1]);
        g2o::SE3Quat Tcw(vSE3->estimate());
        const auto *vPoint = dynamic_cast<const Vertex3D *>(_vertices[0]);

        Eigen::Vector3d Pc = Tcw.map(vPoint->estimate());
        Eigen::Matrix<double, 2, 3> projectJacobian = camera->getProjJacobian(Pc);

        _jacobianOplusXi = -projectJacobian * Tcw.rotation().toRotationMatrix();
        _jacobianOplusXj.block<2, 3>(0, 0) = projectJacobian * lie::Hat(Pc);
        _jacobianOplusXj.block<2, 3>(0, 3) = -projectJacobian;
    }

    void EdgeMonoOnlyPose::linearizeOplus() {
        const auto pose = dynamic_cast<const VertexPose *>(_vertices[0])->estimate();

        Eigen::Vector3d Pc = pose.worldToCamera(P_w);
        Eigen::Matrix<double, 2, 3> projectJacobian = camera->getProjJacobian(Pc);

        _jacobianOplusXi.block<2, 3>(0, 0) = -projectJacobian * pose.getImuRotationJacobian(Pc);
        _jacobianOplusXi.block<2, 3>(0, 3) = -projectJacobian * pose.getImuTranslationJacobian();
    }

    void EdgeMono::linearizeOplus() {
        const auto pose = dynamic_cast<const VertexPose *>(_vertices[1])->estimate();
        const auto Pw = dynamic_cast<const Vertex3D *>(_vertices[0])->estimate();

        Eigen::Vector3d Pc = pose.worldToCamera(Pw);
        Eigen::Matrix<double, 2, 3> projectJacobian = camera->getProjJacobian(Pc);

        _jacobianOplusXi = -projectJacobian * pose.R_cw;
        _jacobianOplusXj.block<2, 3>(0, 0) = -projectJacobian * pose.getImuRotationJacobian(Pc);
        _jacobianOplusXj.block<2, 3>(0, 3) = -projectJacobian * pose.getImuTranslationJacobian();
    }

    EdgeInertialGS::EdgeInertialGS(const CameraImuPose &pose1, const CameraImuPose &pose2,
                                   const std::shared_ptr<PreIntegrator> &preIntegrator)
            : Rb1w(pose1.R_wb.transpose()), Rwb2(pose2.R_wb), JRg(preIntegrator->JRg.cast<double>()),
              JVg(preIntegrator->JVg.cast<double>()), JPg(preIntegrator->JPg.cast<double>()),
              JVa(preIntegrator->JVa.cast<double>()), JPa(preIntegrator->JPa.cast<double>()),
              pre_integrator(preIntegrator), dt(preIntegrator->delta_t) {
        // this edge links 6 vertices
        g2o::BaseMultiEdge<9, Vector9d>::resize(6);
        gI << 0, 0, -GRAVITY_VALUE;

        Oc1 = -pose1.R_cw.transpose() * pose1.t_cw, Oc2 = -pose2.R_cw.transpose() * pose2.t_cw;
        Pc1b1 = pose1.getTranslationCameraToImu(), Pc2b2 = pose2.getTranslationCameraToImu();

        Eigen::Matrix<double, 9, 9> info = pre_integrator->C.block<9, 9>(0, 0).cast<double>().inverse();
        info = (info + info.transpose()) / 2;
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, 9, 9>> eigenSolver(info);
        Eigen::Matrix<double, 9, 1> eigenValues = eigenSolver.eigenvalues();
        for (int i = 0; i < 9; ++i)
            if (eigenValues[i] < 1e-12) eigenValues[i] = 0;
        info = eigenSolver.eigenvectors() * eigenValues.asDiagonal() * eigenSolver.eigenvectors().transpose();
        setInformation(info);
    }

    void EdgeInertialGS::computeError() {
        const auto velo1 = dynamic_cast<const Vertex3D *>(_vertices[0])->estimate();
        const auto bg = dynamic_cast<const Vertex3D *>(_vertices[1])->estimate().cast<float>();
        const auto ba = dynamic_cast<const Vertex3D *>(_vertices[2])->estimate().cast<float>();
        const auto velo2 = dynamic_cast<const Vertex3D *>(_vertices[3])->estimate();
        const auto gDir = dynamic_cast<const VertexGravity *>(_vertices[4])->estimate();
        const auto scale = dynamic_cast<const VertexScale *>(_vertices[5])->estimate();

        g = gDir.R_wg * gI;
        const Eigen::Matrix3d dR = pre_integrator->getDeltaRotation(bg).cast<double>();
        const Eigen::Vector3d dV = pre_integrator->getDeltaVelocity(bg, ba).cast<double>();
        const Eigen::Vector3d dP = pre_integrator->getDeltaPosition(bg, ba).cast<double>();

        const Eigen::Vector3d er = lie::LogSO3(dR.transpose() * Rb1w * Rwb2);
        const Eigen::Vector3d ev = Rb1w * (scale * (velo2 - velo1) - g * dt) - dV;
        const Eigen::Vector3d ep =
                Rb1w * (scale * Oc2 + Pc2b2 - scale * Oc1 - Pc1b1 - scale * velo1 * dt - 0.5 * g * dt * dt) - dP;

        _error << er, ev, ep;
    }

    void EdgeInertialGS::linearizeOplus() {
        const auto velo1 = dynamic_cast<const Vertex3D *>(_vertices[0])->estimate();
        const auto bg = dynamic_cast<const Vertex3D *>(_vertices[1])->estimate().cast<float>();
        const auto ba = dynamic_cast<const Vertex3D *>(_vertices[2])->estimate().cast<float>();
        const auto velo2 = dynamic_cast<const Vertex3D *>(_vertices[3])->estimate();
        const auto gDir = dynamic_cast<const VertexGravity *>(_vertices[4])->estimate();
        const auto scale = dynamic_cast<const VertexScale *>(_vertices[5])->estimate();

        const Eigen::Vector3d delta_bg = (bg - pre_integrator->bias.bg).cast<double>();
        Eigen::MatrixXd gm = Eigen::MatrixXd::Zero(3, 2);
        gm(0, 1) = -GRAVITY_VALUE, gm(1, 0) = GRAVITY_VALUE;
        const Eigen::MatrixXd JGdTheta = gDir.R_wg * gm;

        const Eigen::Matrix3d dR = pre_integrator->getDeltaRotation(bg).cast<double>();
        const Eigen::Matrix3d eR = dR.transpose() * Rb1w * Rwb2;
        const Eigen::Vector3d er = lie::LogSO3(eR);
        const Eigen::Matrix3d invJr = lie::InverseRightJacobianSO3(er);


        // jacobian wrt velo1
        _jacobianOplus[0].setZero();
        _jacobianOplus[0].block<3, 3>(3, 0) = -scale * Rb1w;
        _jacobianOplus[0].block<3, 3>(6, 0) = -scale * dt * Rb1w;

        // jacobian wrt gyro bias
        _jacobianOplus[1].setZero();
        _jacobianOplus[1].block<3, 3>(0, 0) = -invJr * eR.transpose() * lie::RightJacobianSO3(JRg * delta_bg) * JRg;
        /*_jacobianOplus[1].block<3, 3>(3, 0) = -JVg;
        _jacobianOplus[1].block<3, 3>(6, 0) = -JPg;*/

        // jacobian wrt acc bias
        _jacobianOplus[2].setZero();
        _jacobianOplus[2].block<3, 3>(3, 0) = -JVa;
        _jacobianOplus[2].block<3, 3>(6, 0) = -JPa;

        // jacobian wrt velo2
        _jacobianOplus[3].setZero();
        _jacobianOplus[3].block<3, 3>(3, 0) = scale * Rb1w;

        // jacobian wrt gravity direction
        _jacobianOplus[4].setZero();
        _jacobianOplus[4].block<3, 2>(3, 0) = -Rb1w * JGdTheta * dt;
        _jacobianOplus[4].block<3, 2>(6, 0) = -0.5 * dt * dt * Rb1w * JGdTheta;

        // jacobian wrt scale factor
        _jacobianOplus[5].setZero();
        _jacobianOplus[5].block<3, 1>(3, 0) = scale * Rb1w * (velo2 - velo1);
        _jacobianOplus[5].block<3, 1>(6, 0) = scale * Rb1w * (Oc2 - Oc1 - velo1 * dt);
    }

    EdgeInertialGS2::EdgeInertialGS2(const CameraImuPose &pose1, const CameraImuPose &pose2,
                                 const std::shared_ptr<PreIntegrator> &preIntegrator)
            : Rb1w(pose1.R_wb.transpose()), Rwb2(pose2.R_wb),
              JRg(preIntegrator->JRg.cast<double>()), JVg(preIntegrator->JVg.cast<double>()),
              JPg(preIntegrator->JPg.cast<double>()), JVa(preIntegrator->JVa.cast<double>()),
              JPa(preIntegrator->JPa.cast<double>()), pre_integrator(preIntegrator), dt(preIntegrator->delta_t) {
        // this edge links 6 vertices
        g2o::BaseMultiEdge<9, Vector9d>::resize(6);
        gI << 0, 0, -GRAVITY_VALUE;

        Oc1 = -pose1.R_cw.transpose() * pose1.t_cw, Oc2 = -pose2.R_cw.transpose() * pose2.t_cw;
        Pc1b1 = pose1.getTranslationCameraToImu(), Pc2b2 = pose2.getTranslationCameraToImu();

        Eigen::Matrix<double, 9, 9> info = pre_integrator->C.block<9, 9>(0, 0).cast<double>().inverse();
        info = (info + info.transpose()) / 2;
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, 9, 9>> eigenSolver(info);
        Eigen::Matrix<double, 9, 1> eigenValues = eigenSolver.eigenvalues();
        for (int i = 0; i < 9; ++i)
            if (eigenValues[i] < 1e-12) eigenValues[i] = 0;
        info = eigenSolver.eigenvectors() * eigenValues.asDiagonal() * eigenSolver.eigenvectors().transpose();

        setInformation(info);
    }

    void EdgeInertialGS2::computeError() {
        const auto velo1 = dynamic_cast<const Vertex3D *>(_vertices[0])->estimate();
        const auto bg = dynamic_cast<const Vertex3D *>(_vertices[1])->estimate().cast<float>();
        const auto ba = dynamic_cast<const Vertex3D *>(_vertices[2])->estimate().cast<float>();
        const auto velo2 = dynamic_cast<const Vertex3D *>(_vertices[3])->estimate();
        const auto gDir = dynamic_cast<const VertexGravity *>(_vertices[4])->estimate();
        const auto scale = dynamic_cast<const VertexScale *>(_vertices[5])->estimate();

        g = gDir.R_wg * gI;
        const Eigen::Matrix3d dR = pre_integrator->getDeltaRotation(bg).cast<double>();
        const Eigen::Vector3d dV = pre_integrator->getDeltaVelocity(bg, ba).cast<double>();
        const Eigen::Vector3d dP = pre_integrator->getDeltaPosition(bg, ba).cast<double>();

        const Eigen::Vector3d &er = lie::LogSO3(dR.transpose() * Rb1w * Rwb2);
        const Eigen::Vector3d &ev = Rb1w * (velo2 - velo1 - g * dt) - dV;
        const Eigen::Vector3d ep =
                Rb1w * (scale * Oc2 + Pc2b2 - scale * Oc1 - Pc1b1 - velo1 * dt - 0.5 * g * dt * dt) - dP;

        _error << er, ev, ep;
    }

    void EdgeInertialGS2::linearizeOplus() {
        const auto velo1 = dynamic_cast<const Vertex3D *>(_vertices[0])->estimate();
        const auto bg = dynamic_cast<const Vertex3D *>(_vertices[1])->estimate().cast<float>();
        const auto ba = dynamic_cast<const Vertex3D *>(_vertices[2])->estimate().cast<float>();
        const auto velo2 = dynamic_cast<const Vertex3D *>(_vertices[3])->estimate();
        const auto gDir = dynamic_cast<const VertexGravity *>(_vertices[4])->estimate();
        const auto scale = dynamic_cast<const VertexScale *>(_vertices[5])->estimate();

        const Eigen::Vector3d delta_bg = (bg - pre_integrator->bias.bg).cast<double>();
        Eigen::MatrixXd gm = Eigen::MatrixXd::Zero(3, 2);
        gm(0, 1) = -GRAVITY_VALUE, gm(1, 0) = GRAVITY_VALUE;
        const Eigen::MatrixXd JGdTheta = gDir.R_wg * gm;

        const Eigen::Matrix3d dR = pre_integrator->getDeltaRotation(bg).cast<double>();
        const Eigen::Matrix3d eR = dR.transpose() * Rb1w * Rwb2;
        const Eigen::Vector3d er = lie::LogSO3(eR);
        const Eigen::Matrix3d invJr = lie::InverseRightJacobianSO3(er);

        // jacobian wrt velo1
        _jacobianOplus[0].setZero();
        _jacobianOplus[0].block<3, 3>(3, 0) = -Rb1w;
        _jacobianOplus[0].block<3, 3>(6, 0) = -dt * Rb1w;

        // jacobian wrt bg
        _jacobianOplus[1].setZero();
        _jacobianOplus[1].block<3, 3>(0, 0) = -invJr * eR.transpose() * lie::RightJacobianSO3(JRg * delta_bg) * JRg;
        /*_jacobianOplus[1].block<3, 3>(3, 0) = -JVg;
        _jacobianOplus[1].block<3, 3>(6, 0) = -JPg;*/

        // jacobian wrt ba
        _jacobianOplus[2].setZero();
        _jacobianOplus[2].block<3, 3>(3, 0) = -JVa;
        _jacobianOplus[2].block<3, 3>(6, 0) = -JPg;

        // jacobian wrt velo2
        _jacobianOplus[3].setZero();
        _jacobianOplus[3].block<3, 3>(3, 0) = Rb1w;

        // jacobian wrt gravity direction
        _jacobianOplus[4].setZero();
        _jacobianOplus[4].block<3, 2>(3, 0) = -Rb1w * JGdTheta * dt;
        _jacobianOplus[4].block<3, 2>(6, 0) = -0.5 * dt * dt * Rb1w * JGdTheta;

        // jacobian wrt scale
        _jacobianOplus[5].setZero();
        _jacobianOplus[5].block<3, 1>(6, 0) = scale * Rb1w * (Oc2 - Oc1);
    }

    EdgeGravity::EdgeGravity(const std::shared_ptr<KeyFrame> &kf1, const std::shared_ptr<KeyFrame> &kf2) {
        gI << 0, 0, -GRAVITY_VALUE;
        const std::shared_ptr<PreIntegrator> &preIntegrator = kf1->pre_integrator;
        dt = preIntegrator->delta_t;

        const Pose Twb1 = kf1->getImuPose(), Twb2 = kf2->getImuPose();
        const Eigen::Vector3f v1 = kf1->getVelocity(), v2 = kf2->getVelocity();
        mea_velo = ((v2 - v1) - Twb1.R * preIntegrator->getUpdatedDeltaVelocity()).cast<double>();
        mea_pos = ((Twb2.t - Twb1.t - v1 * dt) - Twb1.R * preIntegrator->getUpdatedDeltaPosition()).cast<double>();

        Eigen::Matrix<double, 6, 6> info = preIntegrator->C.block<6, 6>(3, 3).cast<double>().inverse();
        info = (info + info.transpose()) / 2;
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, 6, 6>> eigenSolver(info);
        Eigen::Matrix<double, 6, 1> eigenValues = eigenSolver.eigenvalues();
        for (int i = 0; i < 6; ++i)
            if (eigenValues[i] < 1e-12) eigenValues[i] = 0;
        info = eigenSolver.eigenvectors() * eigenValues.asDiagonal() * eigenSolver.eigenvectors().transpose();
        setInformation(info);
    }

    void EdgeGravity::computeError() {
        const auto gDir = dynamic_cast<const VertexGravity *>(_vertices[0])->estimate();
        g = gDir.R_wg * gI;

        _error.block<3, 1>(0, 0) = mea_velo - g * dt;
        _error.block<3, 1>(3, 0) = mea_pos - 0.5 * g * dt * dt;
    }

    void EdgeGravity::linearizeOplus() {
        const auto gDir = dynamic_cast<const VertexGravity *>(_vertices[0])->estimate();

        Eigen::MatrixXd gm = Eigen::MatrixXd::Zero(3, 2);
        gm(0, 1) = -GRAVITY_VALUE, gm(1, 0) = GRAVITY_VALUE;
        const Eigen::MatrixXd JGdTheta = gDir.R_wg * gm;

        _jacobianOplusXi.block<3, 2>(0, 0) = -JGdTheta * dt;
        _jacobianOplusXi.block<3, 2>(3, 0) = -0.5 * dt * dt * JGdTheta;
    }

    EdgeInertialOnly::EdgeInertialOnly(const Pose &Twb1, const Pose &Twb2,
                                       const std::shared_ptr<PreIntegrator> &preIntegrator)
            : Rb1w(Twb1.R.transpose().cast<double>()), twb1(Twb1.t.cast<double>()), Rwb2(Twb2.R.cast<double>()),
              twb2(Twb2.t.cast<double>()), JRg(preIntegrator->JRg.cast<double>()),
              JVg(preIntegrator->JVg.cast<double>()), JVa(preIntegrator->JVa.cast<double>()),
              JPg(preIntegrator->JPg.cast<double>()), JPa(preIntegrator->JPa.cast<double>()),
              pre_integrator(preIntegrator), dt(preIntegrator->delta_t) {
        // this edge links 4 vertices
        g2o::BaseMultiEdge<9, Vector9d>::resize(4);
        g << 0, 0, -GRAVITY_VALUE;

        Eigen::Matrix<double, 9, 9> info = pre_integrator->C.block<9, 9>(0, 0).cast<double>().inverse();
        info = (info + info.transpose()) / 2;
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, 9, 9>> eigenSolver(info);
        Eigen::Matrix<double, 9, 1> eigenValues = eigenSolver.eigenvalues();
        for (int i = 0; i < 9; ++i)
            if (eigenValues[i] < 1e-12) eigenValues[i] = 0;
        info = eigenSolver.eigenvectors() * eigenValues.asDiagonal() * eigenSolver.eigenvectors().transpose();

        setInformation(info);
    }

    void EdgeInertialOnly::computeError() {
        const Eigen::Vector3d velo1 = dynamic_cast<const Vertex3D *>(_vertices[0])->estimate();
        const Eigen::Vector3f bg = dynamic_cast<const Vertex3D *>(_vertices[1])->estimate().cast<float>();
        const Eigen::Vector3f ba = dynamic_cast<const Vertex3D *>(_vertices[2])->estimate().cast<float>();
        const Eigen::Vector3d velo2 = dynamic_cast<const Vertex3D *>(_vertices[3])->estimate();

        const Eigen::Matrix3d dR = pre_integrator->getDeltaRotation(bg).cast<double>();
        const Eigen::Vector3d dV = pre_integrator->getDeltaVelocity(bg, ba).cast<double>();
        const Eigen::Vector3d dP = pre_integrator->getDeltaPosition(bg, ba).cast<double>();

        const Eigen::Vector3d er = lie::LogSO3(dR.transpose() * Rb1w * Rwb2);
        const Eigen::Vector3d ev = Rb1w * (velo2 - velo1 - g * dt) - dV;
        const Eigen::Vector3d ep = Rb1w * (twb2 - twb1 - velo1 * dt - 0.5 * g * dt * dt) - dP;
        _error << er, ev, ep;
    }

    void EdgeInertialOnly::linearizeOplus() {
        const Eigen::Vector3d velo1 = dynamic_cast<const Vertex3D *>(_vertices[0])->estimate();
        const Eigen::Vector3f bg = dynamic_cast<const Vertex3D *>(_vertices[1])->estimate().cast<float>();
        const Eigen::Vector3f ba = dynamic_cast<const Vertex3D *>(_vertices[2])->estimate().cast<float>();
        const Eigen::Vector3d velo2 = dynamic_cast<const Vertex3D *>(_vertices[3])->estimate();

        const Eigen::Vector3d delta_bg = (bg - pre_integrator->bias.bg).cast<double>();
        const Eigen::Matrix3d dR = pre_integrator->getDeltaRotation(bg).cast<double>();
        const Eigen::Matrix3d eR = dR.transpose() * Rb1w * Rwb2;
        const Eigen::Vector3d er = lie::LogSO3(eR);
        const Eigen::Matrix3d invJr = lie::InverseRightJacobianSO3(er);

        // jacobian wrt velo1
        _jacobianOplus[0].setZero();
        _jacobianOplus[0].block<3, 3>(3, 0) = -Rb1w;
        _jacobianOplus[0].block<3, 3>(6, 0) = -Rb1w * dt;

        // jacobian wrt gyro bias
        _jacobianOplus[1].setZero();
        _jacobianOplus[1].block<3, 3>(0, 0) = -invJr * eR.transpose() * lie::RightJacobianSO3(JRg * delta_bg) * JRg;
        /*_jacobianOplus[1].block<3, 3>(3, 0) = -JVg;
        _jacobianOplus[1].block<3, 3>(6, 0) = -JPg;*/

        // jacobian wrt acc bias
        _jacobianOplus[2].setZero();
        _jacobianOplus[2].block<3, 3>(3, 0) = -JVa;
        _jacobianOplus[2].block<3, 3>(6, 0) = -JPa;

        // jacobian wrt velo2
        _jacobianOplus[3].setZero();
        _jacobianOplus[3].block<3, 3>(3, 0) = Rb1w;
    }

    EdgeInertial::EdgeInertial(const std::shared_ptr<PreIntegrator> &preIntegrator)
            : JRg(preIntegrator->JRg.cast<double>()), JVg(preIntegrator->JVg.cast<double>()),
              JVa(preIntegrator->JVa.cast<double>()), JPg(preIntegrator->JPg.cast<double>()),
              JPa(preIntegrator->JPa.cast<double>()), pre_integrator(preIntegrator), dt(preIntegrator->delta_t) {
        // this edge links 6 vertices
        g2o::BaseMultiEdge<9, Vector9d>::resize(6);
        g << 0, 0, -GRAVITY_VALUE;

        Eigen::Matrix<double, 9, 9> info = pre_integrator->C.block<9, 9>(0, 0).cast<double>().inverse();
        info = (info + info.transpose()) / 2;
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, 9, 9>> eigenSolver(info);
        Eigen::Matrix<double, 9, 1> eigenValues = eigenSolver.eigenvalues();
        for (int i = 0; i < 9; ++i)
            if (eigenValues[i] < 1e-12) eigenValues[i] = 0;
        info = eigenSolver.eigenvectors() * eigenValues.asDiagonal() * eigenSolver.eigenvectors().transpose();

        setInformation(info);
    }

    void EdgeInertial::computeError() {
        const auto pose1 = dynamic_cast<const VertexPose *>(_vertices[0])->estimate();
        const Eigen::Vector3d velo1 = dynamic_cast<const Vertex3D *>(_vertices[1])->estimate();
        const Eigen::Vector3f bg = dynamic_cast<const Vertex3D *>(_vertices[2])->estimate().cast<float>();
        const Eigen::Vector3f ba = dynamic_cast<const Vertex3D *>(_vertices[3])->estimate().cast<float>();
        const auto pose2 = dynamic_cast<const VertexPose *>(_vertices[4])->estimate();
        const Eigen::Vector3d velo2 = dynamic_cast<const Vertex3D *>(_vertices[5])->estimate();

        const Eigen::Matrix3d dR = pre_integrator->getDeltaRotation(bg).cast<double>();
        const Eigen::Vector3d dV = pre_integrator->getDeltaVelocity(bg, ba).cast<double>();
        const Eigen::Vector3d dP = pre_integrator->getDeltaPosition(bg, ba).cast<double>();

        const Eigen::Vector3d er = lie::LogSO3(dR.transpose() * pose1.R_wb.transpose() * pose2.R_wb);
        const Eigen::Vector3d ev = pose1.R_wb.transpose() * (velo2 - velo1 - g * dt) - dV;
        const Eigen::Vector3d ep =
                pose1.R_wb.transpose() * (pose2.t_wb - pose1.t_wb - velo1 * dt - 0.5 * g * dt * dt) - dP;

        _error << er, ev, ep;
    }

    void EdgeInertial::linearizeOplus() {
        const auto pose1 = dynamic_cast<const VertexPose *>(_vertices[0])->estimate();
        const Eigen::Vector3d velo1 = dynamic_cast<const Vertex3D *>(_vertices[1])->estimate();
        const Eigen::Vector3f bg = dynamic_cast<const Vertex3D *>(_vertices[2])->estimate().cast<float>();
        const Eigen::Vector3f ba = dynamic_cast<const Vertex3D *>(_vertices[3])->estimate().cast<float>();
        const auto pose2 = dynamic_cast<const VertexPose *>(_vertices[4])->estimate();
        const Eigen::Vector3d velo2 = dynamic_cast<const Vertex3D *>(_vertices[5])->estimate();

        const Eigen::Vector3d delta_bg = (bg - pre_integrator->bias.bg).cast<double>();
        const Eigen::Matrix3d Rb1w = pose1.R_wb.transpose();

        const Eigen::Matrix3d dR = pre_integrator->getDeltaRotation(bg).cast<double>();
        const Eigen::Matrix3d eR = dR.transpose() * Rb1w * pose2.R_wb;
        const Eigen::Vector3d er = lie::LogSO3(eR);
        const Eigen::Matrix3d invJr = lie::InverseRightJacobianSO3(er);

        // jacobian wrt pose1
        _jacobianOplus[0].setZero();
        _jacobianOplus[0].block<3, 3>(0, 0) = -invJr * pose2.R_wb.transpose() * pose1.R_wb;
        _jacobianOplus[0].block<3, 3>(3, 0) = lie::Hat(Rb1w * (velo2 - velo1 - g * dt));
        _jacobianOplus[0].block<3, 3>(6, 0) = lie::Hat(
                Rb1w * (pose2.t_wb - pose1.t_wb - velo1 * dt - 0.5 * g * dt * dt));
        _jacobianOplus[0].block<3, 3>(6, 3) = -Eigen::Matrix3d::Identity();

        // jacobian wrt velo1
        _jacobianOplus[1].setZero();
        _jacobianOplus[1].block<3, 3>(3, 0) = -Rb1w;
        _jacobianOplus[1].block<3, 3>(6, 0) = -Rb1w * dt;

        // jacobian wrt gyro bias
        _jacobianOplus[2].setZero();
        _jacobianOplus[2].block<3, 3>(0, 0) = -invJr * eR.transpose() * lie::RightJacobianSO3(JRg * delta_bg) * JRg;
        _jacobianOplus[2].block<3, 3>(3, 0) = -JVg;
        _jacobianOplus[2].block<3, 3>(6, 0) = -JPg;

        // jacobian wrt acc bias
        _jacobianOplus[3].setZero();
        _jacobianOplus[3].block<3, 3>(3, 0) = -JVa;
        _jacobianOplus[3].block<3, 3>(6, 0) = -JPa;

        // jacobian wrt pose2
        _jacobianOplus[4].setZero();
        _jacobianOplus[4].block<3, 3>(0, 0) = invJr;
        _jacobianOplus[4].block<3, 3>(6, 3) = Rb1w * pose2.R_wb;

        // jacobian wrt velo2
        _jacobianOplus[5].setZero();
        _jacobianOplus[5].block<3, 3>(3, 0) = Rb1w;
    }
}
