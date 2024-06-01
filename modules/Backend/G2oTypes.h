//
// Created by whitby on 8/22/23.
//

#ifndef MONO_ORB_SLAM3_G2OTYPES_H
#define MONO_ORB_SLAM3_G2OTYPES_H

#include "Sensor/Camera.h"
#include "Sensor/Imu.h"
#include "BasicObject/Pose.h"
#include "Utils/LieAlgeBra.h"
#include "BasicObject/KeyFrame.h"

#include <g2o/core/base_vertex.h>
#include <g2o/core/base_unary_edge.h>
#include <g2o/core/base_binary_edge.h>
#include <g2o/core/base_multi_edge.h>
#include <g2o/types/slam3d/se3quat.h>

#include <utility>

namespace mono_orb_slam3 {
    typedef Eigen::Matrix<double, 2, 1> Vector2d;
    typedef Eigen::Matrix<double, 3, 1> Vector3d;
    typedef Eigen::Matrix<double, 6, 1> Vector6d;
    typedef Eigen::Matrix<double, 9, 1> Vector9d;

    /// camera and imu pose, update imu pose directly
    class CameraImuPose {
    public:
        CameraImuPose() {
            R_cw.setIdentity(), R_wb = imu_ptr->R_cb;
            t_cw.setZero(), t_wb = imu_ptr->t_cb;
        }

        CameraImuPose(const Pose &Tcw, const Pose &Twb) {
            R_cw = Tcw.R.cast<double>(), t_cw = Tcw.t.cast<double>();
            R_wb = Twb.R.cast<double>(), t_wb = Twb.t.cast<double>();
        }

        [[nodiscard]] inline Eigen::Vector3d worldToCamera(const Eigen::Vector3d &Pw) const {
            return R_cw * Pw + t_cw;
        }

        void update(const double *pu);

        [[nodiscard]] Eigen::Matrix3d getImuRotationJacobian(const Eigen::Vector3d &Pc) const {
            const Eigen::Vector3d Pb = imu_ptr->R_bc * Pc + imu_ptr->t_bc;
            return imu_ptr->R_cb * lie::Hat(Pb);
        }

        [[nodiscard]] Eigen::Matrix3d getImuTranslationJacobian() const {
            return -imu_ptr->R_cb;
        }

        [[nodiscard]] Eigen::Vector3d getTranslationCameraToImu() const {
            return R_cw.transpose() * imu_ptr->t_cb;
        }

        /* Imu Pose */
        Eigen::Matrix3d R_wb;
        Eigen::Vector3d t_wb;

        /* Camera Pose */
        Eigen::Matrix3d R_cw;
        Eigen::Vector3d t_cw;

    private:
        const ImuCalib *imu_ptr = ImuCalib::getImuCalib();
        short iter = 0;
    };

    /// gravity direction
    class GravityDirection {
    public:
        GravityDirection() = default;

        explicit GravityDirection(Eigen::Matrix3d Rwg) : R_wg(std::move(Rwg)) {}

        void update(const double *pu) {
            R_wg = R_wg * lie::ExpSO3(pu[0], pu[1], 0);
            iter++;
            if (iter == 3) {
                R_wg = lie::NormalizeRotation(R_wg);
                iter = 0;
            }
        }

        Eigen::Matrix3d R_wg;

    private:
        short iter = 0;
    };

    /// SE3 vertex
    class VertexSE3 : public g2o::BaseVertex<6, g2o::SE3Quat> {
    public:
        VertexSE3() = default;

        VertexSE3(const Eigen::Matrix3f &Rcw, const Eigen::Vector3f &tcw) {
            _estimate = g2o::SE3Quat(Rcw.cast<double>(), tcw.cast<double>());
        }

        bool read(std::istream &is) override { return true; }

        bool write(std::ostream &os) const override { return true; }

        void setToOriginImpl() override {
            _estimate = g2o::SE3Quat();
        }

        void oplusImpl(const double *update_) override {
            Eigen::Map<const Vector6d> update(update_);
            setEstimate(g2o::SE3Quat::exp(update) * estimate());
        }
    };

    /// CameraImuPose vertex
    class VertexPose : public g2o::BaseVertex<6, CameraImuPose> {
    public:
        VertexPose() = default;

        explicit VertexPose(const CameraImuPose &pose) {
            setEstimate(pose);
        }

        bool read(std::istream &is) override { return true; }

        bool write(std::ostream &os) const override { return true; }

        void setToOriginImpl() override {}

        void oplusImpl(const double *update_) override {
            _estimate.update(update_);
            updateCache();
        }
    };

    /// 3d vector vertex (like map-point, gyro_bias, velocity and so on)
    class Vertex3D : public g2o::BaseVertex<3, Vector3d> {
    public:
        Vertex3D() = default;

        explicit Vertex3D(const Eigen::Vector3f &v) {
            _estimate = v.cast<double>();
        }

        bool read(std::istream &is) override { return true; }

        bool write(std::ostream &os) const override { return true; }

        void setToOriginImpl() override {
            _estimate.fill(0);
        }

        void oplusImpl(const double *update) override {
            Eigen::Map<const Vector3d> v(update);
            _estimate += v;
        }

    };

    /// gravity vertex
    class VertexGravity : public g2o::BaseVertex<2, GravityDirection> {
    public:
        VertexGravity() = default;

        explicit VertexGravity(Eigen::Matrix3d Rwg) {
            setEstimate(GravityDirection(std::move(Rwg)));
        }

        bool read(std::istream &is) override { return true; }

        bool write(std::ostream &os) const override { return true; }

        void setToOriginImpl() override {}

        void oplusImpl(const double *update_) override {
            _estimate.update(update_);
            updateCache();
        }
    };

    /// scale vertex
    class VertexScale : public g2o::BaseVertex<1, double> {
    public:
        VertexScale() {
            setEstimate(1.0);
        }

        explicit VertexScale(double predict_scale) {
            setEstimate(predict_scale);
        }

        bool read(std::istream &is) override { return true; }

        bool write(std::ostream &os) const override { return true; }

        void setToOriginImpl() override {
            setEstimate(1.0);
        }

        void oplusImpl(const double *update_) override {
            setEstimate(estimate() * exp(*update_));
        }
    };

    /// the edge between se3 pose and map-point, only optimize pose
    class EdgeSE3Project3DOnlyPose : public g2o::BaseUnaryEdge<2, Vector2d, VertexSE3> {
    public:
        EdgeSE3Project3DOnlyPose() = delete;

        explicit EdgeSE3Project3DOnlyPose(const Eigen::Vector3f &Pw) : camera(Camera::getCamera()) {
            P_w = Pw.cast<double>();
        }

        bool read(std::istream &is) override { return true; }

        bool write(std::ostream &os) const override { return true; }

        void computeError() override {
            const auto *vSE3 = dynamic_cast<const VertexSE3 *>(_vertices[0]);
            _error = _measurement - camera->project(vSE3->estimate().map(P_w));
        }

        bool isDepthPositive() {
            const auto *vPose = dynamic_cast<const VertexSE3 *>(_vertices[0]);
            return (vPose->estimate().map(P_w))(2) > 0;
        }

        void linearizeOplus() override;

    private:
        Vector3d P_w;
        const Camera *camera;
    };

    /// the edge between se3 pose and map-point
    class EdgeSE3Project3D : public g2o::BaseBinaryEdge<2, Vector2d, Vertex3D, VertexSE3> {
    public:
        EdgeSE3Project3D() : camera(Camera::getCamera()) {}

        bool read(std::istream &is) override { return true; }

        bool write(std::ostream &os) const override { return true; }

        void computeError() override {
            const auto *vSE3 = dynamic_cast<const VertexSE3 *>(_vertices[1]);
            const auto *vPoint = dynamic_cast<const Vertex3D *>(_vertices[0]);
            _error = _measurement - camera->project(vSE3->estimate().map(vPoint->estimate()));
        }

        bool isDepthPositive() {
            const auto *vSE3 = dynamic_cast<const VertexSE3 *>(_vertices[1]);
            const auto *vPoint = dynamic_cast<const Vertex3D *>(_vertices[0]);
            return vSE3->estimate().map(vPoint->estimate())(2) > 0;
        }

        void linearizeOplus() override;

    private:
        const Camera *camera;
    };

    /// the edge between CameraImuPose and map-point, only optimize pose
    class EdgeMonoOnlyPose : public g2o::BaseUnaryEdge<2, Eigen::Vector2d, VertexPose> {
    public:
        explicit EdgeMonoOnlyPose(const Eigen::Vector3f &Pw)
                : g2o::BaseUnaryEdge<2, Eigen::Vector2d, VertexPose>(), P_w(Pw.cast<double>()),
                  camera(Camera::getCamera()) {}

        bool read(std::istream &is) override { return true; }

        bool write(std::ostream &os) const override { return true; }

        void computeError() override {
            const auto *vPose = dynamic_cast<const VertexPose *>(_vertices[0]);
            _error = _measurement - camera->project(vPose->estimate().worldToCamera(P_w));
        }

        void linearizeOplus() override;

        bool isDepthPositive() {
            const auto *vPose = dynamic_cast<const VertexPose *>(_vertices[0]);
            return vPose->estimate().worldToCamera(P_w)(2) > 0;
        }

    private:
        const Eigen::Vector3d P_w;
        const Camera *camera;
    };

    /// the edge between CameraImuPose and map-point
    class EdgeMono : public g2o::BaseBinaryEdge<2, Eigen::Vector2d, Vertex3D, VertexPose> {
    public:
        explicit EdgeMono(const cv::Point2f &p)
                : g2o::BaseBinaryEdge<2, Eigen::Vector2d, Vertex3D, VertexPose>(), camera(Camera::getCamera()) {
            _measurement = Eigen::Vector2d(p.x, p.y);
        }

        bool read(std::istream &is) override { return true; }

        bool write(std::ostream &os) const override { return true; }

        void computeError() override {
            const auto *vPose = dynamic_cast<const VertexPose *>(_vertices[1]);
            const auto *vPoint = dynamic_cast<const Vertex3D *>(_vertices[0]);
            _error = _measurement - camera->project(vPose->estimate().worldToCamera(vPoint->estimate()));
        }

        void linearizeOplus() override;

        bool isDepthPositive() {
            const auto *vPose = dynamic_cast<const VertexPose *>(_vertices[1]);
            const auto *vPoint = dynamic_cast<const Vertex3D *>(_vertices[0]);
            return vPose->estimate().worldToCamera(vPoint->estimate())(2) > 0;
        }

    private:
        const Camera *camera;
    };

    /// 3d priori edge
    class EdgePriori3D : public g2o::BaseUnaryEdge<3, Eigen::Vector3d, Vertex3D> {
    public:
        explicit EdgePriori3D(const Eigen::Vector3f &_priori)
                : g2o::BaseUnaryEdge<3, Eigen::Vector3d, Vertex3D>() {
            _measurement = _priori.cast<double>();
        }

        bool read(std::istream &is) override { return true; }

        bool write(std::ostream &os) const override { return true; }

        void computeError() override {
            const auto posteriori = dynamic_cast<const Vertex3D *>(_vertices[0])->estimate();
            _error = _measurement - posteriori;
        }

        void linearizeOplus() override {
            _jacobianOplusXi = -Eigen::Matrix3d::Identity();
        }
    };

    /// inertial, gravity and scale edge
    class EdgeInertialGS : public g2o::BaseMultiEdge<9, Vector9d> {
    public:
        EdgeInertialGS(const CameraImuPose &pose1, const CameraImuPose &pose2,
                       const std::shared_ptr<PreIntegrator> &preIntegrator);

        bool read(std::istream &is) override { return true; }

        bool write(std::ostream &os) const override { return true; }

        void computeError() override;

        void linearizeOplus() override;

        Eigen::Matrix3d Rb1w, Rwb2;
        Eigen::Vector3d Oc1, Oc2;
        Eigen::Vector3d Pc1b1, Pc2b2;
        const Eigen::Matrix3d JRg, JVg, JPg;
        const Eigen::Matrix3d JVa, JPa;
        std::shared_ptr<PreIntegrator> pre_integrator;
        const double dt;
        Eigen::Vector3d g, gI;
    };

    class EdgeInertialGS2 : public g2o::BaseMultiEdge<9, Vector9d> {
    public:
        EdgeInertialGS2(const CameraImuPose &pose1, const CameraImuPose &pose2,
                      const std::shared_ptr<PreIntegrator> &preIntegrator);

        bool read(std::istream &is) override { return true; }

        bool write(std::ostream &os) const override { return true; }

        void computeError() override;

        void linearizeOplus() override;

        Eigen::Matrix3d Rb1w, Rwb2;
        Eigen::Vector3d Oc1, Oc2;
        Eigen::Vector3d Pc1b1, Pc2b2;
        const Eigen::Matrix3d JRg, JVg, JPg;
        const Eigen::Matrix3d JVa, JPa;
        std::shared_ptr<PreIntegrator> pre_integrator;
        const double dt;
        Eigen::Vector3d g, gI;
    };

    /// gravity edge
    class EdgeGravity : public g2o::BaseUnaryEdge<6, Vector6d, VertexGravity> {
    public:
        EdgeGravity(const std::shared_ptr<KeyFrame> &kf1, const std::shared_ptr<KeyFrame> &kf2);

        bool read(std::istream &is) override { return true; }

        bool write(std::ostream &os) const override { return true; }

        void computeError() override;

        void linearizeOplus() override;

        Eigen::Vector3d mea_velo, mea_pos;
        double dt;
        Eigen::Vector3d g, gI;
    };

    /// only inertial edge
    class EdgeInertialOnly : public g2o::BaseMultiEdge<9, Vector9d> {
    public:
        EdgeInertialOnly(const Pose &Twb1, const Pose &Twb2, const std::shared_ptr<PreIntegrator> &preIntegrator);

        bool read(std::istream &is) override { return true; }

        bool write(std::ostream &os) const override { return true; }

        void computeError() override;

        void linearizeOplus() override;

        const Eigen::Matrix3d Rb1w, Rwb2;
        const Eigen::Vector3d twb1, twb2;
        const Eigen::Matrix3d JRg, JVg, JPg;
        const Eigen::Matrix3d JVa, JPa;
        std::shared_ptr<PreIntegrator> pre_integrator;
        const double dt;
        Eigen::Vector3d g;
    };

    /// inertial edge
    class EdgeInertial : public g2o::BaseMultiEdge<9, Vector9d> {
    public:
        explicit EdgeInertial(const std::shared_ptr<PreIntegrator> &preIntegrator);

        bool read(std::istream &is) override { return true; }

        bool write(std::ostream &os) const override { return true; }

        void computeError() override;

        void linearizeOplus() override;

        const Eigen::Matrix3d JRg, JVg, JPg;
        const Eigen::Matrix3d JVa, JPa;
        std::shared_ptr<PreIntegrator> pre_integrator;
        const double dt;
        Eigen::Vector3d g;
    };

    /// imu bias walk edge
    class EdgeBiasWalk : public g2o::BaseBinaryEdge<3, Eigen::Vector3d, Vertex3D, Vertex3D> {
    public:
        EdgeBiasWalk() = default;

        bool read(std::istream &is) override { return true; }

        bool write(std::ostream &os) const override { return true; }

        void computeError() override {
            const auto *vBias1 = dynamic_cast<const Vertex3D *>(_vertices[0]);
            const auto *vBias2 = dynamic_cast<const Vertex3D *>(_vertices[1]);
            _error = vBias2->estimate() - vBias1->estimate();
        }

        void linearizeOplus() override {
            _jacobianOplusXi = -Eigen::Matrix3d::Identity();
            _jacobianOplusXj.setIdentity();
        }

        Eigen::Matrix<double, 6, 6> getHessian() {
            linearizeOplus();
            Eigen::Matrix<double, 3, 6> J;
            J.block<3, 3>(0, 0) = _jacobianOplusXi;
            J.block<3, 3>(0, 3) = _jacobianOplusXj;
            return J.transpose() * information() * J;
        }

        Eigen::Matrix3d getHessian2() {
            linearizeOplus();
            return _jacobianOplusXj.transpose() * information() * _jacobianOplusXj;
        }
    };

} // mono_orb_slam3


#endif //MONO_ORB_SLAM3_G2OTYPES_H
