//
// Created by whitby on 9/1/23.
//

#ifndef MONO_ORB_SLAM3_IMU_H
#define MONO_ORB_SLAM3_IMU_H

#include "BasicObject/Pose.h"

#include <Eigen/Dense>
#include <opencv2/core/core.hpp>
#include <utility>

namespace mono_orb_slam3 {
    const float GRAVITY_VALUE = 9.79;

    /// Imu measurement (accelerometer and timestamp)
    struct ImuData {
        ImuData(Eigen::Vector3f angel_velo, Eigen::Vector3f acceleration, double timestamp)
                : w(std::move(angel_velo)), a(std::move(acceleration)), t(timestamp) {}

        Eigen::Vector3f w, a;
        double t = 0;
    };

    /// Imu bias
    struct Bias {
        Eigen::Vector3f bg = Eigen::Vector3f::Zero();
        Eigen::Vector3f ba = Eigen::Vector3f::Zero();

        Bias() = default;

        Bias(Eigen::Vector3f gyro_bias, Eigen::Vector3f acc_bias) : bg(std::move(gyro_bias)), ba(std::move(acc_bias)) {}

        inline void setZero() {
            bg.setZero();
            ba.setZero();
        }

        Bias operator-(const Bias &bias) {
            return {bg - bias.bg, ba - bias.ba};
        }
    };

    /// Imu calibration (Tbc, Tcb, noise, bias_walk
    class ImuCalib {
    private:
        ImuCalib(Eigen::Matrix3f Rcb, Eigen::Vector3f tcb, const Eigen::DiagonalMatrix<float, 6> &covNoise,
                 const Eigen::DiagonalMatrix<float, 6> &covWalk, Bias bias)
                : T_cb(std::move(Rcb), std::move(tcb)), cov_noise(covNoise), cov_walk(covWalk),
                  initial_bias(std::move(bias)) {
            T_bc = T_cb.inverse();
            R_cb = T_cb.R.cast<double>(), R_bc = T_bc.R.cast<double>();
            t_cb = T_cb.t.cast<double>(), t_bc = T_bc.t.cast<double>();
        }

        // ImuCalib pointer
        static ImuCalib *imu_calib_ptr;

    public:
        ImuCalib() = delete;

        ImuCalib(const ImuCalib &imuCalib) = delete;

        const ImuCalib &operator=(const ImuCalib &imuCalib) = delete;

        // Create imu_calib by yaml file
        static bool create(const cv::FileNode &imuNode);

        // destroy imu
        static void destroy();

        // Get const pointer of imu_calib
        static const ImuCalib *getImuCalib();

        // Print imu calib information
        void print() const;

        // Imu parameters
        Pose T_cb, T_bc;
        Eigen::Matrix3d R_cb, R_bc;
        Eigen::Vector3d t_cb, t_bc;
        Eigen::DiagonalMatrix<float, 6> cov_noise, cov_walk;
        Bias initial_bias;
    };

    /// Imu pre-integration
    class PreIntegrator {
    public:
        PreIntegrator();

        explicit PreIntegrator(Bias bias);

        explicit PreIntegrator(const std::shared_ptr<PreIntegrator> &preIntegrator);

        PreIntegrator(const PreIntegrator &preIntegrator) = default;

        void Reset(const Bias &bias);

        void IntegrateNewMeasurement(const Eigen::Vector3f &gyro, const Eigen::Vector3f &acc, float dt);

        void ReIntegrate();

        void MergeNext(const std::shared_ptr<PreIntegrator> &preIntegrator);

        void setNewBias(const Bias &newBias);

        Eigen::Matrix3f getDeltaRotation(const Eigen::Vector3f &bias_gyro);

        Eigen::Vector3f getDeltaVelocity(const Eigen::Vector3f &bias_gyro, const Eigen::Vector3f &bias_acc);

        Eigen::Vector3f getDeltaPosition(const Eigen::Vector3f &bias_gyro, const Eigen::Vector3f &bias_acc);

        Eigen::Matrix3f getUpdatedDeltaRotation();

        Eigen::Vector3f getUpdatedDeltaVelocity();

        Eigen::Vector3f getUpdatedDeltaPosition();

        float delta_t = 0;
        Eigen::Matrix<float, 15, 15> C;
        Eigen::Matrix<float, 15, 15> information;   // information matrix

        Bias bias;
        Bias updated_bias;
        Bias delta_bias;

        Eigen::Matrix3f dR;
        Eigen::Vector3f dV;
        Eigen::Vector3f dP;
        Eigen::Matrix3f JRg, JVg, JVa, JPg, JPa;
        Eigen::DiagonalMatrix<float, 6> cov_noise, cov_walk;

        std::vector<ImuData> measurements;      // imu measurements

    private:
        inline void reset() {
            delta_t = 0;
            C.setZero(), information.setZero();
            dR.setIdentity(), dV.setZero(), dP.setZero();
            JRg.setZero();
            JVg.setZero(), JVa.setZero();
            JPg.setZero(), JPa.setZero();
            updated_bias = bias;
            delta_bias.setZero();
        }
    };
}

#endif //MONO_ORB_SLAM3_IMU_H
