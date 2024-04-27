//
// Created by whitby on 9/1/23.
//

#include "Imu.h"
#include "Utils/Converter.h"
#include "Utils/LieAlgeBra.h"

#include <iostream>

using namespace std;

namespace mono_orb_slam3 {
    ImuCalib *ImuCalib::imu_calib_ptr = nullptr;

    bool ImuCalib::create(const cv::FileNode &imuNode) {
        if (imu_calib_ptr == nullptr) {
            float noiseGyro, noiseAcc;
            float walkGyro, walkAcc;
            cv::Mat mat_Rcb, mat_tcb, mat_gyro_bias, mat_acc_bias;
            imuNode["NoiseGyro"] >> noiseGyro, imuNode["NoiseAcc"] >> noiseAcc;
            imuNode["WalkGyro"] >> walkGyro, imuNode["WalkAcc"] >> walkAcc;
            imuNode["GyroBias"] >> mat_gyro_bias, imuNode["AccBias"] >> mat_acc_bias;

            if (imuNode["Rcb"].empty()) {
                cv::Mat mat_Rbc, mat_tbc;
                imuNode["Rbc"] >> mat_Rbc, imuNode["tbc"] >> mat_tbc;
                mat_Rcb = mat_Rbc.t();
                mat_tcb = -mat_Rcb * mat_tbc;
            } else {
                imuNode["Rcb"] >> mat_Rcb, imuNode["tcb"] >> mat_tcb;
            }

            Eigen::Matrix3f Rbc;
            Eigen::Vector3f tcb;
            memcpy(Rbc.data(), mat_Rcb.data, sizeof(float) * 3 * 3);
            memcpy(tcb.data(), mat_tcb.data, sizeof(float) * 3);

            Eigen::DiagonalMatrix<float, 6> covNoise;
            covNoise.diagonal() << noiseGyro * noiseGyro, noiseGyro * noiseGyro, noiseGyro * noiseGyro,
                    noiseAcc * noiseAcc, noiseAcc * noiseAcc, noiseAcc * noiseAcc;

            Eigen::DiagonalMatrix<float, 6> covWalk;
            covWalk.diagonal() << walkGyro * walkGyro, walkGyro * walkGyro, walkGyro * walkGyro,
                    walkAcc * walkAcc, walkAcc * walkAcc, walkAcc * walkAcc;

            Bias initialBias({mat_gyro_bias.at<float>(0), mat_gyro_bias.at<float>(1), mat_gyro_bias.at<float>(2)},
                             {mat_acc_bias.at<float>(0), mat_acc_bias.at<float>(1), mat_acc_bias.at<float>(2)});

            imu_calib_ptr = new ImuCalib(Rbc.transpose(), tcb, covNoise, covWalk, initialBias);
            return true;
        } else {
            cerr << "imu_calib_ptr not null" << endl;
            return false;
        }
    }

    void ImuCalib::destroy() {
        delete imu_calib_ptr;
        imu_calib_ptr = nullptr;
    }

    const ImuCalib *ImuCalib::getImuCalib() {
        return imu_calib_ptr;
    }

    void ImuCalib::print() const {
        cout << endl << "Imu Parameters" << endl;
        cout << " - rotation from imu to camera: \n" << R_bc << endl;
        cout << " - translation from imu to camera: " << t_bc.transpose() << endl;
        cout << " - noise covariance: \n" << cov_noise.toDenseMatrix() << endl;
        cout << " - walk covariance: \n" << cov_walk.toDenseMatrix() << endl;
        cout << " - initial bias: " << initial_bias.bg.transpose() << " | " << initial_bias.ba.transpose() << endl;
    }

    PreIntegrator::PreIntegrator() {
        reset();
        const ImuCalib *imu_ptr = ImuCalib::getImuCalib();
        cov_noise = imu_ptr->cov_noise;
        cov_walk = imu_ptr->cov_walk;
    }

    PreIntegrator::PreIntegrator(Bias imuBias) : bias(std::move(imuBias)) {
        reset();
        const ImuCalib *imu_ptr = ImuCalib::getImuCalib();
        cov_noise = imu_ptr->cov_noise;
        cov_walk = imu_ptr->cov_walk;
    }

    PreIntegrator::PreIntegrator(const std::shared_ptr<PreIntegrator> &preIntegrator)
        : bias(preIntegrator->updated_bias), cov_noise(preIntegrator->cov_noise), cov_walk(preIntegrator->cov_walk) {
        reset();
    }

    void PreIntegrator::Reset(const Bias &imuBias) {
        bias = imuBias;
        reset();
        measurements.clear();
    }

    void PreIntegrator::IntegrateNewMeasurement(const Eigen::Vector3f &gyro, const Eigen::Vector3f &acc, float dt) {
        measurements.emplace_back(gyro, acc, dt);

        // matrices to compute covariance
        Eigen::Matrix<float, 9, 9> A = Eigen::Matrix<float, 9, 9>::Identity();
        Eigen::Matrix<float, 9, 6> B = Eigen::Matrix<float, 9, 6>::Zero();

        Eigen::Vector3f w(gyro - bias.bg), a(acc - bias.ba);
        float dt2 = dt * dt;

        // update dP and dV
        dP = dP + dV * dt + 0.5f * dR * a * dt2;
        dV = dV + dR * a * dt;

        // compute velocity and position parts of A and B
        Eigen::Matrix3f aHat = lie::Hatf(a);
        A.block<3, 3>(3, 0) = -dR * aHat * dt;
        A.block<3, 3>(6, 0) = -0.5f * dR * aHat * dt2;
        A.block<3, 3>(6, 3) = dt * Eigen::Matrix3f::Identity();
        B.block<3, 3>(3, 3) = dR * dt;
        B.block<3, 3>(6, 3) = 0.5f * dR * dt2;

        // update position and velocity jacobian wrt bias correction
        JPg = JPg + JVg * dt - 0.5f * dR * aHat * JRg * dt2;
        JPa = JPa + JVa * dt - 0.5 * dR * dt2;

        JVg = JVg - dR * aHat * JRg * dt;
        JVa = JVa - dR * dt;

        // update dR
        const Eigen::Vector3f delta_w = w * dt;
        const Eigen::Matrix3f deltaR = lie::ExpSO3f(delta_w);
        const Eigen::Matrix3f rightJ = lie::RightJacobianSO3f(delta_w);
        dR = lie::NormalizeRotationf(dR * deltaR);

        // compute rotation parts of matrices A and B
        A.block<3, 3>(0, 0) = deltaR.transpose();
        B.block<3, 3>(0, 0) = rightJ * dt;

        // update covariance
        C.block<9, 9>(0, 0) = A * C.block<9, 9>(0, 0) * A.transpose() + B * cov_noise * B.transpose();
        C.block<6, 6>(9, 9) += cov_walk;

        JRg = deltaR.transpose() * JRg - rightJ * dt;

        // total integrate time
        delta_t += dt;
    }

    void PreIntegrator::ReIntegrate() {
        const std::vector<ImuData> copyMeasurements = measurements;
        Reset(updated_bias);
        for (auto &m: copyMeasurements)
            IntegrateNewMeasurement(m.w, m.a, m.t);
    }

    void PreIntegrator::MergeNext(const std::shared_ptr<PreIntegrator> &preIntegrator) {
        if (preIntegrator.get() == this) return;
        const std::vector<ImuData> measurements1 = measurements;
        const std::vector<ImuData> measurements2 = preIntegrator->measurements;
        if (delta_bias.bg.norm() > 1e-5) {
            Reset(updated_bias);

            for (auto &m1: measurements1)
                IntegrateNewMeasurement(m1.w, m1.a, m1.t);
            for (auto &m2: measurements2)
                IntegrateNewMeasurement(m2.w, m2.a, m2.t);
        } else {
            for (auto &m2: measurements2)
                IntegrateNewMeasurement(m2.w, m2.a, m2.t);
        }
    }

    void PreIntegrator::setNewBias(const Bias &newBias) {
        updated_bias = newBias;
        delta_bias = updated_bias - bias;
        if (delta_bias.bg.norm() > 0.01) {
            ReIntegrate();
        }
    }

    Eigen::Matrix3f PreIntegrator::getDeltaRotation(const Eigen::Vector3f &bias_gyro) {
        return lie::NormalizeRotationf(dR * lie::ExpSO3f(JRg * (bias_gyro - bias.bg)));
    }

    Eigen::Vector3f PreIntegrator::getDeltaVelocity(const Eigen::Vector3f &bias_gyro, const Eigen::Vector3f &bias_acc) {
        return dV + JVg * (bias_gyro - bias.bg) + JVa * (bias_acc - bias.ba);
    }

    Eigen::Vector3f PreIntegrator::getDeltaPosition(const Eigen::Vector3f &bias_gyro, const Eigen::Vector3f &bias_acc) {
        return dP + JPg * (bias_gyro - bias.bg) + JPa * (bias_acc - bias.ba);
    }

    Eigen::Matrix3f PreIntegrator::getUpdatedDeltaRotation() {
        return lie::NormalizeRotationf(dR * lie::ExpSO3f(JRg * delta_bias.bg));
    }

    Eigen::Vector3f PreIntegrator::getUpdatedDeltaVelocity() {
        return dV + JVg * delta_bias.bg + JVa * delta_bias.ba;
    }

    Eigen::Vector3f PreIntegrator::getUpdatedDeltaPosition() {
        return dP + JPg * delta_bias.bg + JPa * delta_bias.ba;
    }
}