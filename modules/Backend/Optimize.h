//
// Created by whitby on 8/22/23.
//

#ifndef MONO_ORB_SLAM3_OPTIMIZE_H
#define MONO_ORB_SLAM3_OPTIMIZE_H

#include "BasicObject/KeyFrame.h"
#include "BasicObject/Map.h"

#include <g2o/core/block_solver.h>
#include <g2o/solvers/eigen/linear_solver_eigen.h>

namespace mono_orb_slam3 {

    class Optimize {
        typedef g2o::BlockSolver<g2o::BlockSolverTraits<6, 2>> BlockSolver_6_2;
        typedef g2o::LinearSolverEigen<g2o::BlockSolver_6_3::PoseMatrixType> LinearSolverEigen_6_3;
        typedef g2o::LinearSolverEigen<BlockSolver_6_2 ::PoseMatrixType> LinearSolverEigen_6_2;
        typedef g2o::LinearSolverEigen<g2o::BlockSolverX::PoseMatrixType> LinearSolverEigenX;

    public:
        /// Initialize
        static void
        initialOptimize(const std::shared_ptr<KeyFrame> &lastKeyFrame, const std::shared_ptr<KeyFrame> &curKeyFrame);

        static void inertialOptimize(Map *pointMap, Eigen::Matrix3d &Rwg, double &scale, float prioriG, float prioriA, bool beFirst);

        static void gravityOptimize(Map *pointMap, Eigen::Matrix3d &Rwg);

        static void fullInertialOptimize(Map *pointMap, int iteration, bool beInit, bool fixedMP, float prioriG, float prioriA);

        static int poseOptimize(const std::shared_ptr<Frame> &frame);

        static void poseInertialOptimize(const std::shared_ptr<KeyFrame> &lastKF, const std::shared_ptr<Frame> &frame);

        static int poseFullOptimize(const std::shared_ptr<KeyFrame> &lastKF, const std::shared_ptr<Frame> &frame);

        static void localBundleAdjustment(const std::shared_ptr<KeyFrame> &keyFrame, Map *pointMap, bool *stopFlag = nullptr);

        static void localInertialBundleAdjustment(const std::shared_ptr<KeyFrame> &keyFrame, Map *pointMap, bool beLarge);

        static void localFullBundleAdjustment(const std::shared_ptr<KeyFrame> &keyFrame, Map *pointMap, bool beLarge, bool *stopFlag = nullptr);
    };

} // mono_orb_slam3

#endif //MONO_ORB_SLAM3_OPTIMIZE_H
