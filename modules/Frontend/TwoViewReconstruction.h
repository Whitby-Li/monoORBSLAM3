//
// Created by whitby on 8/22/23.
//

#ifndef MONO_ORB_SLAM3_TWOVIEWRECONSTRUCTION_H
#define MONO_ORB_SLAM3_TWOVIEWRECONSTRUCTION_H

#include <vector>
#include <Eigen/Dense>
#include <opencv2/core/core.hpp>

namespace mono_orb_slam3 {

    class TwoViewReconstruction {
        typedef std::pair<int, int> Match;

    public:
        explicit TwoViewReconstruction(Eigen::Matrix3f cameraMatrix, float sigma_ = 1.f, int iterations = 200)
                : K(std::move(cameraMatrix)), sigma(sigma_), max_iterations(iterations) {
            sigma2 = sigma * sigma;
        }

        // Compute in parallel a fundamental matrix and a homography
        // Select a model and try to recover the pose and structure from motion
        bool Reconstruct(const std::vector<cv::KeyPoint> &keyPoints1, const std::vector<cv::KeyPoint> &keyPoints2,
                         const std::vector<int> &matches12, Eigen::Matrix3f &R21, Eigen::Vector3f &t21,
                         std::vector<cv::Point3f> &points3D, std::vector<bool> &vecBeTriangulated);

        static bool Triangulate(const Eigen::Vector3f &x1, const Eigen::Vector3f &x2, const Eigen::Matrix<float, 3, 4> &P1,
                                const Eigen::Matrix<float, 3, 4> &P2, Eigen::Vector3f &P);

    private:
        // Have 200 eight-point pairs, compute homography matrix and score it iteratively, retain the best
        void FindHomography(std::vector<bool> &beInlierMatches, float &score, Eigen::Matrix3f &H21);

        // Have 200 eight-point pairs, compute fundamental matrix and score it iteratively, retain the best
        void FindFundamental(std::vector<bool> &beInlierMatches, float &score, Eigen::Matrix3f &F21);

        // Have a group matched points (default size 8), compute H21
        Eigen::Matrix3f ComputeH21(const std::vector<cv::Point2f> &points1, const std::vector<cv::Point2f> &points2);

        // Have a group matched points (default size 8), compute F21
        Eigen::Matrix3f ComputeF21(const std::vector<cv::Point2f> &points1, const std::vector<cv::Point2f> &points2);

        float CheckHomography(const Eigen::Matrix3f &H21, std::vector<bool> &beInlierMatches);

        float CheckFundamental(const Eigen::Matrix3f &F21, std::vector<bool> &beInlierMatches);

        bool ReconstructH(Eigen::Matrix3f &H21, std::vector<bool> &beInlierMatches, Eigen::Matrix3f &R21,
                          Eigen::Vector3f &t21, std::vector<cv::Point3f> &points3D,
                          std::vector<bool> &vecBeTriangulated, const float &minParallax);

        bool ReconstructF(Eigen::Matrix3f &F21, std::vector<bool> &beInlierMatches, Eigen::Matrix3f &R21,
                          Eigen::Vector3f &t21, std::vector<cv::Point3f> &points3D,
                          std::vector<bool> &vecBeTriangulated, const float &minParallax);

        // T * p = p_normalized
        static void Normalize(const std::vector<cv::KeyPoint> &keyPoints, std::vector<cv::Point2f> &normalize_points,
                              Eigen::Matrix3f &T);

        int CheckRT(const Eigen::Matrix3f &R21, const Eigen::Vector3f &t21, std::vector<bool> &beInlierMatches,
                    std::vector<cv::Point3f> &points3D, float th2, std::vector<bool> &vbGood, float &parallax);

        static void DecomposeE(const Eigen::Matrix3f &E, Eigen::Matrix3f &R1, Eigen::Matrix3f &R2, Eigen::Vector3f &t);

        // key-points from reference frame (frame1)
        std::vector<cv::KeyPoint> key_points1;
        // key-points from current frame (frame2)
        std::vector<cv::KeyPoint> key_points2;

        // current matches from reference to current
        std::vector<Match> match_pairs;
        int num_matches = 0;

        // camera calibration
        const Eigen::Matrix3f K;

        // standard deviation and variance
        float sigma, sigma2;

        // RANSAC max iterations
        int max_iterations;

        // RANSAC sets
        std::vector<std::vector<size_t>> vec_eight_points;
    };

} // mono_orb_slam3

#endif //MONO_ORB_SLAM3_TWOVIEWRECONSTRUCTION_H
