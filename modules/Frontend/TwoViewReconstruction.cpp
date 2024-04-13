//
// Created by whitby on 8/22/23.
//

#include "TwoViewReconstruction.h"
#include "Log/Logger.h"

#include <thread>
#include <random>

using namespace std;

namespace mono_orb_slam3 {
    bool TwoViewReconstruction::Reconstruct(const std::vector<cv::KeyPoint> &keyPoints1,
                                            const std::vector<cv::KeyPoint> &keyPoints2,
                                            const std::vector<int> &matches12,
                                            Eigen::Matrix3f &R21, Eigen::Vector3f &t21,
                                            std::vector<cv::Point3f> &points3D,
                                            std::vector<bool> &vecBeTriangulated) {
        key_points1.clear(), key_points2.clear();
        key_points1 = keyPoints1, key_points2 = keyPoints2;

        match_pairs.clear();
        match_pairs.reserve(key_points2.size());

        for (size_t i = 0; i < matches12.size(); ++i) {
            if (matches12[i] >= 0) {
                match_pairs.emplace_back(i, matches12[i]);
            }
        }

        num_matches = (int) match_pairs.size();

        // Indices for minimum set selection
        vector<size_t> allIndices;
        allIndices.reserve(num_matches);
        vector<size_t> availableIndices;

        for (int i = 0; i < num_matches; ++i)
            allIndices.push_back(i);

        // generate sets of 8 points for each RANSAC iteration
        vec_eight_points = vector<vector<size_t>>(max_iterations, vector<size_t>(8, 0));
        cv::RNG rng;
        for (int iter = 0; iter < max_iterations; ++iter) {
            availableIndices = allIndices;

            // Select a minimum set
            for (size_t j = 0; j < 8; ++j) {
                int randi = rng.uniform(0, (int) availableIndices.size());
                size_t idx = availableIndices[randi];

                vec_eight_points[iter][j] = idx;

                availableIndices[randi] = availableIndices.back();
                availableIndices.pop_back();
            }
        }

        // launch threads to compute in parallel a fundamental matrix and a homography
        vector<bool> beInlierMatchesH, beInlierMatchesF;
        float SH, SF;
        Eigen::Matrix3f H, F;

        thread threadH(&TwoViewReconstruction::FindHomography, this, ref(beInlierMatchesH), ref(SH), ref(H));
        thread threadF(&TwoViewReconstruction::FindFundamental, this, ref(beInlierMatchesF), ref(SF), ref(F));

        // wait until both thread have finished
        threadH.join();
        threadF.join();

        // compute ratio of scores
        if (SH + SF == 0.f) return false;
        float RH = SH / (SH + SF);
        float minParallax = 1.f;

        initial_logger << titles[0] << "launch two thread to compute F and H in parallel, RH = " << RH << "\n";

        if (RH > 0.5) {
            return ReconstructH(H, beInlierMatchesH, R21, t21, points3D, vecBeTriangulated, minParallax);
        } else {
            return ReconstructF(F, beInlierMatchesF, R21, t21, points3D, vecBeTriangulated, minParallax);
        }
    }

    void TwoViewReconstruction::FindHomography(std::vector<bool> &beInlierMatches, float &score, Eigen::Matrix3f &H21) {
        // normalize coordinates
        vector<cv::Point2f> points1, points2;
        Eigen::Matrix3f T1, T2;
        Normalize(key_points1, points1, T1);
        Normalize(key_points2, points2, T2);

        // best results variables
        score = 0.f;
        beInlierMatches = vector<bool>(num_matches, false);

        // iteration variables
        vector<cv::Point2f> iterPoints1(8), iterPoints2(8);
        Eigen::Matrix3f iterH21;
        vector<bool> iterBeInlierMatches(num_matches, false);
        float iterScore;

        // perform all RANSAC iterations and save the solution with the highest score
        for (int iter = 0; iter < max_iterations; ++iter) {
            for (int k = 0; k < 8; ++k) {
                size_t idx = vec_eight_points[iter][k];
                iterPoints1[k] = points1[match_pairs[idx].first];
                iterPoints2[k] = points2[match_pairs[idx].second];
            }

            Eigen::Matrix3f Hn = ComputeH21(iterPoints1, iterPoints2);
            iterH21 = T2.inverse() * Hn * T1;

            iterScore = CheckHomography(iterH21, iterBeInlierMatches);

            if (iterScore > score) {
                H21 = iterH21;
                beInlierMatches = iterBeInlierMatches;
                score = iterScore;
            }
        }
    }

    void
    TwoViewReconstruction::FindFundamental(std::vector<bool> &beInlierMatches, float &score, Eigen::Matrix3f &F21) {
        // normalize coordinates
        vector<cv::Point2f> points1, points2;
        Eigen::Matrix3f T1, T2;
        Normalize(key_points1, points1, T1);
        Normalize(key_points2, points2, T2);

        // best results variables
        score = 0.f;
        beInlierMatches = vector<bool>(num_matches, false);

        // iteration variables
        vector<cv::Point2f> iterPoints1(8), iterPoints2(8);
        Eigen::Matrix3f iterF21;
        vector<bool> iterBeInlierMatches(num_matches, false);
        float iterScore;

        // perform all RANSAC iterations and save the solution with the highest score
        for (int iter = 0; iter < max_iterations; ++iter) {
            for (int k = 0; k < 8; ++k) {
                size_t idx = vec_eight_points[iter][k];
                iterPoints1[k] = points1[match_pairs[idx].first];
                iterPoints2[k] = points2[match_pairs[idx].second];
            }

            Eigen::Matrix3f Fn = ComputeF21(iterPoints1, iterPoints2);
            iterF21 = T2.transpose() * Fn * T1;

            iterScore = CheckFundamental(iterF21, iterBeInlierMatches);

            if (iterScore > score) {
                F21 = iterF21;
                beInlierMatches = iterBeInlierMatches;
                score = iterScore;
            }
        }
    }

    Eigen::Matrix3f TwoViewReconstruction::ComputeH21(const std::vector<cv::Point2f> &points1,
                                                      const std::vector<cv::Point2f> &points2) {
        const int N = (int) points1.size();
        Eigen::MatrixXf A(2 * N, 9);

        for (int i = 0; i < N; ++i) {
            const float u1 = points1[i].x, v1 = points1[i].y;
            const float u2 = points2[i].x, v2 = points2[i].y;

            A(2 * i, 0) = 0.f;
            A(2 * i, 1) = 0.f;
            A(2 * i, 2) = 0.f;
            A(2 * i, 3) = -u1;
            A(2 * i, 4) = -v1;
            A(2 * i, 5) = -1;
            A(2 * i, 6) = v2 * u1;
            A(2 * i, 7) = v2 * v1;
            A(2 * i, 8) = v2;

            A(2 * i + 1, 0) = u1;
            A(2 * i + 1, 1) = v1;
            A(2 * i + 1, 2) = 1.f;
            A(2 * i + 1, 3) = 0.f;
            A(2 * i + 1, 4) = 0.f;
            A(2 * i + 1, 5) = 0.f;
            A(2 * i + 1, 6) = -u2 * u1;
            A(2 * i + 1, 7) = -u2 * v1;
            A(2 * i + 1, 8) = -u2;
        }

        Eigen::JacobiSVD<Eigen::MatrixXf> svd(A, Eigen::ComputeFullV);
        Eigen::Matrix<float, 3, 3, Eigen::RowMajor> H(svd.matrixV().col(8).data());
        return H;
    }

    Eigen::Matrix3f TwoViewReconstruction::ComputeF21(const std::vector<cv::Point2f> &points1,
                                                      const std::vector<cv::Point2f> &points2) {
        const int N = (int) points1.size();
        Eigen::MatrixXf A(N, 9);

        for (int i = 0; i < N; ++i) {
            const float u1 = points1[i].x, v1 = points1[i].y;
            const float u2 = points2[i].x, v2 = points2[i].y;

            A(i, 0) = u2 * u1;
            A(i, 1) = u2 * v1;
            A(i, 2) = u2;
            A(i, 3) = v2 * u1;
            A(i, 4) = v2 * v1;
            A(i, 5) = v2;
            A(i, 6) = u1;
            A(i, 7) = v1;
            A(i, 8) = 1;
        }

        Eigen::JacobiSVD<Eigen::MatrixXf> svd(A, Eigen::ComputeFullU | Eigen::ComputeFullV);
        Eigen::Matrix<float, 3, 3, Eigen::RowMajor> preF(svd.matrixV().col(8).data());
        Eigen::JacobiSVD<Eigen::Matrix3f> svd2(preF, Eigen::ComputeFullU | Eigen::ComputeFullV);
        Eigen::Vector3f w = svd2.singularValues();
        w(2) = 0;
        return svd2.matrixU() * Eigen::DiagonalMatrix<float, 3>(w) * svd2.matrixV().transpose();
    }

    float TwoViewReconstruction::CheckHomography(const Eigen::Matrix3f &H21, std::vector<bool> &beInlierMatches) {
        const Eigen::Matrix3f H12 = H21.inverse();

        const float h11 = H21(0, 0);
        const float h12 = H21(0, 1);
        const float h13 = H21(0, 2);
        const float h21 = H21(1, 0);
        const float h22 = H21(1, 1);
        const float h23 = H21(1, 2);
        const float h31 = H21(2, 0);
        const float h32 = H21(2, 1);
        const float h33 = H21(2, 2);

        const float h11inv = H12(0, 0);
        const float h12inv = H12(0, 1);
        const float h13inv = H12(0, 2);
        const float h21inv = H12(1, 0);
        const float h22inv = H12(1, 1);
        const float h23inv = H12(1, 2);
        const float h31inv = H12(2, 0);
        const float h32inv = H12(2, 1);
        const float h33inv = H12(2, 2);

        beInlierMatches.resize(num_matches);
        float score = 0;
        const float th = 5.991, invSigma2 = 1.f / sigma2;
        for (int i = 0; i < num_matches; ++i) {
            bool beInlier = true;
            const cv::KeyPoint &kp1 = key_points1[match_pairs[i].first];
            const cv::KeyPoint &kp2 = key_points2[match_pairs[i].second];

            const float u1 = kp1.pt.x, v1 = kp1.pt.y;
            const float u2 = kp2.pt.x, v2 = kp2.pt.y;

            // re-projection error in first image
            // x1' = H12 * x2;
            const float w2in1inv = 1.f / (h31inv * u2 + h32inv * v2 + h33inv);
            const float u2in1 = (h11inv * u2 + h12inv * v2 + h13inv) * w2in1inv;
            const float v2in1 = (h21inv * u2 + h22inv * v2 + h23inv) * w2in1inv;

            const float squareDist1 = (u1 - u2in1) * (u1 - u2in1) + (v1 - v2in1) * (v1 - v2in1);
            const float chiSquare1 = squareDist1 * invSigma2;

            if (chiSquare1 > th) beInlier = false;
            else score += th - chiSquare1;

            // re-projection error in second image
            // x2' = H21 * x1
            const float w1in2inv = 1.f / (h31 * u1 + h32 * v1 + h33);
            const float u1in2 = (h11 * u1 + h12 * v1 + h13) * w1in2inv;
            const float v1in2 = (h21 * u1 + h22 * v1 + h23) * w1in2inv;

            const float squareDist2 = (u2 - u1in2) * (u2 - u1in2) + (v2 - v1in2) * (v2 - v1in2);
            const float chiSquare2 = squareDist2 * invSigma2;

            if (chiSquare2 > th) beInlier = false;
            else score += th - chiSquare2;

            beInlierMatches[i] = beInlier;
        }

        return score;
    }

    float TwoViewReconstruction::CheckFundamental(const Eigen::Matrix3f &F21, std::vector<bool> &beInlierMatches) {
        const float f11 = F21(0, 0);
        const float f12 = F21(0, 1);
        const float f13 = F21(0, 2);
        const float f21 = F21(1, 0);
        const float f22 = F21(1, 1);
        const float f23 = F21(1, 2);
        const float f31 = F21(2, 0);
        const float f32 = F21(2, 1);
        const float f33 = F21(2, 2);

        beInlierMatches.resize(num_matches);
        float score = 0;
        const float th = 3.841, thScore = 5.991;
        const float invSigma2 = 1.f / sigma2;

        for (int i = 0; i < num_matches; ++i) {
            bool beInlier = true;
            const cv::KeyPoint &kp1 = key_points1[match_pairs[i].first];
            const cv::KeyPoint &kp2 = key_points2[match_pairs[i].second];

            const float u1 = kp1.pt.x, v1 = kp1.pt.y;
            const float u2 = kp2.pt.x, v2 = kp2.pt.y;

            // re-projection error in second image
            // l2 = F12 * x1 = (a2, b2, c2)
            const float a2 = f11 * u1 + f12 * v1 + f13;
            const float b2 = f21 * u1 + f22 * v1 + f23;
            const float c2 = f31 * u1 + f32 * v1 + f33;

            const float num2 = a2 * u2 + b2 * v2 + c2;
            const float squareDist1 = num2 * num2 / (a2 * a2 + b2 * b2);
            const float chiSquare1 = squareDist1 * invSigma2;

            if (chiSquare1 > th) beInlier = false;
            else score += thScore - chiSquare1;

            // re-projection error in first image
            // l1 = x2^T * F21 = (a1, b1, c1)
            const float a1 = f11 * u2 + f21 * v2 + f31;
            const float b1 = f12 * u2 + f22 * v2 + f32;
            const float c1 = f13 * u2 + f23 * v2 + f33;

            const float num1 = a1 * u1 + b1 * v1 + c1;
            const float squareDist2 = num1 * num1 / (a1 * a1 + b1 * b1);
            const float chiSquare2 = squareDist2 * invSigma2;

            if (chiSquare2 > th) beInlier = false;
            else score += thScore - chiSquare2;

            beInlierMatches[i] = beInlier;
        }

        return score;
    }

    bool
    TwoViewReconstruction::ReconstructH(Eigen::Matrix3f &H21, std::vector<bool> &beInlierMatches, Eigen::Matrix3f &R21,
                                        Eigen::Vector3f &t21, std::vector<cv::Point3f> &points3D,
                                        std::vector<bool> &vecBeTriangulated, const float &minParallax) {
        int nInlier = 0;
        for (auto beInlier: beInlierMatches)
            if (beInlier) nInlier++;
        int minGood = min(cvRound(0.6 * nInlier), 100);

        initial_logger << titles[0] << "ReconstructH: " << nInlier << " inlier matches, minGood = " << minGood
                       << ", minParallax = " << minParallax << "\n";

        // we recover 8 motion hypotheses using the method of Faugeras et al.
        // motion and structure from motion in a piecewise planar environment.
        Eigen::Matrix3f invK = K.inverse();
        Eigen::Matrix3f A = invK * H21 * K;

        Eigen::JacobiSVD<Eigen::Matrix3f> svd(A, Eigen::ComputeFullU | Eigen::ComputeFullV);
        Eigen::Matrix3f U = svd.matrixU(), V = svd.matrixV();
        Eigen::Matrix3f Vt = V.transpose();
        Eigen::Vector3f w = svd.singularValues();

        float s = U.determinant() * Vt.determinant();
        float d1 = w(0), d2 = w(1), d3 = w(2);
        if (d1 / d2 < 1.00001 || d2 / d3 < 1.00001) return false;

        vector<Eigen::Matrix3f> vR(8);
        vector<Eigen::Vector3f> vt(8), vn(8);

        // n'=[x1 0 x3] 4 possibilities e1=e3=1, e1=1 e3=-1, e1=-1 e3=1, e1=e3=-1
        float aux1 = sqrt((d1 * d1 - d2 * d2) / (d1 * d1 - d3 * d3));
        float aux3 = sqrt((d2 * d2 - d3 * d3) / (d1 * d1 - d3 * d3));
        float x1[] = {aux1, aux1, -aux1, -aux1};
        float x3[] = {aux3, -aux3, aux3, -aux3};

        // case d'=d2
        float aux_s_theta = sqrt((d1 * d1 - d2 * d2) * (d2 * d2 - d3 * d3)) / ((d1 + d3) * d2);
        float c_theta = (d2 * d2 + d1 * d3) / ((d1 + d3) * d2);
        float s_theta[] = {aux_s_theta, -aux_s_theta, -aux_s_theta, aux_s_theta};

        for (int i = 0; i < 4; ++i) {
            Eigen::Matrix3f Rp = Eigen::Matrix3f::Zero();
            Rp(0, 0) = c_theta;
            Rp(0, 2) = -s_theta[i];
            Rp(1, 1) = 1.f;
            Rp(2, 0) = s_theta[i];
            Rp(2, 2) = c_theta;

            Eigen::Matrix3f R = s * U * Rp * Vt;
            vR.push_back(R);

            Eigen::Vector3f tp(x1[i], 0, -x3[i]);
            tp *= d1 - d3;

            Eigen::Vector3f t = U * tp;
            vt.emplace_back(t / t.norm());

            Eigen::Vector3f np(x1[i], 0, x3[i]);
            Eigen::Vector3f n = V * np;
            if (n(2) < 0) n = -n;
            vn.push_back(n);
        }

        // case d' = -d2
        float aux_s_phi = sqrt((d1 * d1 - d2 * d2) * (d2 * d2 - d3 * d3)) / ((d1 - d3) * d2);

        float c_phi = (d1 * d3 - d2 * d2) / ((d1 - d3) * d2);
        float s_phi[] = {aux_s_phi, -aux_s_phi, -aux_s_phi, aux_s_phi};

        for (int i = 0; i < 4; ++i) {
            Eigen::Matrix3f Rp = Eigen::Matrix3f::Zero();
            Rp(0, 0) = c_phi;
            Rp(0, 2) = s_phi[i];
            Rp(1, 1) = -1;
            Rp(2, 0) = s_phi[i];
            Rp(2, 2) = -c_phi;

            Eigen::Matrix3f R = s * U * Rp * Vt;
            vR.push_back(R);

            Eigen::Vector3f tp(x1[i], 0, x3[i]);
            tp *= d1 + d3;

            Eigen::Vector3f t = U * tp;
            vt.emplace_back(t / t.norm());

            Eigen::Vector3f np(x1[i], 0, x3[i]);
            Eigen::Vector3f n = V * np;
            if (n(2) < 0) n = -n;
            vn.push_back(n);
        }

        initial_logger << titles[0] << "compute 8 pose hypotheses, check them: \n";

        int bestGood = 0, secondBestGood = -1;
        int bestIdx = -1;
        float bestParallax = -1;
        vector<cv::Point3f> bestPoints3D;
        vector<bool> bestTriangulated;

        // we reconstruct all hypotheses and check in terms of triangulated points and parallax
        for (int iter = 0; iter < 8; ++iter) {
            float iterParallax;
            vector<cv::Point3f> iterPoints3D;
            vector<bool> iterVecBeTriangulated;
            int nGood = CheckRT(vR[iter], vt[iter], beInlierMatches, iterPoints3D, 4.f * sigma2, iterVecBeTriangulated,
                                iterParallax);

            if (nGood > bestGood) {
                secondBestGood = bestGood;
                bestGood = nGood;
                bestIdx = iter;
                bestParallax = iterParallax;
                bestPoints3D = iterPoints3D;
                bestTriangulated = iterVecBeTriangulated;
            } else if (nGood > secondBestGood) {
                secondBestGood = nGood;
            }
        }

        if (secondBestGood < 0.75 * bestGood && bestParallax >= minParallax && bestGood > minGood) {
            R21 = vR[bestIdx], t21 = vt[bestIdx];
            vecBeTriangulated = bestTriangulated;
            points3D = bestPoints3D;
            return true;
        }

        initial_logger << titles[0] << "reconstruct fail\n";
        return false;
    }

    bool
    TwoViewReconstruction::ReconstructF(Eigen::Matrix3f &F21, std::vector<bool> &beInlierMatches, Eigen::Matrix3f &R21,
                                        Eigen::Vector3f &t21, std::vector<cv::Point3f> &points3D,
                                        std::vector<bool> &vecBeTriangulated, const float &minParallax) {
        int nInlier = 0;
        for (auto beInlier: beInlierMatches)
            if (beInlier) nInlier++;
        int minGood = min(cvFloor(0.6 * nInlier), 100);

        initial_logger << titles[0] << "ReconstructF: " << nInlier << " inlier matches, minGood = " << minGood
                       << ", minParallax = " << minParallax << "\n";

        // compute essential matrix from F
        Eigen::Matrix3f E21 = K.transpose() * F21 * K;
        Eigen::Matrix3f R1, R2;
        Eigen::Vector3f t;

        // recover 4 motion hypotheses
        DecomposeE(E21, R1, R2, t);
        Eigen::Vector3f t1 = t;
        Eigen::Vector3f t2 = -t;

        initial_logger << titles[0] << "compute 4 pose hypotheses, check them: \n";
        // reconstruct with 4 hypotheses and check
        vector<cv::Point3f> points3D1, points3D2, points3D3, points3D4;
        vector<bool> vbTriangulated1, vbTriangulated2, vbTriangulated3, vbTriangulated4;
        float parallax1, parallax2, parallax3, parallax4;

        int nGood1 = CheckRT(R1, t1, beInlierMatches, points3D1, 4.f * sigma2, vbTriangulated1, parallax1);
        int nGood2 = CheckRT(R2, t1, beInlierMatches, points3D2, 4.f * sigma2, vbTriangulated2, parallax2);
        int nGood3 = CheckRT(R1, t2, beInlierMatches, points3D3, 4.f * sigma2, vbTriangulated3, parallax3);
        int nGood4 = CheckRT(R2, t2, beInlierMatches, points3D4, 4.f * sigma2, vbTriangulated4, parallax4);

        int maxGood = max(nGood1, max(nGood2, max(nGood3, nGood4)));
        int nSimilar = 0;
        int goodTh = cvFloor(0.7 * maxGood);
        if (nGood1 > goodTh) nSimilar++;
        if (nGood2 > goodTh) nSimilar++;
        if (nGood3 > goodTh) nSimilar++;
        if (nGood4 > goodTh) nSimilar++;

        // if there is not a clear winner or not enough triangulated points reject initialization
        if (maxGood < minGood || nSimilar > 1) {
            return false;
        }

        // if best reconstruction has enough parallax initialize
        if (maxGood == nGood1) {
            if (parallax1 > minParallax) {
                points3D = points3D1;
                vecBeTriangulated = vbTriangulated1;
                R21 = R1, t21 = t1;
                return true;
            }
        } else if (maxGood == nGood2) {
            if (parallax2 > minParallax) {
                points3D = points3D2;
                vecBeTriangulated = vbTriangulated2;
                R21 = R2, t21 = t1;
                return true;
            }
        } else if (maxGood == nGood3) {
            if (parallax3 > minParallax) {
                points3D = points3D3;
                vecBeTriangulated = vbTriangulated3;
                R21 = R1, t21 = t2;
                return true;
            }
        } else {
            if (parallax4 > minParallax) {
                points3D = points3D4;
                vecBeTriangulated = vbTriangulated4;
                R21 = R2, t21 = t2;
                return true;
            }
        }

        initial_logger << titles[0] << "reconstruct fail\n";
        return false;
    }

    void TwoViewReconstruction::Normalize(const std::vector<cv::KeyPoint> &keyPoints,
                                          std::vector<cv::Point2f> &normalize_points,
                                          Eigen::Matrix3f &T) {
        float meanX = 0, meanY = 0;
        const int N = (int) keyPoints.size();
        normalize_points.resize(N);

        for (auto &kp: keyPoints) {
            meanX += kp.pt.x;
            meanY += kp.pt.y;
        }

        meanX = meanX / N;
        meanY = meanY / N;

        float meanDevX = 0, meanDevY = 0;
        for (int i = 0; i < N; ++i) {
            normalize_points[i].x = keyPoints[i].pt.x - meanX;
            normalize_points[i].y = keyPoints[i].pt.y - meanY;

            meanDevX += fabs(normalize_points[i].x);
            meanDevY += fabs(normalize_points[i].y);
        }

        meanDevX = meanDevX / N;
        meanDevY = meanDevY / N;

        float sX = 1.f / meanDevX;
        float sY = 1.f / meanDevY;

        for (int i = 0; i < N; ++i) {
            normalize_points[i].x *= sX;
            normalize_points[i].y *= sY;
        }

        T.setIdentity();
        T(0, 0) = sX, T(1, 1) = sY;
        T(0, 2) = -meanX * sX, T(1, 2) = -meanY * sY;
    }

    int TwoViewReconstruction::CheckRT(const Eigen::Matrix3f &R21, const Eigen::Vector3f &t21,
                                       std::vector<bool> &beInlierMatches,
                                       std::vector<cv::Point3f> &points3D, float th2, std::vector<bool> &vbGood,
                                       float &parallax) {
        // calibration parameters
        const float fx = K(0, 0), cx = K(0, 2);
        const float fy = K(1, 1), cy = K(1, 2);

        vbGood = vector<bool>(key_points1.size(), false);
        points3D.resize(key_points1.size());
        vector<float> vecCosParallax;
        vecCosParallax.reserve(num_matches);

        // camera 1 projection matrix K[I|0]
        Eigen::Matrix<float, 3, 4> P1;
        P1.setZero();
        P1.block<3, 3>(0, 0) = K;

        Eigen::Vector3f O1 = Eigen::Vector3f::Zero();

        // camera 2 projection matrix K[R|t]
        Eigen::Matrix<float, 3, 4> P2;
        P2.block<3, 3>(0, 0) = R21;
        P2.block<3, 1>(0, 3) = t21;
        P2 = K * P2;

        Eigen::Vector3f O2 = -R21.transpose() * t21;

        int nGood = 0;
        for (int i = 0, end = (int) match_pairs.size(); i < end; ++i) {
            if (!beInlierMatches[i]) continue;

            const cv::KeyPoint &kp1 = key_points1[match_pairs[i].first];
            const cv::KeyPoint &kp2 = key_points2[match_pairs[i].second];

            Eigen::Vector3f Pc1, p1(kp1.pt.x, kp1.pt.y, 1), p2(kp2.pt.x, kp2.pt.y, 1);
            Triangulate(p1, p2, P1, P2, Pc1);

            if (!isfinite(Pc1(0)) || !isfinite(Pc1(1)) || !isfinite(Pc1(2))) {
                vbGood[match_pairs[i].first] = false;
                continue;
            }

            // check parallax
            Eigen::Vector3f n1 = Pc1 - O1;
            n1.normalize();
            Eigen::Vector3f n2 = Pc1 - O2;
            n2.normalize();
            const float cosParallax = n1.dot(n2);

            // check in front of first camera (only if enough parallax, as "infinite" points can easily go to negative depth)
            if (Pc1(2) <= 0 && cosParallax < 0.99998) continue;

            // check in front of second camera (only if enough parallax, as "infinite" points can easily go to negative depth)
            Eigen::Vector3f Pc2 = R21 * Pc1 + t21;
            if (Pc2(2) <= 0 && cosParallax < 0.99998) continue;

            // check re-projection error in first image
            float invZ1 = 1.f / Pc1(2);
            float proj_u1 = fx * Pc1(0) * invZ1 + cx;
            float proj_v1 = fy * Pc1(1) * invZ1 + cy;
            float squareError1 =
                    (kp1.pt.x - proj_u1) * (kp1.pt.x - proj_u1) + (kp1.pt.y - proj_v1) * (kp1.pt.y - proj_v1);
            if (squareError1 > th2) continue;

            // check re-projection error in second image
            float invZ2 = 1.f / Pc2(2);
            float proj_u2 = fx * Pc2(0) * invZ2 + cx;
            float proj_v2 = fy * Pc2(1) * invZ2 + cy;
            float squareError2 =
                    (kp2.pt.x - proj_u2) * (kp2.pt.x - proj_u2) + (kp2.pt.y - proj_v2) * (kp2.pt.y - proj_v2);
            if (squareError2 > th2) continue;

            vecCosParallax.push_back(cosParallax);
            points3D[match_pairs[i].first] = cv::Point3f(Pc1(0), Pc1(1), Pc1(2));
            nGood++;

            if (cosParallax < 0.99998)
                vbGood[match_pairs[i].first] = true;
        }

        if (nGood > 0) {
            sort(vecCosParallax.begin(), vecCosParallax.end());
            int idx = min(50, (int) vecCosParallax.size() - 1);
            parallax = acosf(vecCosParallax[idx]) * 180 / M_PIf32;
        } else parallax = 0;

        initial_logger << titles[0] << " -have " << nGood << " good matches, parallax = " << parallax << "\n";
        return nGood;
    }

    bool TwoViewReconstruction::Triangulate(Eigen::Vector3f &p1, Eigen::Vector3f &p2, Eigen::Matrix<float, 3, 4> &P1,
                                            Eigen::Matrix<float, 3, 4> &P2, Eigen::Vector3f &P) {
        Eigen::Matrix4f A;
        A.block<1, 4>(0, 0) = p1(0) * P1.block<1, 4>(2, 0) - P1.block<1, 4>(0, 0);
        A.block<1, 4>(1, 0) = p1(1) * P1.block<1, 4>(2, 0) - P1.block<1, 4>(1, 0);
        A.block<1, 4>(2, 0) = p2(0) * P2.block<1, 4>(2, 0) - P2.block<1, 4>(0, 0);
        A.block<1, 4>(3, 0) = p2(1) * P2.block<1, 4>(2, 0) - P2.block<1, 4>(1, 0);

        Eigen::JacobiSVD<Eigen::Matrix4f> svd(A, Eigen::ComputeFullV);
        Eigen::Vector4f Ph = svd.matrixV().col(3);
        if (Ph(3) == 0) return false;

        // euclidean coordinates
        P = Ph.head(3) / Ph(3);
        return true;
    }

    void TwoViewReconstruction::DecomposeE(const Eigen::Matrix3f &E, Eigen::Matrix3f &R1, Eigen::Matrix3f &R2,
                                           Eigen::Vector3f &t) {
        Eigen::JacobiSVD<Eigen::Matrix3f> svd(E, Eigen::ComputeFullU | Eigen::ComputeFullV);
        Eigen::Matrix3f U = svd.matrixU();
        Eigen::Matrix3f Vt = svd.matrixV().transpose();

        t = U.col(2);
        t.normalize();

        Eigen::Matrix3f W = Eigen::Matrix3f::Zero();
        W(0, 1) = -1, W(1, 0) = 1, W(2, 2) = 1;

        R1 = U * W * Vt;
        if (R1.determinant() < 0) R1 = -R1;

        R2 = U * W.transpose() * Vt;
        if (R2.determinant() < 0) R2 = -R2;
    }

} // mono_orb_slam3