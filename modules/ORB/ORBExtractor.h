//
// Created by whitby on 8/22/23.
//

#ifndef MONO_ORB_SLAM3_ORBEXTRACTOR_H
#define MONO_ORB_SLAM3_ORBEXTRACTOR_H

#include <vector>
#include <list>
#include <opencv2/core/core.hpp>

namespace mono_orb_slam3 {
    typedef std::pair<unsigned int, unsigned int> Match;

    class ExtractorNode {
    public:
        ExtractorNode() : beNoMore(false) {}

        void DivideNode(ExtractorNode &n1, ExtractorNode &n2, ExtractorNode &n3, ExtractorNode &n4);

        std::vector <cv::KeyPoint> key_points;
        cv::Point2i UL, UR, BL, BR;
        std::list<ExtractorNode>::iterator iter;
        bool beNoMore;
    };

    class ORBExtractor {
    public:
        explicit ORBExtractor(int nFeatures = 1000, float scaleFactor = 1.2, int nLevels = 8, int iniThFast = 20,
                              int minThFast = 10);

        ORBExtractor(int nFeatures, const ORBExtractor &orbExtractor);

        ~ORBExtractor() = default;

        // Compute the Pyramid features and descriptors on an image
        // Pyramid are dispersed on the image using an octree
        void operator()(const cv::Mat &image, std::vector <cv::KeyPoint> &keyPoints,
                        cv::Mat &descriptors);

        // print pyramid information
        void print() const;

        inline static float getScaleFactor(int level = 0) {
            assert(level >= 0 && level < n_levels);
            return scale_factors[level];
        }

        inline static float getLogScaleFactor() {
            return log_sale_factor;
        }

        inline static float getMaxScaleFactor() {
            return scale_factors[n_levels - 1];
        }

        inline static std::vector<float> getScaleFactors() {
            return scale_factors;
        }

        inline static float getInvScaleFactor(int level) {
            assert(level >= 0 && level < n_levels);
            return inv_scale_factors[level];
        }

        inline static std::vector<float> getInvScaleFactors() {
            return inv_scale_factors;
        }

        inline static int getNumLevels() {
            return n_levels;
        }

        inline static std::vector<float> getSquareSigmas() {
            return square_sigmas;
        }

        inline static float getSquareSigma(int level) {
            assert(level >= 0 && level < n_levels);
            return square_sigmas[level];
        }

        inline static float getInvSquareSigma(int level) {
            assert(level >= 0 && level < n_levels);
            return inv_square_sigmas[level];
        }

        std::vector <cv::Mat> image_pyramid;

    protected:
        // compute image pyramid
        void ComputePyramid(const cv::Mat &image);

        // extract ORB features and distribute them with quadtree
        void ComputeKeyPointsOctTree(std::vector <std::vector<cv::KeyPoint>> &allKeyPoints);

        // distribute orb features with quadtree
        static std::vector <cv::KeyPoint>
        DistributeOctree(const std::vector <cv::KeyPoint> &vecToDistributeKPs, const int &minX, const int &maxX,
                         const int &minY, const int &maxY, const int &nFeatures);

        std::vector <cv::Point> pattern;    // orb descriptor pattern

        int n_features;       // target num of features
        int ini_th_fast;      // initial threshold of FAST (if FAST features not enough, extract again with min_th_fast
        int min_th_fast;      // minimum threshold of FAST, using it when FAST features are not enough

        // pyramid information
        static float scale_factor;
        static float log_sale_factor;
        static int n_levels;
        static std::vector<float> scale_factors;
        static std::vector<float> inv_scale_factors;
        static std::vector<float> square_sigmas;
        static std::vector<float> inv_square_sigmas;

        // num of features per level
        std::vector<int> n_features_per_level;

        // the pixel circle of FAST corner
        std::vector<int> u_max;
    };

} // mono_orb_slam3

#endif //MONO_ORB_SLAM3_ORBEXTRACTOR_H
