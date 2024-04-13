//
// Created by whitby on 8/23/23.
//

#ifndef MONO_ORB_SLAM3_ORBVOCABULARY_H
#define MONO_ORB_SLAM3_ORBVOCABULARY_H

#include "DBoW2/FORB.h"
#include "DBoW2/TemplatedVocabulary.h"

namespace mono_orb_slam3 {
    typedef DBoW2::TemplatedVocabulary<DBoW2::FORB::TDescriptor , DBoW2::FORB> Vocabulary;

    class ORBVocabulary {
    public:
        static bool createORBVocabulary(const string &path);

        static const Vocabulary *getORBVocabulary();

    private:
        ORBVocabulary() = default;

        static Vocabulary  *vocabulary;
    };
}


#endif //MONO_ORB_SLAM3_ORBVOCABULARY_H
