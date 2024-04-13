//
// Created by whitby on 8/23/23.
//

#include "ORBVocabulary.h"

namespace mono_orb_slam3 {
    Vocabulary *ORBVocabulary::vocabulary = nullptr;

    bool ORBVocabulary::createORBVocabulary(const string &path) {
        if (vocabulary == nullptr) {
            vocabulary = new Vocabulary();
            bool sign = vocabulary->loadFromTextFile(path);
            if (!sign) {
                delete vocabulary;
                vocabulary = nullptr;
            }
            return sign;
        }
        return false;
    }

    const Vocabulary *ORBVocabulary::getORBVocabulary() {
        return vocabulary;
    }
}
