//
// Created by whitby on 10/24/23.
//

#include <iostream>
#include <fstream>

#include <Eigen/Core>

using namespace std;

struct Velo {
    double t{};
    Eigen::Vector3f v;
};

istream &operator>>(istream &input, Velo &velo) {
    input >> velo.t >> velo.v[0] >> velo.v[1] >> velo.v[2];
    return input;
}

void loadVelo1(const string &path, vector<Velo> &vecVelo) {
    ifstream fin(path);
    vecVelo.reserve(5000);
    while (!fin.eof()) {
        string lineStr;
        getline(fin, lineStr);

        if (!lineStr.empty()) {
            stringstream ss(lineStr);
            Velo velo;
            ss >> velo;
            vecVelo.push_back(velo);
        }
    }

    fin.close();
}

void loadVelo2(const string &path, vector<Velo> &vecVelo) {
    ifstream fin(path);
    vecVelo.reserve(5000);
    while (!fin.eof()) {
        string lineStr;
        getline(fin, lineStr);

        if (!lineStr.empty()) {
            stringstream ss(lineStr);
            Velo velo;
            int id;
            ss >> id;
            ss >> velo;
            vecVelo.push_back(velo);
        }
    }

    fin.close();
}

int main(int argc, char *argv[]) {
    if (argc != 3) {
        cout << "Usage: ./test/compute_velo_acc test_velo.txt truth_velo.txt" << endl;
        return -1;
    }

    vector<Velo> testVelo, truthVelo;
    loadVelo2(argv[1], testVelo);
    int numTest = testVelo.size();

    loadVelo1(argv[2], truthVelo);
    int numTruth = truthVelo.size();

    cout << "load " << numTest << " test velo data, " << numTruth << " truth velo data" << endl;

    int idx1 = 0, idx2 = 0;
    double error = 0;
    int count = 0;
    while (idx1 < numTest) {
        while (idx2 < numTruth && truthVelo[idx2].t < testVelo[idx1].t) {
            idx2++;
        }

        if (idx2 >= numTruth) break;
        count++;
        error += truthVelo[idx2].v.norm() - testVelo[idx1].v.norm();

        idx1++;
    }

    cout << "sum error is " << error << endl;
    cout << "average error is " << error / count << endl;

    return 0;
}