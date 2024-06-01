import argparse
import numpy as np


def kitti_associate(truth_data, test_data, max_differ):
    match_truth, match_test = [], []
    len_truth, len_test = len(truth_data), len(test_data)
    idx1, idx2 = 0, 0

    while idx2 < len_test:
        while idx1 < len_truth and test_data[idx2][0] > truth_data[idx1][0]:
            idx1 += 1

        if idx1 == len_truth: break

        if abs(truth_data[idx1][0] - test_data[idx2][0]) < max_differ:
            match_test.append(test_data[idx2])
            match_truth.append(truth_data[idx1])
        elif idx1 > 0 and abs(truth_data[idx1 - 1][0] - test_data[idx2][0]) < max_differ:
            match_test.append(test_data[idx2])
            match_truth.append(truth_data[idx1 - 1])

        idx2 += 1

    match_truth = np.array(match_truth, dtype=float)
    match_test = np.array(match_test, dtype=float)

    return match_truth, match_test


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='''This script takes two data files, associate them with timestamp''')
    parser.add_argument('first_file', help='truth data (format: timestamp data)(suffix: .csv')
    parser.add_argument('second_file', help='test data (format: timestamp data)(suffix: .txt')
    parser.add_argument('--max_differ', help='maximum time difference for matching entries (default: 0.005)',
                        default=0.025)
    args = parser.parse_args()

    # 1. load truth and test trajectories
    truth_trajectory = np.loadtxt(args.first_file, dtype=float)
    test_trajectory = np.loadtxt(args.second_file, dtype=float)

    print(args.first_file)
    print("load %d truth trajectory data, %d test trajectory data" %(len(truth_trajectory), len(test_trajectory)))

    # 2. match two trajectories by timestamp
    match_truth_trajectory, match_test_trajectory = kitti_associate(truth_trajectory, test_trajectory, float(args.max_differ))

    if len(match_test_trajectory) == len(match_truth_trajectory):
        print("match success, there are %d pairs" % len(match_test_trajectory))
    else:
        exit(-1)

    # 3. compute absolute translation error
    truth_xyz = np.matrix(match_truth_trajectory[:, 1:4]).transpose()
    test_xyz = np.matrix(match_test_trajectory[:, 1:4]).transpose()

    align_error = truth_xyz - test_xyz
    translation_error = np.sqrt(np.sum(np.multiply(align_error, align_error), 0)).A[0]

    print("absolute translation error: ", np.sqrt(np.dot(translation_error, translation_error)/ len(translation_error)))
    print()
