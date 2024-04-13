import argparse
import numpy as np
import matplotlib.pyplot as plt


def associate(truth_data, test_data, max_differ):
    """
    associate truth data and test data by their timestamp
    :param truth_data: [[timestamp, ...]]
    :param test_data: [[timestamp, ...]]
    :param max_differ: maximum difference time
    :return: matched truth data and test data
    """
    match_truth, match_test = [], []
    len_truth, len_test = len(truth_data), len(test_data)
    idx1, idx2 = 0, 0

    while idx2 < len_test and truth_data[idx1][0] > test_data[idx2][0]:
        idx2 += 1

    while idx2 < len_test:
        while idx1 < len_truth and abs(test_data[idx2][0] - truth_data[idx1][0]) > max_differ:
            idx1 += 1

        if idx1 < len_truth:
            match_truth.append(truth_data[idx1]), match_test.append(test_data[idx2])

        idx2 += 1

    match_truth = np.array(match_truth, dtype=float)
    match_test = np.array(match_test, dtype=float)

    return match_truth, match_test


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


def align_mono(model, data):
    """Align two trajectories using the method of Horn(closed-from)."""
    np.set_printoptions(precision=3, suppress=True)
    model_centered = model - model.mean(1)
    data_centered = data - data.mean(1)

    W = np.zeros((3, 3), dtype=float)
    for column in range(model.shape[1]):
        W += np.outer(model_centered[:, column], data_centered[:, column])

    U, d, Vh = np.linalg.linalg.svd(W.transpose())
    S = np.matrix(np.identity(3))
    if np.linalg.det(U) * np.linalg.det(Vh) < 0:
        S[2, 2] = -1

    rot = U * S * Vh

    rot_model = rot * model_centered
    dots, norms = 0, 0
    for column in range(data_centered.shape[1]):
        dots += np.dot(data_centered[:, column].transpose(), rot_model[:, column])
        normi = np.linalg.norm(model_centered[:, column])
        norms += normi * normi

    s = float(dots / norms)
    trans = data.mean(1) - s * rot * model.mean(1)

    return rot, s, trans


def align(model, data):
    """Align two trajectories using the method of Horn (closed-form).
    @:param model: first trajectory (3, n)
    @:param data: second trajectory (3, n)
    @:returns:
        - rot: rotation matrix (3, 3)
        - trans: translation vector (3, 1)
        - trans_error: translation error per point (1, n)
    """
    np.set_printoptions(precision=3, suppress=True)
    model_centered, data_centered = model - model.mean(1), data - data.mean(1)

    W = np.zeros((3, 3), dtype=float)
    for col in range(model.shape[1]):
        W += np.outer(model_centered[:, col], data_centered[:, col])

    U, d, Vh = np.linalg.linalg.svd(W.transpose())

    S = np.matrix(np.identity(3))
    if np.linalg.det(U) * np.linalg.det(Vh) < 0:
        S[2, 2] = -1
    rot = U * S * Vh

    model_rotated = rot * model_centered
    dots, norms = 0.0, 0.0
    for col in range(data_centered.shape[1]):
        dots += np.dot(data_centered[:, col].transpose(), model_rotated[:, col])
        norm = np.linalg.norm(model_centered[:, col])
        norms += norm * norm

    s = float(dots / norms)

    trans_gt = data.mean(1) - s * rot * model.mean(1)
    trans = data.mean(1) - rot * model.mean(1)

    model_aligned_gt = s * rot * model + trans_gt
    model_aligned = rot * model + trans

    align_error_gt = model_aligned_gt - data
    align_error = model_aligned - data

    trans_error_gt = np.sqrt(np.sum(np.multiply(align_error_gt, align_error_gt), 0)).A[0]
    trans_error = np.sqrt(np.sum(np.multiply(align_error, align_error), 0)).A[0]

    return rot, trans_gt, trans_error_gt, trans, trans_error, s


def plot_trajectory_3d(ax, data, color, label):
    data = np.array(data)
    ax.plot(data[0], data[1], data[2], color=color, linestyle='-', label=label)
    ax.scatter(data[0][0], data[1][0], data[2][0], color='red', marker='o', label='Start Point')
    ax.scatter(data[0][-1], data[1][-1], data[2][-1], color='red', marker='x', label='End Point')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='''This script takes two data files, associate them with timestamp''')
    parser.add_argument('first_file', help='truth data (format: timestamp data)(suffix: .csv')
    parser.add_argument('second_file', help='test data (format: timestamp data)(suffix: .txt')
    parser.add_argument('--max_differ', help='maximum time difference for matching entries (default: 0.005)',
                        default=0.005)
    args = parser.parse_args()

    # 1. load truth and test trajectories
    truth_trajectory = np.loadtxt(args.first_file, dtype=float)
    test_trajectory = np.loadtxt(args.second_file, dtype=float)

    print("load %d truth trajectory data, %d test trajectory data" %(len(truth_trajectory), len(test_trajectory)))

    # 2. match two trajectories by timestamp
    match_truth_trajectory, match_test_trajectory = kitti_associate(truth_trajectory, test_trajectory, float(args.max_differ))

    if len(match_test_trajectory) == len(match_truth_trajectory):
        print("match success, there are %d pairs" % len(match_test_trajectory))
    else:
        exit(-1)

    # 3. align two trajectories
    truth_xyz = np.matrix(match_truth_trajectory[:, 1:4]).transpose()
    test_xyz = np.matrix(match_test_trajectory[:, 1:4]).transpose()

    rotation, translation_gt, translation_error_gt, translation, translation_error, scale = align(test_xyz, truth_xyz)
    test_xyz_aligned = scale * rotation * test_xyz + translation_gt

    print("scale error: ", scale)
    print("absolute translation error with scale: ",
          np.sqrt(np.dot(translation_error_gt, translation_error_gt) / len(translation_error_gt)))
    #
    # # 4. plot
    # print("rotation: \n", rotation)
    # print("translation: ", translation.transpose())

    # test_timestamps = match_test_trajectory[:, 0]
    # test_xyz_aligned = scale * rotation * test_xyz + translation
    # test_xyz_not_scaled = rotation * test_xyz + translation
    #
    # test_xyz_full = np.matrix(test_trajectory[:, 1:4]).transpose()
    # test_xyz_full_aligned = scale * rotation * test_xyz_full + translation
    #
    # truth_timestamps = match_truth_trajectory[:, 0]
    # truth_xyz_full = np.matrix(truth_trajectory[:, 1:4]).transpose()

    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    #
    # plot_trajectory_3d(ax, test_xyz_aligned, 'blue', 'estimated trajectory')
    # plot_trajectory_3d(ax, truth_xyz, 'black', 'truth trajectory')
    #
    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')
    # ax.set_zlabel('Z')
    # ax.set_title('3D Trajectory')
    #
    # ax.set_box_aspect([1, 1, 1])
    #
    # ax.legend()
    # plt.show()

    print(test_xyz_aligned.shape, truth_xyz.shape)

    estimated_xyz = np.array(test_xyz_aligned)
    plt.plot(estimated_xyz[0], estimated_xyz[1], color='green', linestyle='-', label='Estimated')
    # plt.scatter(test_xyz_aligned[0, 0], test_xyz_aligned[1, 0], color='green', marker='o', label='Estimated Start')
    # plt.scatter(test_xyz_aligned[0,  -1], test_xyz_aligned[1, -1], color='green', marker='*', label='Estimated End')

    truth_xyz = np.array(truth_xyz)
    plt.plot(truth_xyz[0], truth_xyz[1], color='red', linestyle='-', label='Truth')
    # plt.scatter(truth_xyz[0, 0], truth_xyz[1, 0], color='red', marker='o', label='Truth Start')
    # plt.scatter(truth_xyz[0, -1], truth_xyz[1, -1], color='red', marker='*', label='Truth End')

    # 设置两轴尺度一致
    plt.axis('equal')  # 或者可以使用 plt.gca().set_aspect('equal')

    # 添加标题和标签
    plt.title('2D Trajectory with Start and End Points (Equal Scale)')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')

    # 显示图例
    plt.legend()

    # 显示图形
    plt.show()
