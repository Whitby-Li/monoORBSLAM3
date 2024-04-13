import numpy as np
import matplotlib.pyplot as plt

sequence = 'tnp_01'
folder = '/home/whitby/Development/CLionProjects/slam/mono_orb_slam3/results/aligned'


if __name__ == '__main__':
    truth_path = '/datasets/ntu_viral/'+ sequence + '/truth_pos.txt'
    compare_path = folder + '/old_ntu_viral_' + sequence + '.txt'
    test_path = folder + '/ntu_viral_' + sequence + '.txt'

    # truth_path = '/datasets/kitti/'+ sequence[0:10] + '/' + sequence[0:11] + 'drive_' + sequence[11:] + '_extract/oxts/full_pos.txt'
    # compare_path = folder + '/mono_kitti_' + sequence + '.txt'
    # test_path = folder + '/kitti_' + sequence + '.txt'

    truth_trajectory = np.loadtxt(truth_path, dtype=float)
    compare_trajectory = np.loadtxt(compare_path, dtype=float)
    test_trajectory = np.loadtxt(test_path, dtype=float)

    print(truth_trajectory.shape, compare_trajectory.shape, test_trajectory.shape)

    # plot trajectories

    # truth
    plt.plot(truth_trajectory[:, 1], truth_trajectory[:, 2], color='red', linestyle='-', label='Truth')

    # compare
    plt.plot(compare_trajectory[:, 0], compare_trajectory[:, 1], color='green', linestyle='-', label='ORB-SLAM3')
    # plt.plot(compare_trajectory[:, 0], compare_trajectory[:, 1], color='green', marker='o', markersize=3)
    # plt.scatter(compare_trajectory[0, 0], compare_trajectory[0, 1], color='green', marker='o', label='ORB-SLAM3 Start')
    # plt.scatter(compare_trajectory[-1, 0], compare_trajectory[-1, 1], color='green', marker='*', label='ORB-SLAM3 End')

    # test
    plt.plot(test_trajectory[:, 0], test_trajectory[:, 1], color='blue', linestyle='-', label='Ours')
    # plt.plot(test_trajectory[:, 0], test_trajectory[:, 1], color='blue', marker='o', markersize=3)
    # plt.scatter(test_trajectory[0, 0], test_trajectory[0, 1], color='blue', marker='o', label='Ours Start')
    # plt.scatter(test_trajectory[-1, 0], test_trajectory[-1, 1], color='blue', marker='*', label='Ours End')

    # 设置两轴尺度一致
    plt.axis('equal')  # 或者可以使用 plt.gca().set_aspect('equal')

    # 添加标题和标签
    plt.title(sequence + ' Trajectories')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')

    # 显示图例
    plt.legend()

    # 显示图形
    # plt.show()
    plt.savefig('../results/imgs/' + sequence + '.png')

