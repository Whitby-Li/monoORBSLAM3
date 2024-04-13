import numpy as np
import matplotlib.pyplot as plt


path = '/datasets/phone/raw_data/2023-09-22-082154/pos.txt'

if __name__ == '__main__':
    trajectory = np.loadtxt(path, dtype=float)
    trajectory1 = np.loadtxt('../results/aligned/old_phone_092201.txt', dtype=float)
    trajectory2 = np.loadtxt('../results/aligned/phone_092201.txt', dtype=float)

    plt.plot(trajectory[:, 1], trajectory[:, 2], color='red', linestyle='-', label='GNSS')
    plt.plot(trajectory1[:, 0], trajectory1[:, 1], color='green', linestyle='-', label='ORB-SLAM3')
    plt.plot(trajectory2[:, 0], trajectory2[:, 1], color='blue', linestyle='-', label='Ours')

    # 设置两轴尺度一致
    plt.axis('equal')  # 或者可以使用 plt.gca().set_aspect('equal')

    # 添加标题和标签
    plt.title('City Trajectory')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')

    # 显示图例
    plt.legend()

    # 显示图形
    plt.show()