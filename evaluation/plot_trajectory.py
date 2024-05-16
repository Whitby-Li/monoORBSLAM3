import numpy as np
import matplotlib.pyplot as plt


path = '../results/tum_outdoors8/trajectory.txt'

if __name__ == '__main__':
    trajectory = np.loadtxt(path, dtype=float)

    plt.plot(trajectory[:, 1], trajectory[:, 2], color='red', linestyle='-', label='Pos')

    # 设置两轴尺度一致
    plt.axis('equal')  # 或者可以使用 plt.gca().set_aspect('equal')

    # 添加标题和标签
    plt.title('Trajectory')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')

    # 显示图例
    plt.legend()

    # 显示图形
    plt.show()