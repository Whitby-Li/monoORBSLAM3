import argparse
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='''This script plot trajectories''')
    parser.add_argument(
        'triplets',
        metavar='triplet',
        type=str,
        nargs='+',
        help="Triplets of trajectory file, color, name"
    )

    parser.add_argument(
        '--graph_name',
        type=str,
        required=True,
        help='name of result graph.'
    )

    parser.add_argument(
        '--output_dir',
        type=str,
        required=True,
        help='the directory where graph will be saved.'
    )

    args = parser.parse_args()
    if len(args.triplets) % 3 != 0:
        parser.error("You must provide triples of [trajectory, color, name]")

    triplets = [(args.triplets[i], args.triplets[i + 1], args.triplets[i + 2]) for i in range(0, len(args.triplets), 3)]

    for triple in triplets:
        trajectory = np.loadtxt(triple[0], dtype=float)
        color = triple[1]
        name = triple[2]
        plt.plot(trajectory[:, 1], trajectory[:, 2], color=color, linestyle='-', label=name)

    # 设置两轴尺度一致
    plt.axis('equal')  # 或者可以使用 plt.gca().set_aspect('equal')

    # 添加标题和标签
    plt.title(args.graph_name)
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')

    # 显示图例
    plt.legend()

    # 显示图形
    # plt.show()
    plt.savefig(args.output_dir + "/" + args.graph_name + '.png')

