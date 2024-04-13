import os

# data path
data_folder = "/datasets/kitti/2011_09_30/2011_09_30_drive_0033_extract/oxts"


def get_seconds(str_time):
    t = [float(x) for x in str_time.split(":")]
    return t[0] * 3600 + t[1] * 60 + t[2]

def read_time(path):
    timestamps = []
    with open(path, 'r') as f:
        for line in f.readlines():
            timestamps.append(get_seconds(line.split(' ')[1]))

    return timestamps


def read_single_imu(path):
    with open(path, 'r') as f:
        str_list = f.readline().strip().split(' ')

    return str_list[17:20] + str_list[11:14]


def gather_imu_data(data_folder):
    time_path = os.path.join(data_folder, 'timestamps.txt')
    times = read_time(time_path)

    imu_path = os.path.join(data_folder, 'data')
    imu_data = []
    for i in range(len(times)):
        imu_data.append(read_single_imu(imu_path + "/%010d.txt" % i))

    # save imu
    with open(data_folder + "/imu.txt", 'w') as f:
        for i in range(len(times)):
            f.write("{:.6f}".format(times[i]) + " ")
            f.write(" ".join(imu_data[i]) + "\n")


if __name__ == '__main__':
    gather_imu_data(data_folder)
