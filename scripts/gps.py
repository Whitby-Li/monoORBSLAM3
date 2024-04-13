import os

# data path
data_folder = "/datasets/kitti/2011_09_30/2011_09_30_drive_0018_extract/oxts"


def get_seconds(str_time):
    t = [float(x) for x in str_time.split(":")]
    return t[0] * 3600 + t[1] * 60 + t[2]


def read_time(path):
    timestamps = []
    with open(path, 'r') as f:
        for line in f.readlines():
            timestamps.append(get_seconds(line.split(' ')[1]))

    return timestamps


def read_single_gps(path):
    with open(path, 'r') as f:
        str_list = f.readline().strip().split(' ')

    data = []
    data.append(str_list[1])
    data.append(str_list[0])
    data.append(str_list[2])
    return data


def gather_imu_data(data_folder):
    time_path = os.path.join(data_folder, 'timestamps.txt')
    times = read_time(time_path)

    gps_path = os.path.join(data_folder, 'data')
    gps_data = []
    for i in range(len(times)):
        gps_data.append(read_single_gps(gps_path + "/%010d.txt" % i))

    # save imu
    with open(data_folder + "/gps.txt", 'w') as f:
        for i in range(len(times)):
            f.write("{:.6f}".format(times[i]) + " ")
            f.write(" ".join(gps_data[i]) + "\n")


if __name__ == '__main__':
    gather_imu_data(data_folder)
