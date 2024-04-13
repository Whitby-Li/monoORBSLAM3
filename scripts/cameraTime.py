# data path
camera_folder = "/datasets/kitti/2011_09_30/2011_09_30_drive_0033_extract/image_00"


def get_seconds(str_time):
    t = [float(x) for x in str_time.split(":")]
    return t[0] * 3600 + t[1] * 60 + t[2]


def read_time(path):
    timestamps = []
    with open(path, 'r') as f:
        for line in f.readlines():
            timestamps.append(get_seconds(line.split(' ')[1]))

    return timestamps


def save_time(input_path, output_path):
    timestamps = read_time(input_path)

    with open(output_path, 'w') as f:
        for t in timestamps:
            f.write("{:.9f}".format(t) + '\n')


if __name__ == '__main__':
    save_time(camera_folder + "/timestamps.txt", camera_folder + "/times.txt")
