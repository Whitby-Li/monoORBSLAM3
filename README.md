# monoORBSLAM3

### 介绍
个人参照 ORB-SLAM3 官方代码，在 Ubuntu 20.04 上使用较新的 c++ 库和 c++17 标准重新写的单目版本，另外未撰写回环检测、多地图融合和重定位模块；同时对惯性初始化模块、视觉惯性联合优化做了一些调整。

### 依赖包
##### 1. Eigen3
Eigen3 笔者使用的是 Ubuntu 20.04 官方库版本(版本号 3.3.7)，安装命令如下：
```bash
sudo apt-get update
sudo apt-get install libeigen3-dev
```

##### 2. OpenCV4
OpenCV4 同样也使用 Ubuntu 20.04 官方库版本(版本号 4.2.0)，安装命令如下：
```bash
sudo apt-get update
sudo apt-get install libopencv-dev
```

##### 3. g2o
g2o 开源地址为 `https://github.com/RainerKuemmerle/g2o/tree/20201223_git` （这里选择与 eigen、opencv 发布年代相近的分支版本），安装编译命令如下：
```bash
sudo apt-get update
sudo apt-get install libsuitesparse-dev qtdeclarative5-dev qt5-qmake libqglviewer-dev-qt5 # 可选依赖

# 切换到 g2o 下载目录，编译并安装
mkdir build && cd build
cmake ..
make -j 8
sudo make install
```

##### 4. pangolin

可视化库使用 [pangolin-v0.6](https://github.com/stevenlovegrove/Pangolin/tree/v0.6) ，安装编译命令如下：

```bash
# 切换到 pangolin 下载目录
mkdir build && cd build
cmake ..
make -j 8
sudo make install
```

### 编译运行
#### 第三方库 DBoW2 编译
ORB-SLAM2 需要使用词袋模型进行参考关键帧跟踪定义，以及局部建图模块关键帧匹配，三角化新的地图点，编译命令如下：
```bash
cd thirdParty/DBoW2
mkdir build && cd build
cmake ..
make -j 8
```

#### monoORBSLAM3 编译
```bash
# monoORBSLAM3 目录下
mkdir build && cd build
cmake ..
make -j 8
```

#### KITTI 数据运行

KITTI 数据由汽车采集，速度较快，难以稳定运行。ORB-SLAM3 官方代码未在 KITTI 的单目惯性数据上进行测试验证，这里改进了 ORB-SLAM3 的惯性初始化过程和视觉惯性联合跟踪定位，在部分 KITTI 数据上可以运行。

KITTI 原始数据的处理脚本在 scripts 目录下：

```bash
cameraTime.py # 相机时间戳整理
gps.py		  # GPS 定位数据整理
imu.py		  # IMU 数据整理
```

运行命令为：

```bash
test/kitti_demo setting.yaml vocabulary_file data_folder trajectory_save_path
```

如下图为 monoORBSLAM3 在 [2011_09_30_dirve_0018](https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_30_drive_0018/2011_09_30_drive_0018_extract.zip) 跟踪定位轨迹与 GPS 定位轨迹统一变换到 ENU 坐标系下。

<div align=center>
    <img src="imgs/kitti_093018.png" height="300" />
    <img src="imgs/kitti_093018_gps.png" height="300" />
    <img src="imgs/compare.png" height="300" />
</div>


明显视觉惯性定位结果在长时间运行后会出现位置漂移问题，需要后期融合 GPS 解决。

#### 手机采集校园场景数据运行

开发鸿蒙 APP ([SLAMRecord](https://gitee.com/whitbyli/slamrecorder))读取手机设备传感器数据，在校园步行录制数据和城市道路骑行录制数据上运行测试。

##### （1）校园场景数据测试

这里提供一个原始数据的(链接: https://pan.baidu.com/s/1qW_nSO6ptrIKBdOnWe4smA  密码: 6nc0) ，运行命令如下：

```bash
# 对应 test/phoneDemo.cpp 文件
test/slam_demo setting.yaml vocabulary_file data_folder trajectory_save_path
```

在 monoORBSLAM3 与官方代码 ORB-SLAM3 下运行的关键帧点云截图如下所示 （左图为 monoORBSLAM3 ，右图为官方代码截图）

<div align=center>
    <img src="imgs/mono_orb_slam3_081201_1.png" height="400" />
    <img src="imgs/orb_slam3_081201_1.png" height="400" />
</div>


##### （2）城市道路场景测试

这里仅提供一个城市道路场景数据（链接: https://pan.baidu.com/s/1-hHFbiRQPdl4J-TR1cTdbQ  密码: 1gac），运行结果如下：


骑行录制最终回到原点，**最终的位置偏差**如下图所示：


在大规模室外环境下 ORB-SLAM3 基于词袋和共视图的回环检测算法很难准确进行回环修正，经长时间运行后，二者均出现较大的位置偏差