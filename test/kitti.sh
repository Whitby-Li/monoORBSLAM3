#!/bin/bash
# shellcheck disable=SC2164
cd /home/whitby/Development/CLionProjects/slam/mono_orb_slam3

./test/bin/kitti_demo settings/kitti.yaml vocabulary/ORBvoc.txt /datasets/kitti/2011_09_30/2011_09_30_drive_0018_extract results/kitti_2011_09_30_0018.txt results/kitti_2011_09_30_0018_velo_bias.txt
./test/bin/kitti_demo settings/kitti.yaml vocabulary/ORBvoc.txt /datasets/kitti/2011_09_30/2011_09_30_drive_0020_extract results/kitti_2011_09_30_0020.txt results/kitti_2011_09_30_0020_velo_bias.txt
./test/bin/kitti_demo settings/kitti.yaml vocabulary/ORBvoc.txt /datasets/kitti/2011_09_26/2011_09_26_drive_0014_extract results/kitti_2011_09_26_0014.txt results/kitti_2011_09_26_0014_velo_bias.txt
./test/bin/kitti_demo settings/kitti.yaml vocabulary/ORBvoc.txt /datasets/kitti/2011_09_26/2011_09_26_drive_0022_extract results/kitti_2011_09_26_0022.txt results/kitti_2011_09_26_0022_velo_bias.txt
./test/bin/kitti_demo settings/kitti.yaml vocabulary/ORBvoc.txt /datasets/kitti/2011_09_26/2011_09_26_drive_0039_extract results/kitti_2011_09_26_0039.txt results/kitti_2011_09_26_0039_velo_bias.txt
./test/bin/kitti_demo settings/kitti.yaml vocabulary/ORBvoc.txt /datasets/kitti/2011_09_26/2011_09_26_drive_0070_extract results/kitti_2011_09_26_0070.txt results/kitti_2011_09_26_0070_velo_bias.txt