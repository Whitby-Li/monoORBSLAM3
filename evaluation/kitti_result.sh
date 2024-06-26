#!/bin/bash
# shellcheck disable=SC2164
cd /home/whitby/Development/slam/mono_orb_slam3/evaluation
python compare.py /datasets/kitti/2011_09_26/2011_09_26_drive_0014_extract/oxts/full_pos.txt /home/whitby/Development/CLionProjects/slam/mono_orb_slam3/results/kitti_2011_09_26_0014.txt /home/whitby/Development/CLionProjects/slam/mono_orb_slam3/results/aligned/kitti_2011_09_26_0014.txt
python compare.py /datasets/kitti/2011_09_26/2011_09_26_drive_0022_extract/oxts/full_pos.txt /home/whitby/Development/CLionProjects/slam/mono_orb_slam3/results/kitti_2011_09_26_0022.txt /home/whitby/Development/CLionProjects/slam/mono_orb_slam3/results/aligned/kitti_2011_09_26_0022.txt
python compare.py /datasets/kitti/2011_09_26/2011_09_26_drive_0039_extract/oxts/full_pos.txt /home/whitby/Development/CLionProjects/slam/mono_orb_slam3/results/kitti_2011_09_26_0039.txt /home/whitby/Development/CLionProjects/slam/mono_orb_slam3/results/aligned/kitti_2011_09_26_0039.txt
python compare.py /datasets/kitti/2011_09_26/2011_09_26_drive_0070_extract/oxts/full_pos.txt /home/whitby/Development/CLionProjects/slam/mono_orb_slam3/results/kitti_2011_09_26_0070.txt /home/whitby/Development/CLionProjects/slam/mono_orb_slam3/results/aligned/kitti_2011_09_26_0070.txt
python compare.py /datasets/kitti/2011_09_30/2011_09_30_drive_0018_extract/oxts/full_pos.txt /home/whitby/Development/CLionProjects/slam/mono_orb_slam3/results/kitti_2011_09_30_0018.txt /home/whitby/Development/CLionProjects/slam/mono_orb_slam3/results/aligned/kitti_2011_09_30_0018.txt
python compare.py /datasets/kitti/2011_09_30/2011_09_30_drive_0020_extract/oxts/full_pos.txt /home/whitby/Development/CLionProjects/slam/mono_orb_slam3/results/kitti_2011_09_30_0020.txt /home/whitby/Development/CLionProjects/slam/mono_orb_slam3/results/aligned/kitti_2011_09_30_0020.txt

python compare.py /datasets/kitti/2011_09_26/2011_09_26_drive_0014_extract/oxts/full_pos.txt /home/whitby/Development/CLionProjects/slam/mono_orb_slam3/results/mono_kitti_2011_09_26_0014.txt /home/whitby/Development/CLionProjects/slam/mono_orb_slam3/results/aligned/mono_kitti_2011_09_26_0014.txt
python compare.py /datasets/kitti/2011_09_26/2011_09_26_drive_0022_extract/oxts/full_pos.txt /home/whitby/Development/CLionProjects/slam/mono_orb_slam3/results/mono_kitti_2011_09_26_0022.txt /home/whitby/Development/CLionProjects/slam/mono_orb_slam3/results/aligned/mono_kitti_2011_09_26_0022.txt
python compare.py /datasets/kitti/2011_09_26/2011_09_26_drive_0039_extract/oxts/full_pos.txt /home/whitby/Development/CLionProjects/slam/mono_orb_slam3/results/mono_kitti_2011_09_26_0039.txt /home/whitby/Development/CLionProjects/slam/mono_orb_slam3/results/aligned/mono_kitti_2011_09_26_0039.txt
python compare.py /datasets/kitti/2011_09_26/2011_09_26_drive_0070_extract/oxts/full_pos.txt /home/whitby/Development/CLionProjects/slam/mono_orb_slam3/results/mono_kitti_2011_09_26_0070.txt /home/whitby/Development/CLionProjects/slam/mono_orb_slam3/results/aligned/mono_kitti_2011_09_26_0070.txt
python compare.py /datasets/kitti/2011_09_30/2011_09_30_drive_0018_extract/oxts/full_pos.txt /home/whitby/Development/CLionProjects/slam/mono_orb_slam3/results/mono_kitti_2011_09_30_0018.txt /home/whitby/Development/CLionProjects/slam/mono_orb_slam3/results/aligned/mono_kitti_2011_09_30_0018.txt
python compare.py /datasets/kitti/2011_09_30/2011_09_30_drive_0020_extract/oxts/full_pos.txt /home/whitby/Development/CLionProjects/slam/mono_orb_slam3/results/mono_kitti_2011_09_30_0020.txt /home/whitby/Development/CLionProjects/slam/mono_orb_slam3/results/aligned/mono_kitti_2011_09_30_0020.txt
