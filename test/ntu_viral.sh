#!/bin/bash
# shellcheck disable=SC2164
cd /home/whitby/Development/CLionProjects/slam/mono_orb_slam3
./test/bin/ntu_demo settings/ntu_viral.yaml vocabulary/ORBvoc.txt /datasets/ntu_viral/eee_01 results/ntu_viral_eee_01.txt
./test/bin/ntu_demo settings/ntu_viral.yaml vocabulary/ORBvoc.txt /datasets/ntu_viral/nya_01 results/ntu_viral_nya_01.txt
./test/bin/ntu_demo settings/ntu_viral.yaml vocabulary/ORBvoc.txt /datasets/ntu_viral/sbs_01 results/ntu_viral_sbs_01.txt
./test/bin/ntu_demo settings/ntu_viral.yaml vocabulary/ORBvoc.txt /datasets/ntu_viral/tnp_01 results/ntu_viral_tnp_01.txt
