#!/bin/bash
# shellcheck disable=SC2164
cd /home/whitby/Development/CLionProjects/slam/mono_orb_slam3

./test/bin/euroc_demo settings/euroc.yaml vocabulary/ORBvoc.txt /datasets/euroc/MH_01 results/euroc_MH_01.txt results/euroc_MH_01_velo_bias.txt
./test/bin/euroc_demo settings/euroc.yaml vocabulary/ORBvoc.txt /datasets/euroc/MH_03 results/euroc_MH_03.txt results/euroc_MH_03_velo_bias.txt
./test/bin/euroc_demo settings/euroc.yaml vocabulary/ORBvoc.txt /datasets/euroc/MH_04 results/euroc_MH_04.txt results/euroc_MH_04_velo_bias.txt
./test/bin/euroc_demo settings/euroc.yaml vocabulary/ORBvoc.txt /datasets/euroc/MH_05 results/euroc_MH_05.txt results/euroc_MH_05_velo_bias.txt
./test/bin/euroc_demo settings/euroc.yaml vocabulary/ORBvoc.txt /datasets/euroc/V1_01 results/euroc_V1_01.txt results/euroc_V1_01_velo_bias.txt
./test/bin/euroc_demo settings/euroc.yaml vocabulary/ORBvoc.txt /datasets/euroc/V1_02 results/euroc_V1_02.txt results/euroc_V1_02_velo_bias.txt
./test/bin/euroc_demo settings/euroc.yaml vocabulary/ORBvoc.txt /datasets/euroc/V1_03 results/euroc_V1_03.txt results/euroc_V1_03_velo_bias.txt