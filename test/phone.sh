#!/bin/bash
# shellcheck disable=SC2164
cd /home/whitby/Development/CLionProjects/slam/mono_orb_slam3
./test/bin/phone_demo settings/phone.yaml vocabulary/ORBvoc.txt /datasets/phone/raw_data/2023-10-06-073852 results/phone_100601.txt results/phone_100601_velo_bias.txt