#!/bin/bash
# shellcheck disable=SC2164
cd /home/whitby/Development/CLionProjects/slam/mono_orb_slam3

dataset_name="tum"
dataset_folder="/datasets/$dataset_name"

find "$dataset_folder" -maxdepth 1 -mindepth 1 -type d -not -name "calib" -print0 | while IFS= read -r -d '' folder; do
    seq_name=$(basename "$folder")
    echo "$folder $seq_name"

    ./test/bin/demo settings/"$dataset_name".yaml  vocabulary/ORBvoc.txt "$folder" results/"$dataset_name"_"$seq_name"
done