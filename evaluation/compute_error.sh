#!/bin/bash
# shellcheck disable=SC2164
cd /home/whitby/Development/CLionProjects/slam/mono_orb_slam3

dataset_name="euroc"
dataset_folder="/datasets/$dataset_name"

find "$dataset_folder" -maxdepth 1 -mindepth 1 -type d -not -name "calib" -print0 | while IFS= read -r -d '' folder; do
    seq_name=$(basename "$folder")
    result_folder=results/"$dataset_name"_"$seq_name"
    result_folder2=/home/whitby/Documents/slam/paper/slam_test/orb_slam3/results/"$dataset_name"_"$seq_name"

    python evaluation/compare.py "$folder"/ground_truth.txt "$result_folder"/trajectory.txt "$result_folder"/align_trajectory.txt
    python evaluation/compare.py "$folder"/ground_truth.txt "$result_folder2"/trajectory.txt "$result_folder2"/align_trajectory.txt
done