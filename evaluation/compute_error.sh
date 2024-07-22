#!/bin/bash
# shellcheck disable=SC2164
cd /home/whitby/Development/CLionProjects/slam/mono_orb_slam3

dataset_name="euroc"
dataset_folder="/datasets/$dataset_name"

find "$dataset_folder" -maxdepth 1 -mindepth 1 -type d -not -name "calib" -print0 | sort -z | while IFS= read -r -d '' folder; do
    seq_name=$(basename "$folder")
    result_folder=results/"$dataset_name"_"$seq_name"
    result_folder2=/home/whitby/Documents/slam/paper/slam_test/orb_slam3/results/"$dataset_name"_"$seq_name"
    result_folder3=/home/whitby/ros_ws/fusion_ws/src/VINS-Fusion/results/"$dataset_name"_"$seq_name"

    python evaluation/compare.py "$folder"/ground_truth.txt "$result_folder3"/kf_trajectory.txt "$result_folder3" --max_differ 0.005

    python evaluation/compare.py "$folder"/ground_truth.txt "$result_folder2"/kf_trajectory.txt "$result_folder2" --max_differ 0.005

    python evaluation/compare.py "$folder"/ground_truth.txt "$result_folder"/kf_trajectory.txt "$result_folder" --max_differ 0.005

    python evaluation/plot_results.py "$folder"/ground_truth.txt red GroundTruth "$result_folder3"/align_trajectory.txt orange VINS_Fusion "$result_folder2"/align_trajectory.txt green ORB_SLAM3 "$result_folder"/align_trajectory.txt blue Ours --graph_name "$dataset_name"_"$seq_name" --output_dir "$result_folder"
done