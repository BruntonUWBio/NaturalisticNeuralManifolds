#!/bin/bash
set -e

echo "Running EXP results"
echo "-------------------"

exp_ecog_file=$1
exp_pose_file=$2

echo "Running EXP results with ECoG as: $exp_ecog_file"
echo "Running EXP results with Pose as: $exp_pose_file"

# should I include TME eventually?

python run_pca.py $exp_ecog_file
python run_paa.py $exp_ecog_file
python run_movement_correl.py $exp_pose_file $exp_ecog_file
python make_roi_region_contribs_df_exp.py

# causes issues with the notebooks
# runipy figures/ExpNeuralManifolds.ipynb --html figures/ExpNeuralManifolds.html
# runipy figures/SupplementExpROIs.ipynb --html figures/SupplementExpROIs.html