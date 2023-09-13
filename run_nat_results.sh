#!/bin/bash
set -e

echo "Running NAT results"
echo "-------------------"

nat_ecog_file=$1
nat_pose_file=$2

echo "Running NAT results with ECoG as: $nat_ecog_file"
echo "Running NAT results with Pose as: $nat_pose_file"

# should I include TME eventually?

python run_pca.py $nat_ecog_file
python run_pca_bootstrap.py $nat_ecog_file
python run_paa.py $nat_ecog_file

python run_movement_correl.py $nat_pose_file $nat_ecog_file

# runipy -o figures/CrossMovement.ipynb
# runipy -o figures/CrossDays.ipynb

python make_roi_region_contribs_df.py
# runipy -o figures/CrossParticipants.ipynb
