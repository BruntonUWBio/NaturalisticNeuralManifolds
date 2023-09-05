import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import json
import os
import sys
from tqdm import tqdm
import itertools
import scipy.stats as stats
import scipy
import sklearn.metrics as skmetrics
import sklearn
import xarray as xr
from itertools import combinations

import pdb

import src
from src.data_utils import ECoG_Data
import src.manifold_u as mu

PERCENT_BOOTSTRAP = 0.8
N_BOOTSTRAP = 100


def plot_by_freq(ecog_pose_df, freq_bands, pose_exp_params, metric_name, pose_combo):
    for f in freq_bands:
        cur_freq = f
        pas_cur_freq_df = ecog_pose_df[ecog_pose_df["Frequency"] == cur_freq]
        # also remove rest comparisons
        pas_cur_freq_df = pas_cur_freq_df[
            ~pas_cur_freq_df["Movement Comparison"].str.contains("rest")
        ]

        # plot just with points
        fig, ax = plt.subplots()
        sns.scatterplot(
            x=pas_cur_freq_df["Neural Dissimilarity"],
            y=pas_cur_freq_df["Pose Distance"],
            hue=pas_cur_freq_df["Participant"],
        )
        # plot a trendline
        z = np.polyfit(
            pas_cur_freq_df["Neural Dissimilarity"], pas_cur_freq_df["Pose Distance"], 1
        )
        p = np.poly1d(z)
        plt.plot(
            pas_cur_freq_df["Neural Dissimilarity"],
            p(pas_cur_freq_df["Neural Dissimilarity"]),
            "r--",
        )
        # add correlation to plot
        try:
            corr = stats.pearsonr(
                pas_cur_freq_df["Neural Dissimilarity"],
                pas_cur_freq_df["Pose Distance"],
            )
        except ValueError:
            print("Error calculating the correlation")
            corr = [0, 0]
        plt.text(
            0.5,
            0.5,
            "r = " + str(round(corr[0], 2)),
            horizontalalignment="center",
            verticalalignment="center",
            transform=ax.transAxes,
        )
        file_special = ""
        if abs(corr[0]) >= 0.7:
            print("Correlation above 0.7 here!!")
            file_special = "_notice"

        plt.xlabel("Neural Dissimilarity")
        plt.ylabel("Pose Average Distance")
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)
        plt.title(cur_freq)
        plt.savefig(
            os.path.join(
                pose_exp_params["sp"],
                metric_name,
                pose_combo,
                "pose_avg_vs_neural_" + cur_freq + file_special + ".png",
            ),
            bbox_inches="tight",
        )

        # plot with the error bars
        sns.scatterplot(
            x=pas_cur_freq_df["Neural Dissimilarity"],
            y=pas_cur_freq_df["Pose Distance"],
            hue=pas_cur_freq_df["Participant"],
        )
        plt.errorbar(
            x=pas_cur_freq_df["Neural Dissimilarity"],
            y=pas_cur_freq_df["Pose Distance"],
            yerr=pas_cur_freq_df["Pose Distance Std"],
            fmt="o",
        )
        plt.xlabel("Neural Dissimilarity")
        plt.ylabel("Pose Average Distance")
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)
        plt.title(cur_freq)
        plt.savefig(
            os.path.join(
                pose_exp_params["sp"],
                metric_name,
                pose_combo,
                "pose_avg_std_vs_neural_" + cur_freq + file_special + ".png",
            ),
            bbox_inches="tight",
        )


def plot_by_pat(
    ecog_pose_df, pats_ids_in, pose_exp_params, metric_name, pose_combo, cur_freq="LFO"
):
    pas_cur_freq_df = ecog_pose_df[ecog_pose_df["Frequency"] == cur_freq]
    for i, pat_id_curr in enumerate(pats_ids_in):
        fig, ax = plt.subplots()

        cur_sbj_df = pas_cur_freq_df[pas_cur_freq_df["Participant"]
                                     == pat_id_curr]
        sns.scatterplot(
            x=cur_sbj_df["Neural Dissimilarity"], y=cur_sbj_df["Pose Distance"]
        )

        # plot a trendline
        z = np.polyfit(
            cur_sbj_df["Neural Dissimilarity"], cur_sbj_df["Pose Distance"], 1
        )
        p = np.poly1d(z)
        plt.plot(
            cur_sbj_df["Neural Dissimilarity"],
            p(cur_sbj_df["Neural Dissimilarity"]),
            "r--",
        )
        # add correlation to plot
        try:
            corr = stats.pearsonr(
                cur_sbj_df["Neural Dissimilarity"], cur_sbj_df["Pose Distance"]
            )
        except ValueError:
            print("Error calculating the correlation")
            corr = [0, 0]

        plt.text(
            0.5,
            0.5,
            "r = " + str(round(corr[0], 2)),
            horizontalalignment="center",
            verticalalignment="center",
            transform=ax.transAxes,
        )
        file_special = ""
        if abs(corr[0]) >= 0.7:
            print("Correlation above 0.7 here!!")
            file_special = "_notice"

        plt.xlabel("Neural Dissimilarity")
        plt.ylabel("Pose Average")
        plt.title(pat_id_curr)
        plt.savefig(
            os.path.join(
                pose_exp_params["sp"],
                metric_name,
                pose_combo,
                "pose_avg_vs_neural_" + pat_id_curr + file_special + ".png",
            ),
            bbox_inches="tight",
        )

        # plot with error bars
        fig, ax = plt.subplots()

        cur_sbj_df = pas_cur_freq_df[pas_cur_freq_df["Participant"]
                                     == pat_id_curr]
        sns.scatterplot(
            x=cur_sbj_df["Neural Dissimilarity"], y=cur_sbj_df["Pose Distance"]
        )
        plt.errorbar(
            x=cur_sbj_df["Neural Dissimilarity"],
            y=cur_sbj_df["Pose Distance"],
            yerr=cur_sbj_df["Pose Distance Std"],
            fmt="o",
        )

        plt.xlabel("Neural Dissimilarity")
        plt.ylabel("Pose Average")
        plt.title(pat_id_curr)
        plt.savefig(
            os.path.join(
                pose_exp_params["sp"],
                metric_name,
                pose_combo,
                "pose_avg_std_vs_neural_" + pat_id_curr + file_special + ".png",
            ),
            bbox_inches="tight",
        )


def plot_pose_vs_neural_metric(
    ecog_pose_df, freq_bands, pats_ids_in, pose_exp_params, metric_name, pose_combo
):
    pose_combo = "_".join(pose_combo)
    metric_name = metric_name.split(" ")[1]
    if not os.path.exists(os.path.join(pose_exp_params["sp"], metric_name, pose_combo)):
        os.makedirs(os.path.join(
            pose_exp_params["sp"], metric_name, pose_combo))
    print(os.path.join(pose_exp_params["sp"], metric_name, pose_combo))
    plot_by_freq(ecog_pose_df, freq_bands,
                 pose_exp_params, metric_name, pose_combo)
    plot_by_pat(ecog_pose_df, pats_ids_in,
                pose_exp_params, metric_name, pose_combo)
    plt.close()


def load_pose_data(exp_params):
    pose_proj_mat_sp = (
        exp_params["sp"] + exp_params["dataset"] +
        exp_params["experiment_folder"]
    )
    if not os.path.exists(pose_proj_mat_sp):
        os.makedirs(pose_proj_mat_sp)

    pose_data = ECoG_Data(exp_params, pose_proj_mat_sp)

    return pose_data


def pose_joint_angles(x, subset):
    # calculates the joint angles of elbow and shoulder
    # on the given data through time
    # assumes data is in the form (joint, time)

    # subset must contain all angles to be calculated
    pose_marks = [
        "L_Ear",
        "L_Elbow",
        "L_Shoulder",
        "L_Wrist",
        "Nose",
        "R_Ear",
        "R_Elbow",
        "R_Shoulder",
        "R_Wrist",
    ]
    if subset != pose_marks:
        print("subset must contain all pose marks for joint angle calculation")
        return None

    # get the joint angles
    limbs = [["Nose", "L_Shoulder"],
             ["L_Shoulder", "L_Elbow"],
             ["L_Elbow", "L_Wrist"],
             ["Nose", "R_Shoulder"],
             ["R_Shoulder", "R_Elbow"],
             ["R_Elbow", "R_Wrist"],
             ]

    # calculate each limb vector
    limb_vectors = {}
    for limb in limbs:
        # need x and y coords for each limb
        limb0_ind = [subset.index(limb[0]), subset.index(
            limb[0]) + len(subset)]
        limb1_ind = [subset.index(limb[1]), subset.index(
            limb[1]) + len(subset)]
        limb0 = x[limb0_ind, :]
        limb1 = x[limb1_ind, :]

        # get the normalized vector between the two joints
        limb_dist = limb1 - limb0
        norm = np.linalg.norm(limb_dist, axis=0)
        limb_vec = limb_dist / norm
        limb_vectors[limb[0] + "_" + limb[1]] = limb_vec

    # calculate the angles between defined limbs
    angles = [["Nose_L_Shoulder", "L_Shoulder_L_Elbow"],
              ["L_Shoulder_L_Elbow", "L_Elbow_L_Wrist"],
              ["Nose_R_Shoulder", "R_Shoulder_R_Elbow"],
              ["R_Shoulder_R_Elbow", "R_Elbow_R_Wrist"],
              ]
    joint_angles = np.zeros((len(angles), x.shape[1]))
    for i, angle in enumerate(angles):
        # get the two vectors
        vec0 = limb_vectors[angle[0]]
        vec1 = limb_vectors[angle[1]]

        # calculate the angle between the two vectors
        # is there a way to vectorize this?
        for t in range(x.shape[1]):
            joint_angles[i, t] = np.arccos(
                np.clip(np.dot(vec0[:, t], vec1[:, t]), -1.0, 1.0))

    return joint_angles


def joint_joint_orientation(x, subset):
    # find the unit vector between each set of joints
    # assume x is in the form (joint, time)
    # extract joint coords first
    n_joints = len(subset)
    time_len = x.shape[1]
    dist_mat = np.zeros((n_joints, n_joints, time_len, 2))
    for i, s_i in enumerate(subset):
        for j, s_j in enumerate(subset):
            limbi_ind = [subset.index(s_i), subset.index(s_i) + len(subset)]
            limbj_ind = [subset.index(s_j), subset.index(s_j) + len(subset)]
            limb_i = x[limbi_ind, :]
            limb_j = x[limbj_ind, :]
            # get the normalized vector between the two joints
            limb_dist = limb_i - limb_j
            norm = np.linalg.norm(limb_dist, axis=0)
            limb_vec = limb_dist / norm
            dist_mat[i, j, :] = limb_vec.T

    up_inds = np.triu_indices(n_joints, k=1)
    dist_mat = dist_mat[up_inds]
    return dist_mat


def line_line_angle(jj_orient):
    # calculate the angle between two lines
    # assume jj_orient is in the form (line, time, 2)
    # first get the angle between each pair of joints
    comps = list(combinations(np.arange(jj_orient.shape[0]), 2))
    time_len = jj_orient.shape[1]
    angle_mat = np.zeros((len(comps), time_len))
    for c in range(len(comps)):
        for t in range(time_len):
            i_ind = comps[c][0]
            j_ind = comps[c][1]
            angle_mat[c, t] = np.arccos(
                np.clip(np.dot(jj_orient[i_ind, t], jj_orient[j_ind, t]), -1.0, 1.0))

    return angle_mat


def calc_pairwise_pose_dist(pose_data, subset, dist_fn_name=stats.pearsonr):
    """
    Calculate pairwise distance between all poses in the dataset
    """
    n_poses = pose_data.shape[0]
    pose_dist_mat = np.zeros((n_poses, n_poses))
    for i in tqdm(range(n_poses)):
        for j in range(n_poses):
            pose_i = pose_data[i, :]
            pose_j = pose_data[j, :]

            if dist_fn_name == scipy.spatial.distance.mahalanobis:
                v1 = np.cov(pose_data.reshape(n_poses, -1), rowvar=False)
                v1 = np.linalg.inv(v1)
                dist = dist_fn_name(
                    pose_i.flatten(), pose_j.flatten(), v1)

            elif dist_fn_name == percentage_correct_parts:
                dist = dist_fn_name(
                    pose_i, pose_j, subset
                )
            elif dist_fn_name == percentage_detected_joints:
                dist = dist_fn_name(
                    pose_i, pose_j, subset
                )
            elif dist_fn_name == percentage_key_points:
                dist = dist_fn_name(
                    pose_i, pose_j, subset
                )
            elif dist_fn_name == wrap_oks:
                dist = dist_fn_name(
                    pose_i, pose_j
                )
            elif dist_fn_name == mean_joint_angle_error:
                dist = dist_fn_name(
                    pose_i, pose_j, subset
                )
            elif dist_fn_name == joint_angle_dtw:
                dist = dist_fn_name(
                    pose_i, pose_j, subset
                )
            else:
                try:
                    # print("Need to flatten data for " + str(dist_fn_name))
                    dist = dist_fn_name(
                        pose_i.flatten(), pose_j.flatten()
                    )
                except:
                    dist = dist_fn_name(
                        pose_i.reshape(1, -1), pose_j.reshape(1, -1)
                    )

            try:
                pose_dist_mat[i, j] = dist[0]
            except:
                # else:
                pose_dist_mat[i, j] = dist

    return pose_dist_mat


def pose_ll_angles(cur_sbj_pose_data):
    n_trials = cur_sbj_pose_data.shape[0]
    subset = [
        "L_Ear",
        "L_Elbow",
        "L_Shoulder",
        "L_Wrist",
        "Nose",
        "R_Ear",
        "R_Elbow",
        "R_Shoulder",
        "R_Wrist",
    ]
    for t in tqdm(range(n_trials)):
        # cur_trial_ll_angles = pose_joint_angles(
        #     cur_sbj_pose_data[t, :, :], subset)
        # cur_trial_ll_angles = np.expand_dims(cur_trial_ll_angles, axis=0)
        cur_trial_pose_data = cur_sbj_pose_data[t, :, :]
        jj_orient = joint_joint_orientation(cur_trial_pose_data, subset)
        cur_trial_ll_angles = line_line_angle(jj_orient)
        cur_trial_ll_angles = np.expand_dims(cur_trial_ll_angles, axis=0)
        if t == 0:
            ll_angles = cur_trial_ll_angles
        else:
            ll_angles = np.concatenate(
                (ll_angles, cur_trial_ll_angles), axis=0)

    return ll_angles


def bootstrap_ecog_data(split_sbj_eLFO_ECoG_data, cur_classes, per_class_boot_inds):
    if split_sbj_eLFO_ECoG_data == []:
        return [], [], [], [], []

    # take random 80% from each class
    ecog_bootstrap_data = []
    for c, cur_class in enumerate(cur_classes):
        cur_class_data = split_sbj_eLFO_ECoG_data[c]
        cur_class_data = cur_class_data[per_class_boot_inds[c], :, :]
        ecog_bootstrap_data.append(cur_class_data)

    # now can concat the trials and normalize
    print("Concatenating Trials")
    concat_ECoG_data, trial_dim, sr_dim, chan_dim = mu.concat_trials(
        ecog_bootstrap_data
    )
    assert mu.check_repeated_trials(
        concat_ECoG_data, cur_classes, trial_dim, sr_dim, chan_dim
    ), "Trials repeated somewhere"

    print("Normalizing Data")
    norm_concat_ECoG_data = mu.normalize_data(concat_ECoG_data)
    assert mu.check_repeated_trials(
        norm_concat_ECoG_data, cur_classes, trial_dim, sr_dim, chan_dim
    ), "Trials repeated somewhere"

    return cur_classes, trial_dim, sr_dim, chan_dim, norm_concat_ECoG_data


def bootstrap_pose_data(split_sbj_joint_angles, cur_classes, per_class_boot_inds):
    pose_bootstrap_data = []
    for c, cur_class in enumerate(cur_classes):
        cur_class_data = split_sbj_joint_angles[c]
        np.random.shuffle(cur_class_data)
        cur_class_data = cur_class_data[per_class_boot_inds[c], :, :]
        pose_bootstrap_data.append(cur_class_data)

    return pose_bootstrap_data


def bootstrap_data(split_sbj_joint_angles, split_sbj_eLFO_ECoG_data, pose_data, ecog_data, class_dict, pat_curr, cur_classes):
    # need to treat pose and ecog data differently
    # pose data needs to extract joint/line angles

    # need to have matching 80% of trials for each class
    ecog_data.get_single_sbj(pat_curr)
    ecog_int_labels = np.array(ecog_data.sbj_y_test[:, 0], dtype=int)
    pose_int_labels = np.array(pose_data.sbj_y_test[:, 0], dtype=int)
    per_class_trials = [min(len(np.where(ecog_int_labels == cur_class)[0]),
                            len(np.where(pose_int_labels == cur_class)[0]))
                        for cur_class in class_dict.keys()]
    per_class_boot_inds = []
    for c, cur_class in enumerate(class_dict.keys()):
        cur_class_trials = np.arange(per_class_trials[c])
        np.random.shuffle(cur_class_trials)
        cur_class_trials = cur_class_trials[:int(
            PERCENT_BOOTSTRAP*cur_class_trials.shape[0])]
        per_class_boot_inds.append(cur_class_trials)

    pose_bootstrap_data = bootstrap_pose_data(
        split_sbj_joint_angles, cur_classes, per_class_boot_inds)

    cur_classes, trial_dim, sr_dim, chan_dim, bootstrap_ecog = bootstrap_ecog_data(
        split_sbj_eLFO_ECoG_data, cur_classes, per_class_boot_inds)

    return cur_classes, trial_dim, sr_dim, chan_dim, bootstrap_ecog, pose_bootstrap_data


def preprocess_pose(pose_data, pat_id_curr, class_dict, pose_exp_params):
    # get joint angles
    # need to do here because it takes a while to calc
    pose_data.get_single_sbj(pat_id_curr)
    # trim to proper time
    pose_data.sbj_X_test = mu.trim_filtered_data(
        [-1, 1], pose_exp_params["ecog_srate"], pose_data.sbj_X_test
    )
    print(pose_data.sbj_X_test.shape)
    if pose_exp_params["feature_type"] == "exp_pose":
        labels = pose_data.sbj_y_test
    else:
        print("Extracting angles")
        pose_joint_angles = pose_ll_angles(pose_data.sbj_X_test)
        print(pose_joint_angles.shape)
        pose_data.sbj_X_test = pose_joint_angles

        # convert labels to ints first
        labels = np.array(pose_data.sbj_y_test[:, 0], dtype=int)

    split_sbj_joint_angles, cur_classes = mu.split_into_classes(
        class_dict,
        labels,
        pose_data.sbj_X_test,
    )

    return split_sbj_joint_angles, cur_classes


def preprocess_ecog(ecog_data, pat_curr, sbj_sp, freq_bands, freq_name, class_dict):
    ecog_data.get_single_sbj(pat_curr)
    envelope_freq_sbj_X_test, cur_freq_sbj_X_test = mu.extract_analytic_signal(
        freq_bands,
        freq_name,
        ecog_exp_params["ecog_srate"],
        ecog_data.sbj_X_test,
    )

    if cur_freq_sbj_X_test.tolist() != []:
        mu.plot_raw_and_freq(freq_name, sbj_sp, ecog_data, cur_freq_sbj_X_test)
        # do i need this function after adding padding to filter?
        envelope_freq_sbj_X_test = mu.trim_filtered_data(
            [-1, 1], ecog_exp_params["ecog_srate"], cur_freq_sbj_X_test
        )

        split_sbj_eLFO_ECoG_data, cur_classes = mu.split_into_classes(
            class_dict,
            ecog_data.sbj_y_test[:, 0],
            envelope_freq_sbj_X_test,
        )

        # check that all trials distinct here first
        assert mu.check_repeated_trials(
            split_sbj_eLFO_ECoG_data, cur_classes
        ), "Trials repeated somewhere"

    else:
        split_sbj_eLFO_ECoG_data = []
        cur_classes = []

    return split_sbj_eLFO_ECoG_data, cur_classes


def calc_class_vs_class_pose_dist(pose_class1, pose_class2, dist_fn_name=stats.pearsonr):
    """
    Calculate pairwise distance between all poses in the dataset
    """
    n_poses_1 = pose_class1.shape[0]
    n_poses_2 = pose_class2.shape[0]
    pose_dists = np.zeros((n_poses_1 * n_poses_2))
    counter = 0
    for i in tqdm(range(n_poses_1)):
        for j in range(n_poses_2):
            pose_i = pose_class1[i, :]
            pose_j = pose_class2[j, :]
            try:
                # print("Need to flatten data for " + str(dist_fn_name))
                dist = dist_fn_name(
                    pose_i.flatten(), pose_j.flatten()
                )
            except:
                dist = dist_fn_name(
                    pose_i.reshape(1, -1), pose_j.reshape(1, -1)
                )

            try:
                dist = dist[0]
                pose_dists[counter] = dist
            except:
                pose_dists[counter] = dist

            counter += 1

    return pose_dists


def calc_pose_metric(pose_bootstrap_data, class_dict, dist_fn_name):
    subset = [
        "L_Ear",
        "L_Elbow",
        "L_Shoulder",
        "L_Wrist",
        "Nose",
        "R_Ear",
        "R_Elbow",
        "R_Shoulder",
        "R_Wrist",
    ]

    # class_metric_avg = {}
    # class_metric_std = {}
    class_metric_avg = []
    class_metric_std = []
    # for each class combo, calculate the metric
    class_combos = list(itertools.combinations(class_dict.keys(), 2))
    for c, cur_class in enumerate(class_combos):
        if pose_exp_params["feature_type"] == "exp_pose":
            cur_class = (cur_class[0] - 1, cur_class[1] - 1)
        class_1_data = pose_bootstrap_data[cur_class[0]]
        class_2_data = pose_bootstrap_data[cur_class[1]]
        # calc the dist matrix of metric
        pose_dist_mat = calc_class_vs_class_pose_dist(
            class_1_data,
            class_2_data,
            dist_fn_name=dist_fn_name,
        )
        # get the mean and std
        # class_vs_key = class_dict[cur_class[0]] + \
        #     " vs " + class_dict[cur_class[1]]
        # class_metric_avg[class_vs_key] = np.mean(pose_dist_mat)
        # class_metric_std[class_vs_key] = np.std(pose_dist_mat)
        class_metric_avg.append(np.mean(pose_dist_mat))
        class_metric_std.append(np.std(pose_dist_mat))

    return class_metric_avg, class_metric_std


def main(pose_exp_params, ecog_exp_params):
    metrics = [stats.pearsonr]
    bootstrap_n = N_BOOTSTRAP
    # ecog_exp_params["pats_ids_in"]
    pats_ids_in = pose_exp_params["pats_ids_in"]
    pose_proj_mat_sp = pose_exp_params["sp"] + \
        pose_exp_params["dataset"] + pose_exp_params["experiment_folder"]
    ecog_proj_mat_sp = ecog_exp_params["sp"] + \
        ecog_exp_params["dataset"] + ecog_exp_params["experiment_folder"]
    freq_bands = ecog_exp_params["freq_bands"]
    # NOTE: only doing Beta for right now,
    # will probably need to expand later, if we like this result
    freq_name = 'Beta'
    freq_dim = 10
    class_dict = ecog_exp_params["class_dict"]
    class_dict = {int(cur_key): val for cur_key, val in class_dict.items()}

    subset = ["L_Ear", "L_Elbow", "L_Shoulder",  "L_Wrist",
              "Nose", "R_Ear", "R_Elbow", "R_Shoulder", "R_Wrist"]

    # get pose data
    pose_data = load_pose_data(pose_exp_params)
    # get ecog data
    if type(ecog_exp_params["test_day"]) == list:
        ecog_exp_params["test_day"] = "last"
    ecog_data = ECoG_Data(ecog_exp_params, ecog_proj_mat_sp)
    if ecog_exp_params["rois"]:
        ecog_data.X_test = mu.roi_proj(
            ecog_data.X_test,
            ecog_data.sbj_order_test,
            ecog_data.nROIs,
            ecog_data.proj_mat_out,
        )
    else:
        ecog_data.roi_centroids = None
    print(ecog_data.X_test.shape)

    for metric in metrics:
        all_bootstrap_pca = []
        all_bootstrap_metric_avg = []
        all_bootstrap_metric_std = []
        for i, pat_id_curr in enumerate(pats_ids_in):
            sbj_sp = ecog_proj_mat_sp + pat_id_curr + "/"
            if not os.path.exists(sbj_sp):
                os.makedirs(sbj_sp)

            split_sbj_joint_angles, cur_classes = preprocess_pose(
                pose_data, pat_id_curr, class_dict, pose_exp_params)
            split_sbj_eLFO_ECoG_data, cur_classes = preprocess_ecog(
                ecog_data, pat_id_curr, sbj_sp, freq_bands, freq_name, class_dict)

            cur_sbjs_pca = []
            cur_sbjs_metric_avg = []
            cur_sbjs_metric_std = []
            for bn in tqdm(range(bootstrap_n)):
                # get bootstrap data
                cur_classes, trial_dim, sr_dim, chan_dim, bootstrap_ecog, pose_bootstrap_data = bootstrap_data(
                    split_sbj_joint_angles, split_sbj_eLFO_ECoG_data, pose_data, ecog_data, class_dict, pat_id_curr, cur_classes)

                # calculate the ecog subspaces
                class_pca, reduced_class_ECoG_data = mu.calc_class_pca(
                    trial_dim, sr_dim, chan_dim, sbj_sp, freq_name, bootstrap_ecog
                )
                cur_sbjs_pca.append(class_pca)

                # calculate the pose metrics
                class_metric_avg, class_metric_std = calc_pose_metric(
                    pose_bootstrap_data, class_dict, metric)
                cur_sbjs_metric_avg.append(class_metric_avg)
                cur_sbjs_metric_std.append(class_metric_std)

            all_bootstrap_pca.append(cur_sbjs_pca)
            all_bootstrap_metric_avg.append(cur_sbjs_metric_avg)
            all_bootstrap_metric_std.append(cur_sbjs_metric_std)

        all_bootstrap_pca = np.array(all_bootstrap_pca)
        freq_red_dim = [freq_dim for i in range(len(all_bootstrap_pca))]
        grass_dist, pa_by_freq = mu.get_pa_per_pat(
            pats_ids_in,
            np.arange(bootstrap_n),
            class_dict,
            freq_red_dim,
            all_bootstrap_pca
        )
        pa_by_freq = np.array(pa_by_freq)
        print(pa_by_freq.shape)
        if pose_exp_params["feature_type"] == "exp_pose":
            class_dict = {int(cur_key) - 1: val for cur_key,
                          val in class_dict.items()}
        pa_df = mu.get_summed_pas_df(pats_ids_in,
                                     np.arange(bootstrap_n),
                                     class_dict,
                                     pa_by_freq,
                                     freq_dim)
        pa_df = pa_df.rename(
            columns={'Frequency': 'Participant', 'Participant': 'Bootstrap Iteration'})
        pa_df['Frequency'] = [freq_name for i in range(len(pa_df))]
        pa_df['Pose Distance'] = np.array(all_bootstrap_metric_avg).flatten()
        pa_df['Pose Distance Std'] = np.array(
            all_bootstrap_metric_std).flatten()
        print("making plots")
        print(pa_df.head())
        plot_pose_vs_neural_metric(
            pa_df, ['LFO'], pats_ids_in, pose_exp_params, str(metric), subset)

        # save the dataframe
        print("saving to ", pose_proj_mat_sp)
        pa_df.to_csv(pose_proj_mat_sp + "pose_vs_ecog_" + str(metric) + ".csv")


if __name__ == "__main__":
    try:
        pose_filename = sys.argv[1]
        ecog_filename = sys.argv[2]
    except IndexError:
        raise SystemExit(
            f"Usage: {sys.argv[0]} <json file of experiment parameters for pose> <json file of experiment parameters for ecog>")
    with open(pose_filename) as f1:
        pose_exp_params = json.load(f1)
        print(pose_exp_params["comment"])

    with open(ecog_filename) as f2:
        ecog_exp_params = json.load(f2)
        print(ecog_exp_params["comment"])

    main(pose_exp_params, ecog_exp_params)
