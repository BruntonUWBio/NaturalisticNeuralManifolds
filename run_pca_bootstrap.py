import numpy as np
import pandas as pd

import json
from tqdm import tqdm
import sys
import os
import pdb

from src.data_utils import ECoG_Data
import src.manifold_u as mu


def get_ecog_data(freq_bands,
                  class_dict,
                  exp_params,
                  proj_mat_sp,
                  days_tested,
                  pats_ids_in):
    # get all the ECoG data for each frequency, participant, and day
    bootstrap_data_list = []
    # may be worth to swap this loop inside of days
    for f, freq_band in enumerate(freq_bands):
        print("Running for freq_band: {}".format(freq_band))

        for k, day in enumerate(days_tested):
            print("Running Manifolds for ", day)
            # load in the data for the current day
            exp_params["test_day"] = str(day)
            manifold_ECoG_data = ECoG_Data(exp_params, proj_mat_sp)
            if exp_params["rois"]:
                manifold_ECoG_data.X_test = mu.roi_proj(
                    manifold_ECoG_data.X_test,
                    manifold_ECoG_data.sbj_order_test,
                    manifold_ECoG_data.nROIs,
                    manifold_ECoG_data.proj_mat_out,
                )
            else:
                manifold_ECoG_data.roi_centroids = None
            print(manifold_ECoG_data.X_test.shape)

            for i, pat_id_curr in enumerate(pats_ids_in):
                print("Running Manifolds for ", pat_id_curr)
                sbj_sp = proj_mat_sp + pat_id_curr + "/" + str(day) + "/"
                if not os.path.exists(sbj_sp):
                    os.makedirs(sbj_sp)

                manifold_ECoG_data.get_single_sbj(pat_id_curr)
                if not exp_params["rois"]:
                    manifold_ECoG_data.sbj_X_test = (
                        manifold_ECoG_data.remove_zero_channels(
                            manifold_ECoG_data.sbj_X_test
                        )
                    )

                # get the data for bootstrap
                _, cur_freq_sbj_X_test = mu.extract_analytic_signal(
                    freq_bands,
                    freq_band,
                    exp_params["ecog_srate"],
                    manifold_ECoG_data.sbj_X_test,
                )
                if cur_freq_sbj_X_test.tolist() != []:
                    cur_freq_sbj_X_test = mu.trim_filtered_data(
                        [-1, 1], exp_params["ecog_srate"], cur_freq_sbj_X_test
                    )
                    split_sbj_eLFO_ECoG_data, cur_classes = mu.split_into_classes(
                        class_dict,
                        manifold_ECoG_data.sbj_y_test[:, 0],
                        cur_freq_sbj_X_test,
                    )
                else:
                    split_sbj_eLFO_ECoG_data = [[]
                                                for i in range(len(class_dict))]
                for c, class_name in enumerate(class_dict):
                    bootstrap_data_list.append(
                        [
                            freq_band,
                            pat_id_curr,
                            day,
                            class_name,
                            split_sbj_eLFO_ECoG_data[c],
                        ]
                    )
    ecog_data_df = pd.DataFrame(
        bootstrap_data_list,
        columns=["Frequency", "Participant", "Day", "Movement", "ECoG Data"],
    )

    return ecog_data_df


def extract_ecog_data(ecog_data_df, class_dict, pat_id_curr, freq_band):
    pat_bootstrap_data = []
    for m, mvmt in enumerate(class_dict):
        # get the data for this participant and day
        cur_freq_sbj_data = ecog_data_df.loc[
            (ecog_data_df["Participant"] == pat_id_curr)
            & (ecog_data_df["Movement"] == mvmt)
            & (ecog_data_df["Frequency"] == freq_band)
        ]["ECoG Data"].values
        # should give me data in shape (days, trials, samples, channels)
        pat_bootstrap_data.append(cur_freq_sbj_data)

    return pat_bootstrap_data


def calc_bootstrap_pca(freq_bands,
                       class_dict,
                       exp_params,
                       proj_mat_sp,
                       days_tested,
                       pats_ids_in,
                       ecog_data_df,
                       n_samples=100):

    for f, freq_band in enumerate(freq_bands):
        bootstrap_pcas = np.empty(
            (len(pats_ids_in), len(class_dict), len(days_tested), n_samples), dtype=object
        )
        bootstrap_pcas[:] = np.nan

        for p, pat_id_curr in enumerate(pats_ids_in):
            print("Running Bootstrap for ", pat_id_curr)

            sbj_sp = proj_mat_sp + pat_id_curr + "/bootstrap/"
            if not os.path.exists(sbj_sp):
                os.makedirs(sbj_sp)

            # should be (mvmts, days, trials, samples, channels)
            pat_bootstrap_data = extract_ecog_data(
                ecog_data_df, class_dict, pat_id_curr, freq_band)

            # now we can do the bootstrap
            for m, mvmt in enumerate(class_dict):
                for d, day in enumerate(days_tested):
                    # should be of shape (n_trials, n_chans, n_time)
                    cur_ecog_data = pat_bootstrap_data[m][d]

                    for s in tqdm(range(n_samples)):
                        if cur_ecog_data == []:
                            bootstrap_pcas[p, m, d, s] = []
                            continue
                        # get the indices for the current sample
                        sample_inds = np.random.choice(
                            cur_ecog_data.shape[0], cur_ecog_data.shape[0], replace=True
                        )
                        # get the data for the current sample
                        cur_sample_data = cur_ecog_data[sample_inds, ...]
                        cur_samp_concat, trial_dim, sr_dim, chan_dim = mu.concat_trials(
                            [cur_sample_data]
                        )
                        cur_samp_norm = mu.normalize_data(cur_samp_concat)

                        # now run pca on the data
                        samps_pca, red_data = mu.calc_class_pca(
                            trial_dim, sr_dim, chan_dim, sbj_sp, freq_band, cur_samp_norm
                        )
                        # print(samps_pca[0])
                        # samps pca should return with just one pca object for the sample
                        bootstrap_pcas[p, m, d, s] = samps_pca[0]

        # save the pca objects
        print("saving current frequency bootstrapped PCA objects")
        np.save(proj_mat_sp + freq_band +
                "_bs_pca.npy", bootstrap_pcas)


def main():
    rand_seed = 1762
    np.random.seed(rand_seed)

    try:
        json_filename = sys.argv[1]
    except IndexError:
        raise SystemExit(
            f"Usage: {sys.argv[0]} <json file of experiment parameters>")
    with open(json_filename) as f:
        exp_params = json.load(f)

    freq_bands = exp_params["freq_bands"]
    class_dict = exp_params["class_dict"]
    class_dict = {int(cur_key): val for cur_key, val in class_dict.items()}
    class_color = exp_params["class_color"]
    class_color = {int(cur_key): val for cur_key, val in class_color.items()}
    pats_ids_in = exp_params["pats_ids_in"]

    proj_mat_sp = (
        exp_params["sp"] + exp_params["dataset"] +
        exp_params["experiment_folder"]
    )
    if not os.path.exists(proj_mat_sp):
        os.makedirs(proj_mat_sp)

    days_tested = exp_params["test_day"]

    freq_bands = {"Beta": freq_bands["Beta"]}

    ecog_data_df = get_ecog_data(freq_bands,
                                 class_dict,
                                 exp_params,
                                 proj_mat_sp,
                                 days_tested,
                                 pats_ids_in)

    # now we can do the bootstrap
    # ie. make all the PCA objects for bootstrapped data
    # saves to appropriate folder
    calc_bootstrap_pca(freq_bands,
                       class_dict,
                       exp_params,
                       proj_mat_sp,
                       days_tested,
                       pats_ids_in,
                       ecog_data_df)


if __name__ == "__main__":
    sys.exit(main())
