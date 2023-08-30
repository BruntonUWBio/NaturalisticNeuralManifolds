import numpy as np
import pandas as pd

import pickle
import json
from tqdm import tqdm
import sys
import os
import pdb

import src.manifold_utils as mu


def cross_movement_comps(pats_ids_in,
                         days_tested,
                         class_dict,
                         all_sbjs_pca,
                         freq_red_dim,
                         freq_band,
                         freq_null_data_pa,
                         freq_cross_move_pas,
                         freq_cross_move_nd_df):
    cross_move_pas = []
    for p, pat in enumerate(pats_ids_in):
        pat_pas = []
        for d, day in enumerate(days_tested):
            pca_manifolds = all_sbjs_pca[p, d, :]
            pas = mu.calc_comp_dim_pas(
                class_dict, pca_manifolds, freq_red_dim)
            pat_pas.append(pas)
        cross_move_pas.append(pat_pas)
    cross_move_pas = np.array(cross_move_pas)
    # make sure to remove rest comparisons
    class_comps = mu.get_pa_comparison_names(class_dict)
    no_rest_class_comps = {comp_key: class_comps[comp_key]
                           for comp_key in class_comps if 'rest' not in class_comps[comp_key]}
    no_rest_class_comp_inds = [
        comp_key for comp_key in no_rest_class_comps]
    no_rest_cross_move_pas = cross_move_pas[:,
                                            :, no_rest_class_comp_inds, :]
    print("Cross_movement final PAs shape:", no_rest_cross_move_pas.shape)
    # cross_move_pa of shape (sbj, day, movement comp, reduced dim)
    freq_cross_move_pas[freq_band] = no_rest_cross_move_pas

    # save the neural dissimilarity as a dataframe
    nd_df = mu.one_freq_get_summed_pas_df(cross_move_pas,
                                          freq_red_dim,
                                          freq_band,
                                          pats_ids_in,
                                          days_tested,
                                          class_dict,
                                          freq_null_data_pa)
    # make sure to remove rest comparisons
    nd_df = nd_df[nd_df["Movement Comparison"].str.contains(
        "rest") == False]
    nd_df = nd_df.reset_index(drop=True)

    freq_cross_move_nd_df = freq_cross_move_nd_df.append(nd_df)
    print(freq_cross_move_nd_df)

    return freq_cross_move_pas, freq_cross_move_nd_df


def cross_day_comps(exp_params,
                    all_sbjs_pca,
                    proj_mat_sp,
                    freq_red_dim,
                    freq_band,
                    pats_ids_in,
                    days_tested,
                    class_dict,
                    cross_days_null_data_pa,
                    freq_cross_day_pas,
                    freq_cross_day_bs_pas,
                    freq_cross_day_nd_df,
                    freq_cross_day_lagged_nd_df):
    # ultimately need to get PAs as shape (sbjs, movements, day comps, reduced dim)
    all_pca = all_sbjs_pca.transpose((0, 2, 1))
    cross_days_pas = []
    for p, pat in enumerate(pats_ids_in):
        pat_pas = []
        for m, mvmt in enumerate(class_dict):
            pca_manifolds = all_sbjs_pca[p, m, :]
            pas = mu.calc_comp_dim_pas(
                days_tested, pca_manifolds, freq_red_dim)
            pat_pas.append(pas)
        cross_days_pas.append(pat_pas)
    cross_days_pas = np.array(cross_days_pas)
    freq_cross_day_pas[freq_band] = cross_days_pas

    # fantastic, now can start the DF
    # the dataframe will have the following columns:
    # 1) frequency band
    # 2) Movement
    # 3) subject id
    # 4) Day Comparison
    # 4) Summed PA
    summed_pas = []
    comp_names = mu.get_pa_comparison_names(days_tested)
    # for real data
    cur_norm_pa = mu.calc_norm_sum_pa(freq_red_dim, cross_days_pas)
    # print(cur_norm_pa.shape)
    for s, cur_sbj in enumerate(pats_ids_in):
        for m, cur_mvmt in enumerate(class_dict):
            for c, cur_comp in enumerate(comp_names):
                # print(f, s, c)
                summed_pas.append(
                    [freq_band, cur_sbj, class_dict[cur_mvmt], comp_names[c], cur_norm_pa[s][m][c]])

    # also get the bootstrapped PAs
    # unfortunately could only run for Beta so far (and the data is so large will keep that way)
    # ultimately need to get PAs as shape (sbjs, movements, days, bootstrap samps comps, reduced dim)
    if freq_band == 'Beta':
        bootstrap_pca = np.load(
            proj_mat_sp + freq_band + "_bs_pca.npy", allow_pickle=True)
        print("Bootstrap PCA shape:", bootstrap_pca.shape)
        total_samps = bootstrap_pca.shape[-1]
        samps_array = np.arange(total_samps)
        cross_days_bs_pas = []
        for p, pat in enumerate(pats_ids_in):
            pat_bs_pas = []
            for m, mvmt in enumerate(class_dict):
                mvmt_bs_pas = []
                for d, day in enumerate(days_tested):
                    cur_bs_pca = bootstrap_pca[p, m, d, :]
                    cur_bs_pas = mu.calc_comp_dim_pas(
                        samps_array, cur_bs_pca, freq_red_dim)
                    mvmt_bs_pas.append(cur_bs_pas)
                pat_bs_pas.append(mvmt_bs_pas)
            cross_days_bs_pas.append(pat_bs_pas)
        cross_days_bs_pas = np.array(cross_days_bs_pas)
        print("Cross-days bootstrapped PAs shape:",
              cross_days_bs_pas.shape)
        freq_cross_day_bs_pas[freq_band] = cross_days_bs_pas

        # add bootstrap to DF
        for s, pat_id_curr in enumerate(pats_ids_in):
            cur_norm_pa = mu.calc_norm_sum_pa(
                freq_red_dim, cross_days_bs_pas[s])
            # print(cur_norm_pa.shape)
            for m, cur_mvmt in enumerate(class_dict):
                for d, day in enumerate(days_tested):
                    for c in range(cross_days_bs_pas.shape[3]):
                        summed_pas.append(
                            [freq_band, pat_id_curr, class_dict[cur_mvmt], (day + ' vs ' + day), cur_norm_pa[m][d][c]])

        # also only do this on the Beta pass through
        # get the null data PAs
        n_samples = 1000
        n_day_comps = mu.get_num_pa_comparisons(days_tested)[1]
        updated_null_pa = np.empty((1, len(pats_ids_in), len(
            class_dict), n_samples, n_day_comps, freq_red_dim))
        updated_null_pa[:] = np.nan
        for s, pat_id_curr in enumerate(pats_ids_in):
            for m, mvmt in enumerate(class_dict):
                if np.array(cross_days_null_data_pa[0][s][m]).shape[3] < n_day_comps:
                    for i in range(np.array(cross_days_null_data_pa[0][s][m]).shape[3]):
                        updated_null_pa[0, s, m, :, i, :] = np.squeeze(
                            cross_days_null_data_pa[0][s][m][:, :, :, i, 0:freq_red_dim])

                else:
                    updated_null_pa[0, s, m, :, :, :] = np.squeeze(
                        cross_days_null_data_pa[0][s][m][:, :, :, :, 0:freq_red_dim])

        print(updated_null_pa.shape)

        # add null data to DF
        norm_null_pa = mu.calc_norm_sum_pa(
            freq_red_dim, updated_null_pa[0])
        print(norm_null_pa.shape)
        for s, cur_sbj in enumerate(pats_ids_in):
            for m, cur_mvmt in enumerate(class_dict):
                for n in range(n_samples):
                    for c, cur_comp in enumerate(comp_names):
                        # print(f, s, c)
                        summed_pas.append(
                            ['Null', 'Null', class_dict[cur_mvmt], 'Null', norm_null_pa[s][m][n][c]])

    # now make into dataframe
    nd_df = pd.DataFrame(summed_pas, columns=['Frequency',
                                              'Participant',
                                              'Movement',
                                              'Day Comparison',
                                              'Neural Dissimilarity'])
    print(nd_df)
    # remove the rest data
    nd_df = nd_df[nd_df["Movement"] != "rest"]
    nd_df = nd_df.reset_index(drop=True)
    freq_cross_day_nd_df = freq_cross_day_nd_df.append(nd_df)

    # now do the lagged comparisons
    # making data for 'autocorrelation' plot
    zero_days = ['3 vs 3', '4 vs 4', '5 vs 5', '6 vs 6', '7 vs 7']
    one_days = ['3 vs 4', '4 vs 5', '5 vs 6', '6 vs 7']
    two_days = ['3 vs 5', '4 vs 6', '5 vs 7']
    three_days = ['3 vs 6', '4 vs 7']
    four_days = ['3 vs 7']

    day_comps = nd_df["Day Comparison"].unique()
    lag_across_days = []
    for p, pat_id_curr in enumerate(pats_ids_in):
        for m, mvmt in enumerate(['left', 'down', 'right', 'up']):
            for d, day in enumerate(day_comps):
                cur_data = nd_df[(nd_df["Frequency"] == freq_band)
                                 & (nd_df["Participant"] == pat_id_curr)
                                 & (nd_df["Movement"] == mvmt)
                                 & (nd_df["Day Comparison"] == day)]["Neural Dissimilarity"].values

                for d in cur_data:
                    if day in zero_days:
                        lag_across_days.append(
                            [freq_band, pat_id_curr, mvmt, 0, d])
                    elif day in one_days:
                        lag_across_days.append(
                            [freq_band, pat_id_curr, mvmt, 1, d])
                    elif day in two_days:
                        lag_across_days.append(
                            [freq_band, pat_id_curr, mvmt, 2, d])
                    elif day in three_days:
                        lag_across_days.append(
                            [freq_band, pat_id_curr, mvmt, 3, d])
                    elif day in four_days:
                        lag_across_days.append(
                            [freq_band, pat_id_curr, mvmt, 4, d])

    lag_across_days_df = pd.DataFrame(lag_across_days, columns=['Frequency',
                                                                'Participant',
                                                                'Movement',
                                                                'Day Lag',
                                                                'Neural Dissimilarity'])
    print(lag_across_days_df.head())
    freq_cross_day_lagged_nd_df = freq_cross_day_lagged_nd_df.append(
        lag_across_days_df)

    return freq_cross_day_pas, freq_cross_day_bs_pas, freq_cross_day_nd_df, freq_cross_day_lagged_nd_df


def cross_pat_comps(exp_params,
                    pats_ids_in,
                    days_tested,
                    freq_band,
                    class_dict,
                    all_sbjs_pca,
                    freq_red_dim,
                    null_data_pa,
                    freq_cross_pat_pas,
                    freq_cross_pat_nd_df):

    # actually compares over all days and all participants
    all_days_all_sbjs_list = []
    for s, sbj in enumerate(pats_ids_in):
        for d, day in enumerate(days_tested):
            cur_comp = sbj + " day " + str(day)
            all_days_all_sbjs_list.append(cur_comp)

    pca_shape = all_sbjs_pca.shape
    # change to (mvmt, sbj * days)
    all_days_pca = np.reshape(
        all_sbjs_pca, (pca_shape[0] * pca_shape[1], pca_shape[2])).T

    cross_pat_pas = []
    for m, mvmt in enumerate(class_dict):
        mvmt_pca = all_days_pca[m, :]
        pas = mu.calc_comp_dim_pas(
            all_days_all_sbjs_list, mvmt_pca, freq_red_dim)
        cross_pat_pas.append(pas)
    cross_pat_pas = np.array(cross_pat_pas)

    # will end up also getting the same participant across days comparisons
    # so figure out inds to keep
    keep_inds = []
    comp_names = mu.get_pa_comparison_names(all_days_all_sbjs_list)
    for k, cur_comp in enumerate(comp_names):
        comp_first_sbj = comp_names[cur_comp].split(" day ")[0]
        comp_second_sbj = comp_names[cur_comp].split(" day ")[
            1].split(" vs ")[1]
        if comp_first_sbj != comp_second_sbj:
            keep_inds.append(k)
    cross_pat_pas = cross_pat_pas[:, keep_inds, :]

    print("Cross-Pat final PAs shape:", cross_pat_pas.shape)
    # cross_pat_pa of shape (movement, sbj*day comps, reduced dim)
    freq_cross_pat_pas[freq_band] = cross_pat_pas

    # now make the dataframe
    # need the electrode overlap as one of the columns
    elec_overlap = mu.calc_elec_overlap(exp_params)
    elec_overlap = np.array(elec_overlap)[
        np.where(np.array(elec_overlap) != 1.0)[0]]
    # need to index elec_overlap
    sbj_comp_names = mu.get_pa_comparison_names(pats_ids_in)
    sbj_comp_names = {val: key for key, val in sbj_comp_names.items()}

    real_summed_pas = mu.calc_norm_sum_pa(freq_red_dim, cross_pat_pas)
    nd_lst = []
    for m, mvmt in enumerate(class_dict):
        p_comp_i = 0
        for k, cur_comp in enumerate(comp_names):
            comp_first_sbj = comp_names[cur_comp].split(" day ")[0]
            comp_second_sbj = comp_names[cur_comp].split(" day ")[
                1].split(" vs ")[1]
            comp_first_sbj_day = comp_names[cur_comp].split(" day ")[
                1].split(" vs ")[0]
            comp_second_sbj_day = comp_names[cur_comp].split(" day ")[-1]
            sbj_comp = comp_first_sbj + " vs " + comp_second_sbj
            if comp_first_sbj != comp_second_sbj:
                # actually add to DF then
                cur_nd = [freq_band,
                          class_dict[mvmt],
                          sbj_comp,
                          comp_first_sbj_day,
                          comp_second_sbj_day,
                          elec_overlap[sbj_comp_names[sbj_comp]],
                          real_summed_pas[m, p_comp_i]]
                nd_lst.append(cur_nd)
                p_comp_i += 1

    # add the null data in, but only on LFO pass though
    if freq_band == "LFO":
        sbj_comp_names = mu.get_pa_comparison_names(pats_ids_in)
        for m, mvmt in enumerate(class_dict.keys()):
            # skip rest
            if class_dict[mvmt] != "rest":
                cur_mvmt_null = null_data_pa[0][m]
                norm_pa = mu.calc_norm_sum_pa(freq_red_dim, cur_mvmt_null)
                norm_pa = np.squeeze(norm_pa)
                for n in range(norm_pa.shape[0]):
                    for c, cur_comp in enumerate(sbj_comp_names):
                        nd_lst.append(["Null",
                                       "Null",
                                       "Null",
                                       "Null",
                                       "Null",
                                       0.0,
                                       norm_pa[n, c]])

    nd_df = pd.DataFrame(nd_lst, columns=["Frequency",
                                          "Movement",
                                          "Participant Comparison",
                                          "First Participant Day",
                                          "Second Participant Day",
                                          "Electrode Overlap",
                                          "Neural Dissimilarity"])
    # remove the rest data
    nd_df = nd_df[nd_df["Movement"] != "rest"]
    nd_df = nd_df.reset_index(drop=True)
    print(nd_df)
    freq_cross_pat_nd_df = freq_cross_pat_nd_df.append(nd_df)

    return freq_cross_pat_pas, freq_cross_pat_nd_df


def main():
    """
    This script calculates the principal angles and also the neural dissimilarity
    for all of the pairwise comparisons we wish to make
    and saves the results in numpy arrays and dataframes
    """

    try:
        json_filename = sys.argv[1]
    except IndexError:
        raise SystemExit(
            f"Usage: {sys.argv[0]} <json file of experiment parameters>")
    with open(json_filename) as f:
        exp_params = json.load(f)

    freq_bands = exp_params["freq_bands"]
    class_dict = exp_params["class_dict"]
    if "0" in class_dict.keys():
        class_dict = {int(cur_key): val for cur_key, val in class_dict.items()}
    else:
        class_dict = {int(cur_key) - 1: val for cur_key,
                      val in class_dict.items()}

    class_color = exp_params["class_color"]
    class_color = {int(cur_key): val for cur_key, val in class_color.items()}
    pats_ids_in = exp_params["pats_ids_in"]

    proj_mat_sp = (
        exp_params["sp"] + exp_params["dataset"] +
        exp_params["experiment_folder"]
    )
    if not os.path.exists(proj_mat_sp):
        os.makedirs(proj_mat_sp)

    null_data_sp = exp_params["null_data_lp"]
    print(null_data_sp)

    days_tested = exp_params["test_day"]

    null_data_pa = np.load(
        exp_params['null_data_lp'] + 'TME_null_pas.npy', allow_pickle=True)

    if len(days_tested) > 1:
        cross_days_null_data_pa = np.load(
            exp_params['cross_days_null_pa_lp'] + 'TME_null_pas.npy', allow_pickle=True)

    cross_pat_null_data_pa = np.load(
        exp_params['cross_pat_null_pa_lp'] + 'TME_null_pas', allow_pickle=True)

    freq_cross_move_pas = {}
    freq_cross_move_nd_df = pd.DataFrame()

    freq_cross_day_pas = {}
    freq_cross_day_bs_pas = {}
    freq_cross_day_nd_df = pd.DataFrame()
    freq_cross_day_lagged_nd_df = pd.DataFrame()

    freq_cross_pat_pas = {}
    freq_cross_pat_nd_df = pd.DataFrame()

    for f, freq_band in enumerate(freq_bands):
        # load in the PCA objects for this frequency band
        # as shape (sbj, day, movement)
        all_sbjs_pca = np.load(proj_mat_sp + freq_band +
                               "_pca_objects.npy", allow_pickle=True)

        # get the right reduced dimensionality
        freq_red_dim = mu.choose_one_freq_dimensionality(class_dict,
                                                         freq_band,
                                                         pats_ids_in,
                                                         np.expand_dims(
                                                             all_sbjs_pca, axis=0)
                                                         )
        print("Reduced dimensionality for {}: {}".format(freq_band, freq_red_dim))

        # get the null data for this frequency band
        freq_null_data_pa = null_data_pa[f]
        freq_null_data_pa = np.squeeze(
            freq_null_data_pa[:, :, :, 0:freq_red_dim])

        # do the cross-movement comparisons
        freq_cross_move_pas, freq_cross_move_nd_df = cross_movement_comps(pats_ids_in,
                                                                          days_tested,
                                                                          class_dict,
                                                                          all_sbjs_pca,
                                                                          freq_red_dim,
                                                                          freq_band,
                                                                          freq_null_data_pa,
                                                                          freq_cross_move_pas,
                                                                          freq_cross_move_nd_df)

        # do the cross-days comparisons
        if len(days_tested) > 1:
            freq_cross_day_pas, freq_cross_day_bs_pas, freq_cross_day_nd_df, freq_cross_day_lagged_nd_df = cross_day_comps(exp_params,
                                                                                                                           all_sbjs_pca,
                                                                                                                           proj_mat_sp,
                                                                                                                           freq_red_dim,
                                                                                                                           freq_band,
                                                                                                                           pats_ids_in,
                                                                                                                           days_tested,
                                                                                                                           class_dict,
                                                                                                                           cross_days_null_data_pa,
                                                                                                                           freq_cross_day_pas,
                                                                                                                           freq_cross_day_bs_pas,
                                                                                                                           freq_cross_day_nd_df,
                                                                                                                           freq_cross_day_lagged_nd_df)

        # do the cross-participant comparisons
        freq_cross_pat_pas, freq_cross_pat_nd_df = cross_pat_comps(exp_params,
                                                                   pats_ids_in,
                                                                   days_tested,
                                                                   freq_band,
                                                                   class_dict,
                                                                   all_sbjs_pca,
                                                                   freq_red_dim,
                                                                   cross_pat_null_data_pa,
                                                                   freq_cross_pat_pas,
                                                                   freq_cross_pat_nd_df)

    # save the principal angles as dict with numpy arrays
    with open(proj_mat_sp + 'freq_cross_move_pas.pkl', 'wb') as f:
        pickle.dump(freq_cross_move_pas, f)

    with open(proj_mat_sp + 'freq_cross_day_pas.pkl', 'wb') as f:
        pickle.dump(freq_cross_day_pas, f)

    with open(proj_mat_sp + 'freq_cross_day_bs_pas.pkl', 'wb') as f:
        pickle.dump(freq_cross_day_bs_pas, f)

    with open(proj_mat_sp + 'freq_cross_pat_pas.pkl', 'wb') as f:
        pickle.dump(freq_cross_pat_pas, f)

    # save the dataframe of neural dissimilarity
    freq_cross_move_nd_df.to_csv(proj_mat_sp + 'freq_cross_move_nd_df.csv')

    freq_cross_day_nd_df.to_csv(proj_mat_sp + 'freq_cross_day_nd_df.csv')
    freq_cross_day_lagged_nd_df.to_csv(
        proj_mat_sp + 'freq_cross_day_lagged_nd_df.csv')

    freq_cross_pat_nd_df.to_csv(proj_mat_sp + 'freq_cross_pat_nd_df.csv')


if __name__ == "__main__":
    sys.exit(main())
