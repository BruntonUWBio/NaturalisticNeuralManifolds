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

    cross_pat_null_data_pa = np.load(
        exp_params['cross_pat_null_pa_lp'] + 'TME_null_pas', allow_pickle=True)

    freq_cross_move_pas = {}
    freq_cross_move_nd_df = pd.DataFrame()

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
        # freq_cross_move_pas, freq_cross_move_nd_df = cross_movement_comps(pats_ids_in,
        #                                                                   days_tested,
        #                                                                   class_dict,
        #                                                                   all_sbjs_pca,
        #                                                                   freq_red_dim,
        #                                                                   freq_band,
        #                                                                   freq_null_data_pa,
        #                                                                   freq_cross_move_pas,
        #                                                                   freq_cross_move_nd_df)

        # do the cross-days comparisons

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

    with open(proj_mat_sp + 'freq_cross_pat_pas.pkl', 'wb') as f:
        pickle.dump(freq_cross_pat_pas, f)

    # save the dataframe of neural dissimilarity
    freq_cross_move_nd_df.to_csv(proj_mat_sp + 'freq_cross_move_nd_df.csv')

    freq_cross_pat_nd_df.to_csv(proj_mat_sp + 'freq_cross_pat_nd_df.csv')


if __name__ == "__main__":
    sys.exit(main())
