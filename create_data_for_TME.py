import numpy as np
import scipy as sci
import mat73
import pandas as pd
import matplotlib.pyplot as plt
from optht import optht

import json, pickle
from tqdm import tqdm
import sys, os, pdb

from src.data import ECoG_Data
from src.fold_data import Fold_Data_Container
import src.manifold_utils as mu
import src.TME.TME_utils as tu


def get_null_data_sbjs_cross_days(
    null_data_sp: str,
    proj_mat_sp: str,
    class_dict: dict,
    freq_bands: dict,
    pats_ids_in: list,
    freq_red_dim: list,
    days_tested: list,
) -> list:
    """
    Get null data for each sbj, freq and movement across days.
    Parameters
    ----------
    null_data_sp : str
        Path to null data
    proj_mat_sp : str
        Path to projection matrix savepoint
    class_dict : dict
        Dictionary of class names
    freq_bands : dict
        Dictionary of frequency bands
    pats_ids_in : list
        List of participant ids
    freq_red_dim : list
        List of PCA reduced dimensions for each frequency
    Returns
    -------
    null_data_sbjs_freqs : list
        List of 1000 samples of null data for each sbj and freq
    """
    num_days = len(days_tested)
    null_data_sbjs_freqs = []
    for f, freq_name in enumerate(freq_bands):
        null_data_sbjs_freqs.append([])
        for s, pat_id_curr in enumerate(pats_ids_in):
            null_data_sbjs_freqs[f].append([])
            for m, mvmt in enumerate(class_dict):
                sbj_sp = proj_mat_sp + pat_id_curr + "/"
                cur_null_sp = null_data_sp + pat_id_curr + "/" + class_dict[mvmt] + "/"
                null_data = mu.load_null_data(cur_null_sp, freq_name)
                # reshape null data to match expected shape
                # (num_samples, sr_dim, num_days, chans)
                if null_data.shape[2] > num_days:
                    null_data = np.swapaxes(null_data, 2, 3)
                print(null_data.shape)
                cur_num_days = null_data.shape[2]

                null_data_sbjs_freqs[f][s].append(
                    mu.get_null_data_pa(
                        null_data,
                        sbj_sp,
                        np.arange(cur_num_days),
                        cur_dim=freq_red_dim[f],
                    )
                )

    return null_data_sbjs_freqs


def main():
    rand_seed = 1762  # 1337
    np.random.seed(rand_seed)

    try:
        json_filename = sys.argv[1]
    except IndexError:
        raise SystemExit(f"Usage: {sys.argv[0]} <json file of experiment parameters>")
    with open(json_filename) as f:
        exp_params = json.load(f)

    freq_bands = exp_params["freq_bands"]
    class_dict = exp_params["class_dict"]
    class_dict = {int(cur_key): val for cur_key, val in class_dict.items()}
    class_color = exp_params["class_color"]
    class_color = {int(cur_key): val for cur_key, val in class_color.items()}
    pats_ids_in = exp_params["pats_ids_in"]
    days_tested = exp_params["test_day"]

    proj_mat_sp = (
        exp_params["sp"] + exp_params["dataset"] + exp_params["experiment_folder"]
    )
    if not os.path.exists(proj_mat_sp):
        os.makedirs(proj_mat_sp)

    TME_data_sp = exp_params["null_data_lp"]
    if not os.path.exists(TME_data_sp):
        os.makedirs(TME_data_sp)

    # check if data already exists
    if os.path.exists(TME_data_sp + "all_data_df.csv"):
        print("Data already exists. Skipping.")
        all_data_df = pd.read_pickle(TME_data_sp + "all_data_df.csv")
        print(all_data_df)
    else:
        all_data_lst = []
        for k, day in enumerate(days_tested):
            print("Creating data for day ", day)
            exp_params["test_day"] = day
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
                print()
                print("Generating Data for ", pat_id_curr)

                sbj_sp = TME_data_sp + pat_id_curr + "/"
                if not os.path.exists(sbj_sp):
                    os.makedirs(sbj_sp)

                manifold_ECoG_data.get_single_sbj(pat_id_curr)
                if not exp_params["rois"]:
                    manifold_ECoG_data.sbj_X_test = (
                        manifold_ECoG_data.remove_zero_channels(
                            manifold_ECoG_data.sbj_X_test
                        )
                    )

                if manifold_ECoG_data.sbj_X_test.shape[0] == 0:
                    print("No data for this day. Skipping.")
                    for f, freq_name in enumerate(freq_bands):
                        for m, mvmt in enumerate(class_dict):
                            all_data_lst.append(
                                [
                                    freq_name,
                                    pat_id_curr,
                                    day,
                                    class_dict[mvmt],
                                    [],
                                ]
                            )
                    continue

                for f, freq_name in enumerate(freq_bands):
                    print("Calculating and saving " + freq_name + " band data")
                    print(manifold_ECoG_data.sbj_X_test.shape)
                    print(np.unique(manifold_ECoG_data.sbj_y_test[:, 0]))
                    cur_data = mu.preprocess_TME_data(
                        exp_params,
                        sbj_sp,
                        freq_bands,
                        freq_name,
                        class_dict,
                        manifold_ECoG_data,
                    )

                    # use this if you want to compare to matlab implementation
                    # if i == 0 and f == 0:
                    #     print("saving first pat and first freq")
                    #     # print(sbj_sp)
                    #     mu.preprocess_TME_data(
                    #         exp_params,
                    #         sbj_sp,
                    #         freq_bands,
                    #         freq_name,
                    #         class_dict,
                    #         manifold_ECoG_data,
                    #         save_flg=True,
                    #     )

                    for m, mvmt in enumerate(class_dict):
                        all_data_lst.append(
                            [
                                freq_name,
                                pat_id_curr,
                                day,
                                class_dict[mvmt],
                                cur_data[..., m],
                            ]
                        )

        # create dataframe of all freq, sbj data now:
        all_data_df = pd.DataFrame(
            all_data_lst,
            columns=[
                "Frequency",
                "Participant",
                "Day",
                "Movement",
                "Average ECoG Data",
            ],
        )
        print(all_data_df)
        # save all data
        all_data_df.to_pickle(TME_data_sp + "all_data_df.csv")
        print("Saved all data")

    # Q1 calculations
    print("Calculating Q1 Null Data")
    for f, freq_name in enumerate(freq_bands):
        for s, pat_id_curr in enumerate(pats_ids_in):
            sbj_sp = TME_data_sp + "Q1/" + pat_id_curr + "/"
            if not os.path.exists(sbj_sp):
                os.makedirs(sbj_sp)
            # index into all_data_df for this pat_id on day == last
            # but need to find the last day first
            pat_id_df = all_data_df.loc[
                (all_data_df["Frequency"] == freq_name)
                & (all_data_df["Participant"] == pat_id_curr)
            ]
            last_day = pat_id_df["Day"].max()
            # this should ensure we get the correct last day
            while (
                pat_id_df.loc[pat_id_df["Day"] == last_day]["Average ECoG Data"].values[
                    0
                ]
                == []
            ):
                last_day = int(last_day) - 1
                last_day = str(last_day)
            print("Last day for ", pat_id_curr, " is ", last_day)
            pat_last_day_df = pat_id_df.loc[pat_id_df["Day"] == last_day]
            print("At this point, should only have 5 rows")
            # not true if they did not have movements for that day
            print(pat_last_day_df)
            # now pull the data out and add to a list
            pat_day_data_lst = []
            for m, mvmt in enumerate(class_dict):
                cur_mvt_data = pat_last_day_df.loc[
                    pat_last_day_df["Movement"] == class_dict[mvmt]
                ]
                cur_ecog_data = cur_mvt_data["Average ECoG Data"].values[0]
                if type(cur_ecog_data) == list:
                    # skip if it is a list, because that means there was no data for that condition
                    continue
                pat_day_data_lst.append(cur_mvt_data["Average ECoG Data"].values[0])
            pat_day_data = np.array(pat_day_data_lst)
            print("data needs to be in shape (n_timepoints, n_channels, n_mvmts)")
            pat_day_data = np.swapaxes(np.swapaxes(pat_day_data, 0, 1), 1, 2)
            print(pat_day_data.shape)

            cur_surr_data = tu.calc_TME(
                np.swapaxes(pat_day_data, 1, 2),
                sbj_sp,
                freq_name,
                surrogate_type="surrogate-TN",
            )
            np.save(
                sbj_sp + "TME_null_" + freq_name + ".npy",
                cur_surr_data,
                allow_pickle=True,
            )

    # still want to run the PA analysis on the null data
    # choosing to go with k=15 for everything moving forward
    print("calculating Q1 null pas")

    null_pas_sbjs_freqs = mu.get_null_data_sbjs_freqs(
        TME_data_sp + "Q1/",
        TME_data_sp + "Q1/",
        class_dict,
        freq_bands,
        pats_ids_in,
        [15 for i in range(len(freq_bands))],
    )
    # can't load and then run because its just too much data to load
    # null_pas_sbjs_freqs = mu.get_null_data_sbjs_freqs_pa(
    #     proj_mat_sp,
    #     freq_bands,
    #     pats_ids_in,
    #     class_dict,
    #     [15 for i in range(len(freq_bands))],
    #     null_sbjs_freqs,
    # )
    null_pas_sbjs_freqs = np.array(null_pas_sbjs_freqs)
    print(null_pas_sbjs_freqs.shape)
    np.save(
        TME_data_sp + "Q1/" + "TME_null_pas.npy",
        null_pas_sbjs_freqs,
        allow_pickle=True,
    )

    # then do Q3 (cross days)
    # note: for some reason day 3 for EC01 is scuffed, but other data is good
    # maybe EC04 too :(
    # change the rest to only be on LFO, too much data otherwise
    # freq_bands = {"LFO": freq_bands["LFO"]}
    # print("Calculating Q3 Null Data")
    # for f, freq_name in enumerate(freq_bands):
    #     for s, pat_id_curr in enumerate(pats_ids_in):
    #         for m, mvmt in enumerate(class_dict):
    #             sbj_sp = (
    #                 TME_data_sp + "Q3/" + pat_id_curr + "/" + class_dict[mvmt] + "/"
    #             )
    #             if not os.path.exists(sbj_sp):
    #                 os.makedirs(sbj_sp)
    #             # index into all_data_df for this pat_id on day == day and cur freq
    #             pat_id_df = all_data_df.loc[
    #                 (all_data_df["Frequency"] == freq_name)
    #                 & (all_data_df["Participant"] == pat_id_curr)
    #                 & (all_data_df["Movement"] == class_dict[mvmt])
    #             ]
    #             print("At this point, should only have one row per day")
    #             print(pat_id_df)
    #             # now pull the data out and add to a list
    #             pat_day_data_lst = []
    #             for d, day in enumerate(days_tested):
    #                 cur_mvt_data = pat_id_df.loc[pat_id_df["Day"] == day]
    #                 cur_ecog_data = cur_mvt_data["Average ECoG Data"].values[0]
    #                 if (type(cur_ecog_data) == list) or (cur_ecog_data.shape == ()):
    #                     print("why you data no worky")
    #                     # skip if it is a list, because that means there was no data for that condition
    #                     continue
    #                 pat_day_data_lst.append(cur_ecog_data)
    #             pat_day_data = np.array(pat_day_data_lst)
    #             print("data needs to be in shape (n_timepoints, n_channels, n_days)")
    #             pat_day_data = np.swapaxes(np.swapaxes(pat_day_data, 0, 1), 1, 2)
    #             print(pat_day_data.shape)

    #             cur_surr_data = tu.calc_TME(
    #                 np.swapaxes(pat_day_data, 1, 2),
    #                 sbj_sp,
    #                 freq_name,
    #                 surrogate_type="surrogate-TN",
    #             )
    #             np.save(
    #                 sbj_sp + "TME_null_" + freq_name + ".npy",
    #                 cur_surr_data,
    #                 allow_pickle=True,
    #             )

    # null_pas_sbjs_cross_days = get_null_data_sbjs_cross_days(
    #     TME_data_sp + "Q3/",
    #     TME_data_sp + "Q3/",
    #     class_dict,
    #     freq_bands,
    #     pats_ids_in,
    #     [15 for i in range(len(freq_bands))],
    #     days_tested,
    # )
    # # should be something like (freqs, sbjs, mvmts, days, comparisons, dims)
    # # null_pas_sbjs_cross_days = np.array(null_pas_sbjs_cross_days)
    # # print(null_pas_sbjs_cross_days.shape)
    # # np.save(
    # #     TME_data_sp + "Q3/" + "TME_null_pas.npy",
    # #     null_pas_sbjs_cross_days,
    # #     allow_pickle=True,
    # # )
    # # Does not work as numpy array because the sizing is inconsistent
    # with open(TME_data_sp + "Q3/" + "TME_null_pas.npy", "wb") as fp:
    #     pickle.dump(null_pas_sbjs_cross_days, fp)

    # convert the last day, with data, for each sbj to a different label
    for s, pat_id_curr in enumerate(pats_ids_in):
        # first, figure out their last day
        pat_id_df = all_data_df.loc[(all_data_df["Participant"] == pat_id_curr)]
        pat_last_day = pat_id_df["Day"].max()
        while (
            pat_id_df.loc[pat_id_df["Day"] == pat_last_day]["Average ECoG Data"].values[
                0
            ]
            == []
        ):
            pat_last_day = int(pat_last_day) - 1
            pat_last_day = str(pat_last_day)

        # then, change in the original df
        pat_day_inds = all_data_df.index[
            (all_data_df["Participant"] == pat_id_curr)
            & (all_data_df["Day"] == str(pat_last_day))
        ].tolist()
        all_data_df.loc[pat_day_inds, "Day"] = "Last Day"

    # then Q4 (cross participants)
    print("Calculating Q4 Null Data")
    for f, freq_name in enumerate(freq_bands):
        for m, mvmt in enumerate(class_dict):
            sbj_sp = TME_data_sp + "Q4/" + class_dict[mvmt] + "/"
            if not os.path.exists(sbj_sp):
                os.makedirs(sbj_sp)
            # index into all_data_df for the movement on last day
            pat_id_df = all_data_df.loc[
                (all_data_df["Frequency"] == freq_name)
                & (all_data_df["Day"] == "Last Day")
                & (all_data_df["Movement"] == class_dict[mvmt])
            ]
            print("At this point, should only have one row per subject")
            print(pat_id_df)
            # now pull the data out and add to a list
            pat_day_data_lst = []
            for s, pat_id_curr in enumerate(pats_ids_in):
                cur_mvt_data = pat_id_df.loc[pat_id_df["Participant"] == pat_id_curr]
                cur_ecog_data = cur_mvt_data["Average ECoG Data"].values[0]
                if type(cur_ecog_data) == list:
                    # skip if it is a list, because that means there was no data for that condition
                    continue
                pat_day_data_lst.append(cur_mvt_data["Average ECoG Data"].values[0])
            pat_day_data = np.array(pat_day_data_lst)
            print("data needs to be in shape (n_timepoints, n_channels, n_sbjs)")
            pat_day_data = np.swapaxes(np.swapaxes(pat_day_data, 0, 1), 1, 2)
            print(pat_day_data.shape)

            cur_surr_data = tu.calc_TME(
                np.swapaxes(pat_day_data, 1, 2),
                sbj_sp,
                freq_name,
                surrogate_type="surrogate-TN",
            )
            np.save(
                sbj_sp + "TME_null_" + freq_name + ".npy",
                cur_surr_data,
                allow_pickle=True,
            )

    null_pas_sbjs_cross_sbjs = mu.get_null_data_sbjs_freqs(
        TME_data_sp + "Q4/",
        TME_data_sp + "Q4/",
        pats_ids_in,
        freq_bands,
        class_dict.values(),
        [15 for i in range(len(freq_bands))],
    )
    # should be something like (freqs, mvmts, sbjs, comparisons, dims)
    with open(TME_data_sp + "Q4/" + "TME_null_pas", "wb") as fp:
        pickle.dump(null_pas_sbjs_cross_sbjs, fp)
    # print(null_pas_sbjs_cross_sbjs.shape)
    # np.save(
    #     TME_data_sp + "Q4/" + "TME_null_pas.npy",
    #     null_pas_sbjs_cross_sbjs,
    #     allow_pickle=True,
    # )


if __name__ == "__main__":
    main()
    print("Done")
    sys.exit(0)
