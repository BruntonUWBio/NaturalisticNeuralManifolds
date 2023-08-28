import numpy as np
import pandas as pd

import json
from tqdm import tqdm
import sys
import os
import pdb

from src.data import ECoG_Data
import src.manifold_utils as mu


def main():
    """
    This script runs the PCA analysis for the ECoG data.
    Creates a tensor of PCA objects for each frequency band 
    defined in the input json file.
    The PCA objects are saved in a numpy array of shape
    (pat, day, movement)
    """
    rand_seed = 1762  # 1337
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

    null_data_sp = exp_params["null_data_lp"]
    print(null_data_sp)

    days_tested = exp_params["test_day"]

    # may be worth to swap this loop inside of days
    for f, freq_band in enumerate(freq_bands):
        print("Running for freq_band: {}".format(freq_band))
        # THIS GIVES US OUR PCA OBJECTS,
        # as shape (sbj, day, movement)
        pca_objects = np.empty(
            (len(pats_ids_in), len(days_tested), len(class_dict)), dtype=object
        )
        pca_objects[:] = np.nan

        for d, day in enumerate(days_tested):
            print("Running Manifolds for day", day)
            # load in the data for the current day
            exp_params["test_day"] = str(day)
            manifold_ECoG_data = ECoG_Data(exp_params, proj_mat_sp)
            # run the roi projection
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

            for p, pat_id_curr in enumerate(pats_ids_in):
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

                # This creates the data for each participant and day
                (
                    cur_classes,
                    trial_dim,
                    sr_dim,
                    chan_dim,
                    norm_concat_ECoG_data,
                ) = mu.preprocess_freq_sbj_data(
                    exp_params,
                    sbj_sp,
                    freq_bands,
                    freq_band,
                    class_dict,
                    manifold_ECoG_data,
                )

                class_pca, reduced_class_ECoG_data = mu.calc_class_pca(
                    trial_dim,
                    sr_dim,
                    chan_dim,
                    sbj_sp,
                    freq_band,
                    norm_concat_ECoG_data,
                )
                if class_pca == []:
                    for l in range(len(class_dict)):
                        pca_objects[p, d, l] = []
                    continue
                mu.make_pca_plots(
                    sbj_sp,
                    p,
                    freq_band,
                    class_dict,
                    cur_classes,
                    class_color,
                    trial_dim,
                    sr_dim,
                    manifold_ECoG_data.roi_centroids,
                    class_pca,
                    reduced_class_ECoG_data,
                )

                pca_objects[p, d, :] = np.array(class_pca)

        # save the pca objects for each freq band
        np.save(proj_mat_sp + freq_band + "_pca_objects.npy", pca_objects)

        # # need to have the last dim the days dimension
        # pca_objects = pca_objects.transpose((0, 2, 1))
        # # now that we have all the pca info for this freq band,
        # # we can do the principal angles analysis
        # print(pca_objects.shape)
        # print(pca_objects)

        # red_dim = [15 for i in pats_ids_in]
        # dist, pa = mu.get_pa_per_pat(
        #     pats_ids_in, class_dict, days_tested, red_dim, pca_objects
        # )
        # print("size of prinicpal angles is:")
        # print(np.array(pa).shape)

        # np.save(proj_mat_sp + freq_band + "_pa_per_pat_per_day.npy", pa)


if __name__ == "__main__":
    sys.exit(main())
