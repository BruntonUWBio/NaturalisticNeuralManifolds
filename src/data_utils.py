# import ecog_utils as ecogu
import numpy as np
import pandas as pd
import xarray as xr
import mne
from scipy.signal import butter, sosfiltfilt, welch, hilbert
import scipy.io
from sklearn.feature_selection import mutual_info_regression
import pickle
import glob
import random
import natsort
import pdb
from tqdm import tqdm
from tensorflow.keras import utils as np_utils
import sys
import os


# Some refactored code based on code from Dimi and Steve
# https://github.com/zoeis52/ECoG-hypernets/commit/c490c7336239a90c230260ab5469611f29d53fac
class ECoG_Data:
    """
    Holds all of the ECoG data for all subjects and generates the data for different folds and models
    ...
    Attributes
    ----------
    pats_ids_in: ids of the participants in the data
    pats_ids_dict: mapping of participant ids to indicies
    lp: where the data is loaded in from
    roi_proj_loadpath: where the roi projection info is loaded in from
    n_chans_all: max number of channels for the data
    feat_dat: what behavioral label is used for the data
    test_day: what day (if any) being used for testing
    ecog_srate: only needed for frequency sliding computation in neural net
    custom_roi_inds: indicies for custom rois from precentral, postcentral, and inf parietal (AAL2)
    proj_mat_out: mapping from ecog electrodes to ROIs for each
        subject based on radial basis function interpolation
    good_ROIs: ROIs that ecog data will be projected to
    nROIs: number of ROIs that ecog is projected to
    X: raw ecog data
    y: behaviorial labels for ecog data
    X_test: raw ecog data from test_day
    y_test: labels for test ecog data
    sbj_order: subject for each data instance in X data
    sbj_order_test: subject for each data instance in X_test data
    nb_classes: the number of classes in the data

    following attributes only available after using single_sbj_randomize
    sbj_X: ecog data for a specific subject
    sbj_y: labels for a specific subject
    sbj_X_test: test ecog data for a specific subject
    sbj_y_test: test labels for a specific subject

    following attributes only available after using combined_sbjs_randomize
    train_sbjs_folds: training subjects for all folds
    val_sbjs_folds: validation subjects for all folds
    test_sbjs_folds: testing subjects for all folds

    Methods
    -------
    single_sbj_randomize(pat_id_cur, model_type)
        Randomizes the data for a specific subject (indicated by pat_id_curr)
        and sizes the data depending on the type of model (model_type)
    single_sbj_get_fold_data(i, n_folds, model_type)
        gets the fold data for the subject from single_sbj_randomize and shapes
        as needed for the model_type
    combined_sbjs_randomize(sp, n_folds, n_test, n_val, n_train, n_evs_per_sbj, half_n_evs_test)
        randomizes all of the data for the number of train, val and test subjects (n_test, n_val, n_train)
        and creates the random partitions for each fold. Also needs to know how many events per subject
        (n_evs_per_sbj) and how the classes will be padded (half_n_evs_test, deafult 'nopad')
    combined_sbjs_get_fold_data(i, model_type)
        gets the data for the current fold (i) and resizes depending on model_type
    calc_mutual_info(freq_bands)
        calculates the mutual information between different frequency bands in the
        ECoG data, and the movement class
    """

    def __init__(self, exp_params, sp):
        """
        Constructs the necessary attributes and loads in the data

        Parameters
        ----------
        Required:
            exp_params: info from JSON file about the experiment parameters
            sp: where the projection matrix information gets saved
        """

        # Passed in vars
        self.pats_ids_in = exp_params["pats_ids_in"]
        self.pats_ids_dict = {sbj: i for i, sbj in enumerate(self.pats_ids_in)}
        self.lp = exp_params["ecog_lp"]
        self.roi_proj_loadpath = exp_params["ecog_roi_proj_lp"]
        self.n_chans_all = exp_params["n_chans_all"]
        self.feat_dat = exp_params["feature_type"]
        self.dataset = exp_params["dataset"]
        self.data_format = exp_params["data_format"]
        self.test_day = exp_params["test_day"]
        if self.test_day == "":
            self.test_day = None
        self.val_percent = 0
        self.test_percent = 0
        self.ecog_srate = exp_params["ecog_srate"]
        self.class_or_reg = exp_params["class_or_reg"]

        # Set up projection mat
        # load custom roi's from precentral, postcentral, and inf parietal (AAL2)
        self.custom_roi_inds = (
            self.get_custom_motor_rois(regions=["naturalistic"])
            if exp_params["custom_rois"]
            else None
        )
        if exp_params["rois"]:
            print("Determining ROIs")
            (
                self.proj_mat_out,
                self.good_ROIs,
                _,
                self.roi_centroids,
            ) = self.proj_mats_good_rois(
                self.pats_ids_in,
                dipole_dens_thresh=exp_params["dipole_dens_thresh"],
                n_chans_all=self.n_chans_all,
                roi_proj_loadpath=self.roi_proj_loadpath,
                rem_bad_chans=exp_params["rem_bad_chans"],
                custom_roi_inds=self.custom_roi_inds,
                chan_cut_thres=self.n_chans_all,
            )
            self.nROIs = len(self.good_ROIs)
            print("ROIs found")
            print("Number of ROIs: ", self.nROIs)

            if exp_params["trim_n_chans"]:
                self.n_chans_all = len(
                    np.nonzero(
                        self.proj_mat_out.reshape(
                            -1, self.proj_mat_out.shape[-1]
                        ).mean(axis=0)
                    )[0]
                )
                self.proj_mat_out = self.proj_mat_out[...,
                                                      : self.n_chans_all]
            np.save(sp + "proj_mat_out", self.proj_mat_out)

        else:
            self.nROIs = self.n_chans_all
            self.proj_mat_out = None

        # Load the data in
        (
            self.X,
            self.y,
            self.X_test,
            self.y_test,
            self.sbj_order,
            self.sbj_order_test,
        ) = self.load_data(
            self.pats_ids_in,
            self.lp,
            exp_params,
            self.n_chans_all,
            self.test_day,
            self.test_percent,
            self.feat_dat,
        )
        self.X[np.isnan(self.X)] = 0  # set all NaN's to 0

        # convert the data labels as needed
        if exp_params["radians"]:
            self.y = np.radians(self.y)
            self.y_test = np.radians(self.y_test)
            print("changed to radians")
        elif exp_params["quadrants"]:
            exp_params["class_or_reg"] = "class"
            self.class_or_reg = "class"

        self.nb_classes = len(np.unique(self.y))

        # TO DO: double check that this will work for all data
        if exp_params["class_or_reg"] == "reg":
            self.nb_classes = (
                len(self.y.shape) if (
                    len(self.y.shape) == 1) else self.y.shape[-1]
            )

    # USER FACING METHODS
    def get_single_sbj(self, pat_id_cur):
        cur_sbj = self.pats_ids_dict[pat_id_cur]
        self.cur_sbj_inds = np.nonzero(self.sbj_order == cur_sbj)[0]
        self.cur_sbj_test_inds = np.nonzero(self.sbj_order_test == cur_sbj)[0]

        self.sbj_X = self.X[self.cur_sbj_inds, ...]
        self.sbj_y = self.y[self.cur_sbj_inds, ...]
        self.sbj_X_test = self.X_test[self.cur_sbj_test_inds, ...]
        self.sbj_y_test = self.y_test[self.cur_sbj_test_inds, ...]

    def get_quadrants(self, y):
        """
        Turns labels for reach angle into discrete classes
        based on which quadrant the angle lies in
        """
        # Note: the data angles are inverted
        for i, a in enumerate(y):
            # left
            if -45 <= a < 45:
                y[i] = 1
            # down
            elif 45 <= a < 135:
                y[i] = 2
            # up
            elif -45 > a > -135:
                y[i] = 4
            # right
            else:
                y[i] = 3

        return y

    def proj_mats_good_rois(
        self,
        patient_ids,
        dipole_dens_thresh=0.1,
        n_chans_all=150,
        roi_proj_loadpath=".../",
        atlas="none",
        rem_bad_chans=True,
        custom_roi_inds=None,
        chan_cut_thres=None,
    ):
        """
        Loads projection matrix for each subject and determines good ROIs to use

        Parameters
        ----------
        dipole_dens_thresh : threshold to use when deciding good ROI's (based on average channel density for each ROI)
        n_chans_all : number of channels to output (should be >= to maximum number of channels across subjects)
        roi_proj_loadpath : where to load projection matrix CSV files
        atlas : ROI projection atlas to use (aal, loni, brodmann, or none)
        rem_bad_chans : whether to remove bad channels from projection step, defined from abnormal SD or IQR across entire day
        """

        all_sbjs = patient_ids

        # Find good ROIs first
        df_all = []
        df_centroids = []
        for s, patient in enumerate(all_sbjs):
            df = pd.read_csv(
                roi_proj_loadpath + atlas + "_" + patient + "_elecs2ROI.csv"
            )
            df_locs = pd.read_csv(
                roi_proj_loadpath + atlas + "_" + patient + "_ROIcentroids_Lside.csv"
            )
            if s == 0:
                dipole_densities = df.iloc[0]
            else:
                dipole_densities += df.iloc[0]
            df_all.append(df)
            df_centroids.append(df_locs)

        dipole_densities = dipole_densities / len(patient_ids)
        if custom_roi_inds is None:
            good_ROIs = np.nonzero(np.asarray(
                dipole_densities) > dipole_dens_thresh)[0]
        else:
            good_ROIs = custom_roi_inds.copy()

        # Now create projection matrix output (patients x roi x chans)
        n_pats = len(patient_ids)
        proj_mat_out = np.zeros([n_pats, len(good_ROIs), n_chans_all])
        chan_ind_vals_all = []
        for s, patient in enumerate(patient_ids):
            df_curr = df_all[s].copy()
            chan_ind_vals = np.nonzero(
                df_curr.transpose().mean().values != 0)[0][1:]
            chan_ind_vals_all.append(chan_ind_vals)
            if rem_bad_chans:
                # Load param file from pre-trained model
                file_pkl = open(roi_proj_loadpath +
                                "bad_ecog_electrodes.pkl", "rb")
                bad_elecs_ecog = pickle.load(file_pkl)
                file_pkl.close()

                sbj_map = {sbj: i for i, sbj in enumerate(all_sbjs)}
                inds2drop = bad_elecs_ecog[sbj_map[patient]]

                # only if prev versions had extra padding for electrodes, which get removed
                inds_too_far = []
                for i, ind in enumerate(inds2drop):
                    if ind >= len(df_curr):
                        inds_too_far.append(i)
                inds2drop = np.delete(inds2drop, inds_too_far)

                if chan_cut_thres is not None:
                    all_inds = np.arange(df_curr.shape[0])
                    inds2drop = np.union1d(
                        inds2drop, all_inds[all_inds > chan_cut_thres]
                    )
                df_curr.iloc[inds2drop] = 0
                # Renormalize across ROIs
                sum_vals = df_curr.sum(axis=0).values
                for i in range(len(sum_vals)):
                    df_curr.iloc[:, i] = df_curr.iloc[:, i] / sum_vals[i]
            n_chans_curr = len(chan_ind_vals)  # df_curr.shape[0]
            tmp_mat = df_curr.values[chan_ind_vals, :]
            proj_mat_out[s, :, :n_chans_curr] = tmp_mat[:, good_ROIs].T
            df_centroids[s] = df_centroids[s].rename(
                columns={"x": "X coor", "y": "Y coor", "z": "Z coor"}
            )
            df_centroids[s] = df_centroids[s].iloc[good_ROIs]

        return proj_mat_out, good_ROIs, chan_ind_vals_all, df_centroids

    def get_custom_motor_rois(
        self, regions=["precentral", "postcentral", "parietal_inf"]
    ):
        """
        Returns ROI indices for those within the precentral, postcentral, and inferior parietal regions (accoring to AAL2)
        """
        naturalistic_inds = np.load(
            self.lp + "elec_and_roi/naturalistic_roi_inds.npy")
        precentral_inds = [
            2263,
            2557,
            2558,
            2571,
            2587,
            2845,
            2846,
            2847,
            2858,
            2859,
            2873,
            2874,
            3113,
            3123,
            3124,
            3136,
            3137,
            3138,
            3151,
            3153,
            3154,
            3359,
            3360,
            3369,
            3370,
            3371,
            3383,
            3384,
            3559,
            3565,
            3566,
            3567,
            3568,
            3576,
            3577,
            3578,
            3579,
            3589,
            3590,
            3722,
            3723,
            3724,
            3729,
            3730,
            3731,
            3739,
            3740,
            3752,
            3837,
        ]
        postcentral_inds = [
            2236,
            2237,
            2238,
            2246,
            2247,
            2248,
            2545,
            2546,
            2547,
            2555,
            2556,
            2569,
            2570,
            2835,
            2836,
            2843,
            2844,
            2856,
            2857,
            2871,
            3110,
            3111,
            3112,
            3121,
            3122,
            3133,
            3134,
            3135,
            3149,
            3350,
            3351,
            3355,
            3356,
            3357,
            3358,
            3367,
            3368,
            3381,
            3382,
            3395,
            3555,
            3556,
            3557,
            3563,
            3564,
            3574,
            3575,
            3587,
            3588,
            3720,
            3721,
            3727,
            3728,
            3737,
            3738,
            3832,
            3834,
            3835,
            3836,
            3842,
            3843,
        ]
        parietal_inf_inds = [
            3106,
            3107,
            3108,
            3116,
            3117,
            3118,
            3119,
            3120,
            3131,
            3132,
            3143,
            3144,
            3145,
            3146,
            3147,
            3148,
            3161,
            3347,
            3348,
            3349,
            3352,
            3353,
            3354,
            3364,
            3365,
            3366,
            3376,
            3378,
            3379,
            3380,
            3553,
            3554,
            3561,
            3562,
        ]

        # Account for Matlab indexing starting at 1
        precentral_inds = [val - 1 for val in precentral_inds]
        postcentral_inds = [val - 1 for val in postcentral_inds]
        parietal_inf_inds = [val - 1 for val in parietal_inf_inds]
        # naturalistic_inds = [val - 1 for val in naturalistic_inds]

        #     custom_roi_inds = np.union1d(np.union1d(precentral_inds,postcentral_inds),parietal_inf_inds) #select for sensorimotor ROIs
        custom_roi_inds = []
        for val in regions:
            eval("custom_roi_inds.extend(" + val + "_inds)")
        return custom_roi_inds

    def remove_zero_channels(self, x):
        dp = x[0, :, :]
        # BUG: axis needs to be whatever the timepoint axis is? Probably depends on the data
        filt_dp_shape = dp[~np.all(dp == 0, axis=-1)].shape
        num_good_chans = filt_dp_shape[0]
        print("number good channels: ", num_good_chans)
        X_tmp = np.zeros((x.shape[0], filt_dp_shape[0], x.shape[2]))
        for i, dp in enumerate(x):
            # filt_dp = dp[~np.all(dp == 0, axis=-1)]
            filt_dp = dp[0:num_good_chans, :]
            X_tmp[i] = filt_dp
        x = X_tmp.copy()

        return x

    def get_metadata_labels(self, ep_data, ep_metadata_in):
        htnet_gen_labels = ep_data.to_array()[:, :, -1, 0]
        htnet_gen_labels = np.squeeze(np.array(htnet_gen_labels))
        # event_dict = {'rest':1,'move':2}
        move_indicies = np.where(htnet_gen_labels == 2)[0]
        rest_indicies = np.where(htnet_gen_labels == 1)[0]
        assert (
            move_indicies.shape[0] == rest_indicies.shape[0]
        ), "does not contain the same number of move and rest labels"

        # does not actually pull in time labels atm
        move_times = (
            ep_metadata_in[
                dict(
                    features=np.nonzero(
                        np.asarray(ep_metadata_in.features) == "reach_a"
                    )[0]
                )
            ]
            .to_array()
            .values.squeeze()
        )

        m_times = []
        meta_data_i = 0
        for l in range(len(htnet_gen_labels)):
            if l in move_indicies:
                move_in_hrs = move_times[meta_data_i]
                # move_in_hrs = move_times[meta_data_i] / (1000 * 60 * 60)
                m_times.append(move_in_hrs)
                meta_data_i += 1
            else:
                m_times.append(-1)

        return m_times

    def align_m_v_r_angle(self, ep_data, ep_metadata_in):
        htnet_gen_labels = ep_data.to_array()[:, :, -1, 0]
        htnet_gen_labels = np.squeeze(np.array(htnet_gen_labels))
        # event_dict = {'rest':1,'move':2}
        move_indicies = np.where(htnet_gen_labels == 2)[0]
        rest_indicies = np.where(htnet_gen_labels == 1)[0]
        assert (
            move_indicies.shape[0] == rest_indicies.shape[0]
        ), "does not contain the same number of move and rest labels"

        move_reach_a = (
            ep_metadata_in[
                dict(
                    features=np.nonzero(
                        np.asarray(ep_metadata_in.features) == "reach_a"
                    )[0]
                )
            ]
            .to_array()
            .values.squeeze()
        )
        move_reach_a = self.get_quadrants(move_reach_a)
        # print(move_reach_a.shape)

        m_v_r_quads = []
        meta_data_i = 0
        for l in range(len(htnet_gen_labels)):
            if l in move_indicies:
                m_v_r_quads.append(move_reach_a[meta_data_i])
                meta_data_i += 1
            else:
                m_v_r_quads.append(0)

        return m_v_r_quads

    def load_data(
        self,
        pats_ids_in,
        lp,
        exp_params,
        n_chans_all=64,
        test_day=None,
        test_percent=0,
        feat_dat="reach_a",
    ):
        """
        Load ECoG data from all subjects and combine (uses xarray variables)

        If len(pats_ids_in)>1, the number of electrodes will be padded or cut to match n_chans_all
        If test_day is not None, a variable with test data will be generated for the day specified
            If test_day = 'last', the last day will be set as the test day.
        """

        # correct pat_ids_in type if needed
        if not isinstance(pats_ids_in, list):
            pats_ids_in = [pats_ids_in]
        if len(test_day) == 1:
            test_day = test_day[0]
        sbj_order, sbj_order_test = [], []
        X_test_subj, y_test_subj = [], []  # placeholder vals

        # Gather each subjects data, and concatenate all days
        for j in tqdm(range(len(pats_ids_in))):
            pat_curr = pats_ids_in[j]

            # load in ecog data and metatdata for current sbj
            if "steve_xr" in self.data_format:
                print("loading steve data")
                (
                    ep_data_in,
                    ep_metadata_in,
                    time_inds,
                    n_ecog_chans,
                ) = self.load_nat_ecog(
                    lp, pat_curr, exp_params["tlim"], feat_dat
                )
            elif "exp_xr" in self.data_format:
                print("loading in fingerflex data")
                (
                    ep_data_in,
                    ep_metadata_in,
                    time_inds,
                    n_ecog_chans,
                ) = self.load_fingerflex_data(lp, pat_curr, exp_params["tlim"])
            elif "pose" in self.data_format:
                print("loading pose data")
                (
                    ep_data_in,
                    ep_metadata_in,
                    time_inds,
                    n_ecog_chans,
                ) = self.load_pose(
                    lp, pat_curr, exp_params["tlim"], feat_dat
                )
            else:
                raise ValueError("data format not supported")

            # determine right number of channels
            if n_chans_all < n_ecog_chans:
                n_chans_curr = n_chans_all
            else:
                n_chans_curr = n_ecog_chans

            # Next few lines: Split into train and test data based on the day
            days_train, days_test_inds, days_all_in = self.dl_calc_train_test_days(
                test_day, ep_data_in
            )

            # Compute indices of days_train in xarray dataset
            dat_train, labels_train = self.dl_get_train_data(
                ep_data_in,
                ep_metadata_in,
                days_train,
                days_all_in,
                n_chans_curr,
                time_inds,
                feat_dat,
            )
            # Get the test data out, and modify train data if getting a percentage
            (
                dat_test,
                labels_test,
                dat_train,
                labels_train,
                sbj_order_test,
                sbj_order,
            ) = self.dl_get_test_data(
                test_day,
                days_test_inds,
                ep_data_in,
                ep_metadata_in,
                dat_train,
                labels_train,
                n_chans_curr,
                time_inds,
                feat_dat,
                sbj_order_test,
                sbj_order,
                j,
            )

            # check if balancing data needed
            # dat_train, labels_train, dat_test, labels_test = self.dl_check_balance(
            #     dat_train, labels_train, dat_test, labels_test, exp_params
            # )

            # Pad data in electrode dimension if necessary
            dat_train = self.dl_pad_electrodes(
                pats_ids_in, n_chans_all, n_ecog_chans, dat_train
            )
            dat_test = self.dl_pad_electrodes(
                pats_ids_in, n_chans_all, n_ecog_chans, dat_test
            )

            # Concatenate across subjects
            if j == 0:
                X_subj = dat_train.copy()
                y_subj = labels_train.copy()
                y_subj = np.array(y_subj)
                X_test_subj = dat_test.copy()
                y_test_subj = labels_test.copy()
                y_test_subj = np.array(y_test_subj)
                if len(labels_test) == 0:
                    y_test_subj = np.array([[], []]).T
            else:
                X_subj = np.concatenate((X_subj, dat_train.copy()), axis=0)
                y_subj = np.concatenate((y_subj, labels_train.copy()), axis=0)
                X_test_subj = np.concatenate(
                    (X_test_subj, dat_test.copy()), axis=0)
                if len(labels_test) == 0:
                    continue
                y_test_subj = np.concatenate(
                    (y_test_subj, labels_test.copy()), axis=0)

        sbj_order = np.asarray(sbj_order)
        sbj_order_test = np.asarray(sbj_order_test)
        print("Data loaded!")

        return X_subj, y_subj, X_test_subj, y_test_subj, sbj_order, sbj_order_test

    def load_nat_ecog(self, lp, pat_curr, tlim, feat_dat):
        # ZSH: what is the datatype of ep_data_in???
        # ZSH: xarray object
        # if self.dataset == "pose":
        #     ep_data_in = xr.open_dataset(lp + pat_curr + "_pose_data.nc")
        # else:
        ep_data_in = xr.open_dataset(lp + pat_curr + "_ecog_data.nc")

        if self.class_or_reg == "reg":
            # does not contain region metadata
            ep_metadata_in = xr.open_dataset(lp + pat_curr + "_metadata.nc")
        elif feat_dat == "mvr_quads":
            meta_lp = lp
            ep_metadata_in = xr.open_dataset(
                meta_lp + pat_curr + "_metadata.nc")

            mvmt_time_labels = self.get_metadata_labels(
                ep_data_in, ep_metadata_in)
            mvmt_dir_labels = self.align_m_v_r_angle(
                ep_data_in, ep_metadata_in)
            ep_metadata_in = np.concatenate(
                (
                    np.expand_dims(np.array(mvmt_dir_labels), axis=1),
                    np.expand_dims(np.array(mvmt_time_labels), axis=1),
                ),
                axis=1,
            )
        else:
            ep_metadata_in = None

        ep_times = np.asarray(ep_data_in.time)
        time_inds = np.nonzero(
            np.logical_and(ep_times >= tlim[0], ep_times <= tlim[1])
        )[0]
        n_ecog_chans = len(ep_data_in.channels) - 1

        return ep_data_in, ep_metadata_in, time_inds, n_ecog_chans

    def load_pose(self, lp, pat_curr, tlim, feat_dat):
        pose_data_in = xr.open_dataset(lp + pat_curr + "_pose_data.nc")
        print(pose_data_in)
        ep_times = np.asarray(pose_data_in.time)
        time_inds = np.nonzero(
            np.logical_and(ep_times >= tlim[0], ep_times <= tlim[1])
        )[0]
        n_pose_chans = len(pose_data_in.channels)

        if feat_dat == "mvr_quads":
            ep_metadata_in = pose_data_in.pose_features

            # problem using this with my pose data
            # don't have the move vs rest labels in there
            # but I think could do with the pose_features
            mvmt_time_labels = ep_metadata_in.sel(features="vid_name")
            mvmt_dir_labels = ep_metadata_in.sel(features="reach_a")
            mvmt_dir_labels = [
                0 if float(e) == -1 else self.get_quadrants([float(e)])[0]
                for e in mvmt_dir_labels
            ]
            ep_metadata_in = np.concatenate(
                (
                    np.expand_dims(np.array(mvmt_dir_labels), axis=1),
                    np.expand_dims(np.array(mvmt_time_labels), axis=1),
                ),
                axis=1,
            )
        elif feat_dat == "exp_pose":
            ep_metadata_in = np.array(
                pose_data_in.__xarray_dataarray_variable__)[:, -1, 0]
            # pose_data_in = pose_data_in.__xarray_dataarray_variable__
            n_pose_chans = len(pose_data_in.channels) - 1
            return pose_data_in, ep_metadata_in, time_inds, n_pose_chans
        else:
            ep_metadata_in = None

        # may need to just get the pose_data for pose_data_in
        pose_data_in = (pose_data_in.pose_data).to_dataset()

        return pose_data_in, ep_metadata_in, time_inds, n_pose_chans

    def load_fingerflex_data(self, lp, pat_curr, tlim):
        ep_data_in = xr.open_dataset(lp + pat_curr + "_ec_data.nc")
        ep_times = np.asarray(ep_data_in.time)
        time_inds = np.nonzero(
            np.logical_and(ep_times >= tlim[0], ep_times <= tlim[1])
        )[0]
        n_ecog_chans = len(ep_data_in.channels) - 1

        return ep_data_in, None, time_inds, n_ecog_chans

    def dl_calc_train_test_days(self, test_day, ep_data_in):
        days_all_in = np.asarray(ep_data_in.events)

        # ZSH: figure out which day is test, if it is a specific day
        if test_day == "last":
            test_day_curr = np.unique(ep_data_in.events)[-1]  # select last day
            days_train = np.unique(days_all_in)[:-1]
            day_test_curr = np.unique(days_all_in)[-1]
            days_test_inds = np.nonzero(days_all_in == day_test_curr)[0]
        elif test_day == 'all':
            days_train = np.unique(days_all_in)
            test_day_curr = None
            days_test_inds = None
        elif test_day is None:
            days_train = np.unique(days_all_in)
            test_day_curr = None  # or we will split data by percentages
            days_test_inds = None
        elif test_day.isnumeric():
            test_day_curr = int(test_day)  # otherwise select numeric day
            days_train = np.unique(
                days_all_in[days_all_in != test_day_curr]
            )  # np.delete(np.unique(days_all_in), test_day_curr)
            days_test_inds = np.nonzero(days_all_in == test_day_curr)[0]

        # print(days_test_inds)
        # assert len(days_test_inds) != 0, "test day not found"

        return days_train, days_test_inds, days_all_in

    def dl_get_train_data(
        self,
        ep_data_in,
        ep_metadata_in,
        days_train,
        days_all_in,
        n_chans_curr,
        time_inds,
        feat_dat,
    ):
        # Compute indices of days_train in xarray dataset
        days_train_inds = []
        for day_tmp in list(days_train):
            days_train_inds.extend(np.nonzero(days_all_in == day_tmp)[0])

        # Extract data and labels
        dat_train = (
            ep_data_in[
                dict(
                    events=days_train_inds,
                    channels=slice(0, n_chans_curr),
                    time=time_inds,
                )
            ]
            .to_array()
            .values.squeeze()
        )

        # BUG: there is a bit of an issue here if we want to do percentage of
        # data for mvr_quads
        if self.feat_dat == "mvr_quads":
            labels_train = [
                [ep_metadata_in[i, 0], ep_metadata_in[i, 1]] for i in days_train_inds
            ]
        elif self.class_or_reg == "reg":
            labels_train = (
                ep_metadata_in[
                    dict(
                        events=days_train_inds,
                        features=np.nonzero(
                            np.asarray(ep_metadata_in.features) == feat_dat
                        )[0],
                    )
                ]
                .to_array()
                .values.squeeze()
            )
        else:
            labels_train = (
                ep_data_in[
                    dict(
                        events=days_train_inds, channels=ep_data_in.channels[-1], time=0
                    )
                ]
                .to_array()
                .values.squeeze()
            )
            labels_train = np.expand_dims(labels_train, axis=1)

        return dat_train, labels_train

    def dl_get_test_data(
        self,
        test_day,
        days_test_inds,
        ep_data_in,
        ep_metadata_in,
        dat_train,
        labels_train,
        n_chans_curr,
        time_inds,
        feat_dat,
        sbj_order_test,
        sbj_order,
        j,
    ):
        # pull out test day asked for
        if (test_day == "last") or (test_day is not None):
            print("Pulling out " + test_day + " for test day")
            if test_day == 'all':
                days_test_inds = np.arange(len(ep_data_in.events))

            dat_test = (
                ep_data_in[
                    dict(
                        events=days_test_inds,
                        channels=slice(0, n_chans_curr),
                        time=time_inds,
                    )
                ]
                .to_array()
                .values.squeeze()
            )

            if self.class_or_reg == "reg":
                labels_test = (
                    ep_metadata_in[
                        dict(
                            events=days_test_inds,
                            features=np.nonzero(
                                np.asarray(ep_metadata_in.features) == feat_dat
                            )[0],
                        )
                    ]
                    .to_array()
                    .values.squeeze()
                )
            elif self.feat_dat == "mvr_quads":
                labels_test = [
                    [ep_metadata_in[i, 0], ep_metadata_in[i, 1]] for i in days_test_inds
                ]
            else:
                labels_test = (
                    ep_data_in[
                        dict(
                            events=days_test_inds,
                            channels=ep_data_in.channels[-1],
                            time=0,
                        )
                    ]
                    .to_array()
                    .values.squeeze()
                )
                labels_test = np.expand_dims(labels_test, axis=1)

        # otherwise, chosing to split the data by percent
        else:
            print("Pulling out " + str(self.test_percent) +
                  " percent for test data")
            # first get the train and test indices
            num_events = len(labels_train)
            num_test_events = int(num_events * (self.test_percent / 100))
            rand_train_inds = np.arange(num_events)
            rand_test_inds = np.random.choice(
                num_events, size=num_test_events, replace=False
            )
            rand_train_inds = np.setdiff1d(rand_train_inds, rand_test_inds)

            # then index into the train data and test
            dat_test = dat_train[rand_test_inds, ...]
            labels_test = labels_train[rand_test_inds, ...]

            dat_train = dat_train[rand_train_inds, ...]
            labels_train = labels_train[rand_train_inds, ...]

        sbj_order_test += [j] * dat_test.shape[0]
        sbj_order += [j] * dat_train.shape[0]

        return dat_test, labels_test, dat_train, labels_train, sbj_order_test, sbj_order

    def dl_pad_electrodes(self, pats_ids_in, n_chans_all, n_ecog_chans, data):
        if (len(pats_ids_in) > 1) and (n_chans_all > n_ecog_chans):
            dat_sh = list(data.shape)
            dat_sh[1] = n_chans_all
            # Create dataset padded with zeros if less than n_chans_all, or cut down to n_chans_all
            X_pad = np.zeros(dat_sh)
            X_pad[:, :n_ecog_chans, ...] = data
            data = X_pad.copy()

        return data
