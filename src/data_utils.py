# import ecog_utils as ecogu
# from pyEMG.features_online import get_ar_feat
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
# from src.fold_data import Fold_Data_Container

sys.path.append("/home/zsteineh/ECoG_utils/pyEMG/")
# Code from agamemnonc pyEMG repo

sys.path.append("/home/zsteineh/ECoG_utils/satpreet-analysis/")


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

        # need to generate the sin data if doing synthetic sin
        if self.dataset == "synthetic_sin":
            print("Generating sine data")
            sbj_data_samples = exp_params["n_evs_per_sbj"]
            test_sbj_data_samples = int(sbj_data_samples / 10)
            freq = 5
            time_stamps = 501  # match time sampling of ecog data

            self.X, self.y, self.sbj_order = self.generate_sine_data(
                time_stamps, freq, sbj_data_samples
            )
            self.X_test, self.y_test, self.sbj_order_test = self.generate_sine_data(
                time_stamps, freq, test_sbj_data_samples
            )
            self.nb_classes = len(np.unique(self.y))
            self.nROIs = 100
        # Load in EMG data from Atzori et al., Scientific Data, 2014
        elif self.dataset == "emg":
            print("Gathering EMG data")
            (
                self.X,
                self.y,
                self.X_test,
                self.y_test,
                self.acc_avged,
                self.glove_avged,
                self.acc_test,
                self.glove_test,
                self.sbj_order,
                self.sbj_order_test,
            ) = self.load_emg_data(self.pats_ids_in, self.lp)

            self.nb_classes = len(np.unique(self.y))
            self.nROIs = 100
        # otherwise looking at the ecog data
        else:
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
            # print("SHAPE OF DATA!!")
            # print(self.X.shape)

            # convert the data labels as needed
            if exp_params["radians"]:
                self.y = np.radians(self.y)
                self.y_test = np.radians(self.y_test)
                print("changed to radians")
            elif exp_params["quadrants"]:
                exp_params["class_or_reg"] = "class"
                self.class_or_reg = "class"
            # Trying out synthetic data by summing the data, might be a stupid method for now
            elif self.dataset == "synthetic_sum":
                self.y = self.X.sum(axis=2).sum(axis=1)
                self.y_test = self.X_test.sum(axis=2).sum(axis=1)

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

        # TO DO: this may be an issue to comment out
        # self.remove_nan_data()

    def single_sbj_randomize(self, pat_id_cur, model_type):
        """
        Randomizes the data for a specific subject and sizes the data
        depending on the type of model

        Parameters
        ----------
        pat_id_cur: id of the subject who's data you want to pull
        model_type: string name of the model in use
        """
        # Get the particular subjects data
        self.get_single_sbj(pat_id_cur)

        # Randomize event order (random seed facilitates consistency)
        # BUG: Random seed does not actually make data split consistent
        # only an issue if you plan to use the labels by regnerating them
        # rather than saving the info for each run in a np file
        order_inds = np.arange(len(self.sbj_y))
        np.random.shuffle(order_inds)
        # print('random shuffle of train/val inds')
        # print(order_inds)
        # input()
        self.sbj_X = self.sbj_X[order_inds, ...]
        self.sbj_y = self.sbj_y[order_inds]

        order_inds_test = np.arange(len(self.sbj_y_test))
        np.random.shuffle(order_inds_test)
        # print('random shuffle of test inds')
        # print(order_inds)
        # input()
        self.sbj_X_test = self.sbj_X_test[order_inds_test, ...]
        self.sbj_y_test = self.sbj_y_test[order_inds_test]

        # do this just for simple models of ecog
        # TO DO: can maybe use spec_meas to switch on this part
        # self.sbj_X, self.sbj_X_test = self.emg_get_lr_feats(self.sbj_X, [], self.sbj_X_test, [])

        # if linear regression and emg, do something different to the data
        if self.dataset == "emg" and self.feat_dat == "lr_fts":
            print("Calculating EMG Linear Regression Features")
            # get emg acceleration data
            sbj_acc = self.acc_avged[self.cur_sbj_inds, ...]
            sbj_acc = sbj_acc[order_inds, ...]

            sbj_acc_test = self.acc_test[self.cur_sbj_test_inds, ...]
            sbj_acc_test = sbj_acc_test[order_inds_test, ...]
            # calculate the linear regression fts, including band power
            self.sbj_X, self.sbj_X_test = self.emg_get_lr_feats(
                self.sbj_X, sbj_acc, self.sbj_X_test, sbj_acc_test
            )

            # Also need to get the glove data/y into the right format
            sbj_glove_data = self.glove_avged[self.cur_sbj_inds, ...]
            sbj_glove_data = sbj_glove_data[order_inds, ...]
            self.sbj_y = sbj_glove_data
            # get a specific sensor:
            # self.sbj_y = sbj_glove_data[:,-1]

            sbj_glove_data_test = self.glove_test[self.cur_sbj_test_inds, ...]
            sbj_glove_data_test = sbj_glove_data_test[order_inds_test, ...]
            self.sbj_y_test = sbj_glove_data_test
            # get a specific sensor:
            # self.sbj_y_test = sbj_glove_data_test[:,-1]

            self.nb_classes = self.sbj_y.shape[-1]
            # self.nb_classes = 1

            if (
                model_type != "rf"
                and model_type != "riemann"
                and model_type != "lr"
                and ("ffnet" not in model_type)
            ):
                self.sbj_X = np.expand_dims(self.sbj_X, -1)
                self.sbj_X_test = np.expand_dims(self.sbj_X_test, -1)

        if model_type == "eegnet" or model_type == "eegnet_hilb":
            # print('expanding dims')
            self.sbj_X = np.expand_dims(self.sbj_X, 1)
            self.sbj_X_test = np.expand_dims(self.sbj_X_test, 1)
        # elif ('ffnet' in model_type) and len(self.sbj_X.shape) == 3:
        #     self.sbj_X = np.reshape(self.sbj_X, (self.sbj_X.shape[0], (self.sbj_X.shape[1] * self.sbj_X.shape[2]) ))
        #     self.sbj_X_test = np.reshape(self.sbj_X_test, (self.sbj_X_test.shape[0], (self.sbj_X_test.shape[1] * self.sbj_X_test.shape[2]) ))

        # if self.class_or_reg == "class":
        #     self.sbj_y = np_utils.to_categorical(self.sbj_y - 1)
        #     self.sbj_y_test = np_utils.to_categorical(self.sbj_y_test - 1)

    def single_sbj_get_fold_data(self, i, n_folds, model_type):
        """
        Gets the fold data for the subject from single_sbj_randomize and shapes
        as needed for the model_type

        Parameters
        ----------
        i: the current fold
        n_folds: number of folds for training
        model_type: string name of the model in use
        """
        # Create split len for train/val
        sbj_split_len = self.sbj_X.shape[0] // n_folds

        # Get the train and val indices for this fold
        if self.test_day is None:
            # in the no test day case, split the data into percentages
            # across all of the days
            print("NEW TYPE OF DATA SPLIT!")
            total_num_events = self.sbj_X.shape[0] + self.sbj_X_test.shape[0]
            num_val_events = int(total_num_events * (self.val_percent / 100))
            val_inds = np.random.choice(
                self.sbj_X.shape[0], size=num_val_events, replace=False
            )
        else:
            # otherwise, split the data by holding out one day for testing
            val_inds = np.arange(0, sbj_split_len) + (i * sbj_split_len)
        print("random val inds selected")
        print(val_inds)
        train_inds = np.setdiff1d(
            np.arange(self.sbj_X.shape[0]), val_inds
        )  # take all events not in val set

        # Split data and labels into train/val sets
        X_train = self.sbj_X[train_inds, ...]
        Y_train = self.sbj_y[train_inds]
        X_validate = self.sbj_X[val_inds, ...]
        Y_validate = self.sbj_y[val_inds]
        X_test = self.sbj_X_test.copy()
        Y_test = self.sbj_y_test.copy()

        # can use this to see if the class distribution is decent
        # print("label disrib after val train split")
        # print(self.get_class_distrib(Y_train))
        # print(self.get_class_distrib(Y_validate))
        # print(self.get_class_distrib(Y_test))

        proj_mat_out_nn = None
        sbj_order_train = None
        sbj_order_validate = None
        sbj_order_test = None

        if model_type == "rf" or ("ffnet" in model_type):
            # For random forest, combine electrodes and time dimensions
            X_train = X_train.reshape(X_train.shape[0], -1)
            X_validate = X_validate.reshape(X_validate.shape[0], -1)
            X_test = X_test.reshape(X_test.shape[0], -1)

        elif model_type == "riemann":
            X_train, X_validate, X_test = self.riemann_remove_bad_events(
                X_train, X_validate, X_test
            )

        # and after all that processing, put in the data container :)
        cur_fold_data = Fold_Data_Container(
            Y_train,
            X_train,
            Y_validate,
            X_validate,
            Y_test,
            X_test,
            model_type,
            sbj_order_train,
            sbj_order_validate,
            sbj_order_test,
            proj_mat_out_nn,
            nb_classes=self.nb_classes,
            nROIs=self.nROIs,
            ecog_srate=self.ecog_srate,
            class_or_reg=self.class_or_reg,
        )
        return cur_fold_data

    def combined_sbjs_randomize(
        self,
        sp,
        n_folds,
        n_test=1,
        n_val=4,
        n_train=7,
        n_evs_per_sbj=500,
        half_n_evs_test="nopad",
    ):
        """
        Randomizes all of the data for the number of train, val and test subjects
        and creates the random partitions for each fold.

        Parameters
        ----------
        Required:
            sp: where the data spilts will be saved
            n_folds: number of folds for training

        Not Required:
            n_test: number of subjects for testing (default = 1)
            n_val: number of subjects for validation (default = 4)
            n_train: number of subjects for training (default = 7)
            n_evs_per_sbj: number of events for each subject (default = 500)
            half_n_evs_test: how the classes will be padded (default = 'nopad')
        """
        self.train_inds_folds, self.val_inds_folds, self.test_inds_folds = [], [], []

        # Choose which subjects for training/validation/testing for every fold (splits are based on random seed)
        (
            self.train_sbjs_folds,
            self.val_sbjs_folds,
            self.test_sbjs_folds,
        ) = self.folds_choose_subjects(
            n_folds, self.pats_ids_in, n_test=n_test, n_val=n_val, n_train=n_train
        )

        # Iterate across all model types specified
        labels_unique = np.array(
            [1]
        )  # np.unique(self.y) # use y if doing classification
        if isinstance(n_evs_per_sbj, str):
            half_n_evs = n_evs_per_sbj
        else:
            half_n_evs = n_evs_per_sbj // len(labels_unique)

        # generate the indices for all folds
        # Only need to determine train/val/test inds on first run
        for i in tqdm(range(n_folds)):
            test_sbj = self.test_sbjs_folds[i]
            val_sbj = self.val_sbjs_folds[i]
            train_sbj = self.train_sbjs_folds[i]
            # Find train/val/test indices (test inds differ depending on if test_day is specified or not)
            # Note that subject_data_inds will balance number of trials across classes
            train_inds, val_inds, test_inds = [], [], []
            if self.test_day is None:
                test_inds = self.subject_data_inds(
                    np.full(1, test_sbj),
                    self.sbj_order,
                    labels_unique,
                    i,
                    "test_inds",
                    half_n_evs_test,
                    np.ones_like(self.y),
                    sp,
                    n_folds,
                    test_inds,
                )
            else:
                test_inds = self.subject_data_inds(
                    np.full(1, test_sbj),
                    self.sbj_order_test,
                    labels_unique,
                    i,
                    "test_inds",
                    half_n_evs_test,
                    np.ones_like(self.y_test),
                    sp,
                    n_folds,
                    test_inds,
                )
            val_inds = self.subject_data_inds(
                val_sbj,
                self.sbj_order,
                labels_unique,
                i,
                "val_inds",
                half_n_evs,
                np.ones_like(self.y),
                sp,
                n_folds,
                val_inds,
            )
            train_inds = self.subject_data_inds(
                train_sbj,
                self.sbj_order,
                labels_unique,
                i,
                "train_inds",
                half_n_evs,
                np.ones_like(self.y),
                sp,
                n_folds,
                train_inds,
            )

            self.train_inds_folds.append(train_inds)
            self.val_inds_folds.append(val_inds)
            self.test_inds_folds.append(test_inds)

    def combined_sbjs_get_fold_data(self, i, model_type):
        """
        Gets the data from the combined subjects for the current fold (i) and
        resizes depending on model_type

        Parameters
        ----------
        i: the current fold
        model_type: string name of the model in use
        """
        train_inds = self.train_inds_folds[i]
        val_inds = self.val_inds_folds[i]
        test_inds = self.test_inds_folds[i]

        # Now that we have the train/val/test event indices, generate the data for the models
        proj_mat_out_nn = None
        X_train = self.X[train_inds, ...]
        Y_train = self.y[train_inds]
        sbj_order_train = self.sbj_order[train_inds]
        X_validate = self.X[val_inds, ...]
        Y_validate = self.y[val_inds]
        sbj_order_validate = self.sbj_order[val_inds]
        if self.test_day is None:
            X_test = self.X[test_inds, ...]
            Y_test = self.y[test_inds]
            sbj_order_test = self.sbj_order[test_inds]
        else:
            X_test = self.X_test[test_inds, ...]
            Y_test = self.y_test[test_inds]
            sbj_order_test = self.sbj_order_test[test_inds]

        if model_type == "rf":
            # maps ECoG directly onto ROI space
            # For random forest, project data from electrodes to ROIs in advance
            X_train = self.roi_proj_rf(
                X_train, sbj_order_train, self.nROIs, self.proj_mat_out
            )
            X_validate = self.roi_proj_rf(
                X_validate, sbj_order_validate, self.nROIs, self.proj_mat_out
            )
            X_test = self.roi_proj_rf(
                X_test, sbj_order_test, self.nROIs, self.proj_mat_out
            )

        elif model_type == "riemann":
            # maps ECoG directly onto ROI space
            # Project data from electrodes to ROIs in advance
            X_train_proj = self.roi_proj_rf(
                X_train, sbj_order_train, self.nROIs, self.proj_mat_out
            )
            X_validate_proj = self.roi_proj_rf(
                X_validate, sbj_order_validate, self.nROIs, self.proj_mat_out
            )
            X_test_proj = self.roi_proj_rf(
                X_test, sbj_order_test, self.nROIs, self.proj_mat_out
            )

            # Reshape into 3 dimensions
            X_train = X_train_proj.reshape(
                (X_train.shape[0], -1, X_train.shape[-1]))
            X_validate = X_validate_proj.reshape(
                (X_validate.shape[0], -1, X_validate.shape[-1])
            )
            X_test = X_test_proj.reshape(
                (X_test.shape[0], -1, X_test.shape[-1]))

            # Find any events where std is 0
            X_train, X_validate, X_test = self.riemann_remove_bad_events(
                X_train, X_validate, X_test
            )

        else:
            # Reformat data size for NN fitting
            # Use commented out bits if doing classification
            # Y_train = np_utils.to_categorical(Y_train-1)
            X_train = np.expand_dims(X_train, 1)
            # Y_validate = np_utils.to_categorical(Y_validate-1)
            X_validate = np.expand_dims(X_validate, 1)
            # Y_test = np_utils.to_categorical(Y_test-1)
            X_test = np.expand_dims(X_test, 1)
            proj_mat_out_nn = np.expand_dims(self.proj_mat_out, 1)

        cur_fold_data = Fold_Data_Container(
            Y_train,
            X_train,
            Y_validate,
            X_validate,
            Y_test,
            X_test,
            model_type,
            sbj_order_train,
            sbj_order_validate,
            sbj_order_test,
            proj_mat_out_nn,
            nb_classes=self.nb_classes,
            nROIs=self.nROIs,
            ecog_srate=self.ecog_srate,
            class_or_reg=self.class_or_reg,
        )
        return cur_fold_data

    def calc_mutual_info(self, freq_bands):
        # get the spectral power to use as features
        self.calc_spectral_power(freq_bands)

        print("Calculating mutual information")
        # calculate mutual info for all sbjs, channels and features
        mutual_info = []
        for sbj, s in tqdm(self.pats_ids_dict.items()):
            sbj_power_fts_df = self.power_fts_df[(
                self.power_fts_df["Subject"] == sbj)]
            channels = sbj_power_fts_df["Channel"].unique()
            for chan in channels:
                sbj_chan_power_fts_df = sbj_power_fts_df[
                    (sbj_power_fts_df["Channel"] == chan)
                ]
                band_names = [
                    b_name + " Power" for b_name in freq_bands.keys()]
                sbj_chan_power_fts = sbj_chan_power_fts_df[band_names].to_numpy(
                )
                sbj_chan_y = np.squeeze(
                    sbj_chan_power_fts_df[["y"]].to_numpy())

                mi_calc = mutual_info_regression(
                    sbj_chan_power_fts, sbj_chan_y)
                for b, band in enumerate(freq_bands.keys()):
                    tmp_row = [sbj, chan, band, mi_calc[b]]
                    mutual_info.append(tmp_row)

        self.mi_df = pd.DataFrame(
            mutual_info,
            columns=["Subject", "Channel", "Freq Band", "Mutual Information"],
        )

        # create mutual info df that is filtered to the best mi score across channels for each feature
        max_mi = []
        for sbj, s in self.pats_ids_dict.items():
            for b, band in enumerate(freq_bands.keys()):
                filter_df = self.mi_df[
                    (self.mi_df["Freq Band"] == band) & (
                        self.mi_df["Subject"] == sbj)
                ]
                max_index = filter_df["Mutual Information"].argmax()
                max_row = filter_df.iloc[max_index]
                max_mi.append(
                    [
                        max_row["Subject"],
                        max_row["Channel"],
                        max_row["Freq Band"],
                        max_row["Mutual Information"],
                    ]
                )

        self.max_mi_df = pd.DataFrame(
            max_mi, columns=["Subject", "Channel",
                             "Freq Band", "Mutual Information"]
        )

    # INTERNAL METHODS
    def calc_spectral_power(self, freq_bands):
        # NOTE: Currently will only work for X, not X_test, could be worth to work for that as well eventually
        power_fts = []
        print("calculating spectral power")

        for sbj, s in tqdm(self.pats_ids_dict.items()):
            cur_sbj_inds = np.nonzero(self.sbj_order == s)[0]
            cur_sbj_X = self.X[cur_sbj_inds]
            cur_sbj_y = self.y[cur_sbj_inds]
            for sample in range(cur_sbj_X.shape[0]):
                samp_y = cur_sbj_y[sample]
                for chan in range(cur_sbj_X.shape[1]):
                    cur_samp_chan = cur_sbj_X[sample, chan]
                    freq, pxx = welch(
                        cur_samp_chan,
                        fs=self.ecog_srate,
                        nperseg=250,
                        scaling="spectrum",
                    )

                    tmp_row = [sbj, chan]
                    for f_band in freq_bands.values():
                        band_indices = (np.where(freq == f_band[0]))[0][0], np.where(
                            freq == f_band[1]
                        )[0][0]
                        band_power = np.mean(
                            pxx[band_indices[0]: band_indices[1]])
                        tmp_row.append(band_power)
                    tmp_row.append(samp_y)

                    power_fts.append(tmp_row)

        col_names = ["Subject", "Channel"]
        for b_name in freq_bands.keys():
            col_names.append(b_name + " Power")
        col_names.append("y")

        self.power_fts_df = pd.DataFrame(power_fts, columns=col_names)
        print(self.power_fts_df.head())

    def riemann_remove_bad_events(self, X_train, X_validate, X_test):
        """
        Removes 'bad events' from train, val, and test data
        (max standard deviation = 0) according to riemann model
        """
        # Find any events where std is 0
        train_inds_bad = np.nonzero(X_train.std(axis=-1).max(axis=-1) == 0)[0]
        val_inds_bad = np.nonzero(X_validate.std(axis=-1).max(axis=-1) == 0)[0]
        test_inds_bad = np.nonzero(X_test.std(axis=-1).max(axis=-1) == 0)[0]
        if not not train_inds_bad.tolist():
            first_good_ind = np.setdiff1d(np.arange(X_train.shape[0]), train_inds_bad)[
                0
            ]
            X_train[train_inds_bad, ...] = X_train[
                (train_inds_bad * 0) + first_good_ind, ...
            ]
        if not not val_inds_bad.tolist():
            first_good_ind = np.setdiff1d(np.arange(X_validate.shape[0]), val_inds_bad)[
                0
            ]
            X_validate[val_inds_bad, ...] = X_validate[
                (val_inds_bad * 0) + first_good_ind, ...
            ]
        if not not test_inds_bad.tolist():
            first_good_ind = np.setdiff1d(
                np.arange(X_test.shape[0]), test_inds_bad)[0]
            X_test[test_inds_bad, ...] = X_test[
                (test_inds_bad * 0) + first_good_ind, ...
            ]

        return X_train, X_validate, X_test

    def remove_nan_data(self):
        train_bad_inds = ~np.isnan(self.sbj_X).any(axis=2).any(axis=1)
        test_bad_inds = ~np.isnan(self.sbj_X_test).any(axis=2).any(axis=1)
        self.sbj_X = self.sbj_X[train_bad_inds, ...]
        self.sbj_y = self.sbj_y[train_bad_inds, ...]
        self.sbj_X_test = self.sbj_X_test[test_bad_inds, ...]
        self.sbj_y_test = self.sbj_y_test[test_bad_inds, ...]

    def get_quadrants(self, y):
        """
        Turns labels for reach angle into discrete classes
        based on which quadrant the angle lies in
        """
        for i, a in enumerate(y):
            if -45 <= a < 45:
                y[i] = 1
            elif 45 <= a < 135:
                y[i] = 2
            elif -45 > a > -135:
                y[i] = 4
            else:
                y[i] = 3

        return y

    def balance_quads(self, X, y):
        unique_labels, label_counts = self.get_class_distrib(y)
        label_min = label_counts.min()
        if len(unique_labels) != 4:
            sys.exit("Less than 4 quadrants in the data.")

        print(self.get_class_distrib(y))

        # go through and grab all of the indicies we want to keep
        all_indices_keep = []
        for l, label in enumerate(unique_labels):
            indices_for_label = np.where(y == label)[0]
            indices_to_keep = np.random.choice(
                indices_for_label, size=label_min, replace=False
            )
            all_indices_keep.extend(indices_to_keep)

        # then filter the data
        random.shuffle(all_indices_keep)
        y = y[all_indices_keep]
        X = X[all_indices_keep, :, :]

        return X, y

    #  TO DO: Added a new one to manifold utils to use and test there
    def roi_proj_rf(self, X_in, type="train"):
        """
        Project spectral power from electrodes to ROI's prior for random forest classification
        """
        # Project to ROIs using matrix multiply
        X_in_sh = list(X_in.shape)
        X_in_sh[1] = self.nROIs
        X_in_proj = np.zeros(X_in_sh)
        if type == "train":
            sbj_ord = self.sbj_order
        else:
            sbj_ord = self.sbj_order_test
        for s in range(X_in.shape[0]):
            X_in_proj[s, ...] = self.proj_mat_out[sbj_ord[s], ...] @ X_in[s, ...]
        del X_in
        # not useful if just wanting to project to roi, and not use for RF
        # X_in_proj = X_in_proj.reshape(X_in_proj.shape[0], -1)

        return X_in_proj

    def subject_data_inds(
        self,
        sbj,
        sbj_order,
        labels_unique,
        frodo,
        save_string,
        half_n_evs,
        y,
        sp,
        n_folds,
        inds,
    ):
        """
        Determine the indices for train, val, or test sets, ensuring that:
            number of move events = number of rest events = half_n_evs
        """
        for j in sbj.tolist():
            inds_tmp_orig = np.nonzero(sbj_order == j)[
                0
            ]  # select inds for 1 subject at a time
            y_labs = y[inds_tmp_orig]
            for i, lab_uni in enumerate(labels_unique):
                inds_tmp = inds_tmp_orig[y_labs == lab_uni]
                # Make randomization for each fold the same across models
                order_inds = np.arange(len(inds_tmp))
                np.random.shuffle(order_inds)  # randomize order

                inds_tmp = inds_tmp[order_inds]
                if half_n_evs != "nopad":
                    if len(inds_tmp) < half_n_evs:
                        # Make length >= half_n_evs
                        inds_tmp = list(inds_tmp) * \
                            ((half_n_evs // len(inds_tmp)) + 1)
                        inds_tmp = np.asarray(inds_tmp)[:half_n_evs]
                    else:
                        inds_tmp = inds_tmp[:half_n_evs]

                if i == 0:
                    inds_sbj = inds_tmp.copy()
                else:
                    inds_sbj = np.concatenate((inds_sbj, inds_tmp), axis=0)

            inds += list(inds_sbj)

        return np.asarray(inds)

    def folds_choose_subjects(self, n_folds, sbj_ids_all, n_test=1, n_val=4, n_train=7):
        """
        Randomly pick train/val/test subjects for each fold
        (Updated to assign test subject evenly across subjects (if n_test=1.)
        """
        n_subjs = len(sbj_ids_all)
        sbj_inds_all_train, sbj_inds_all_val, sbj_inds_all_test = [], [], []
        for frodo in range(n_folds):
            if n_test == 1:
                # Assign test subject as evenly as possible (still done randomly, using random permutation)
                if frodo % n_subjs == 0:
                    # New random permutation of test subjects after iterate through all subjects
                    test_sbj_count = 0
                    test_sbj_all = np.random.permutation(n_subjs)
                    if n_val == 1:
                        # Assign validation subject evenly too
                        val_sbj_all = np.zeros(
                            [
                                n_subjs,
                            ]
                        )
                        while np.any(val_sbj_all == test_sbj_all):
                            # Generate permutation that doesn't overlap with test subjects
                            val_sbj_all = np.random.permutation(n_subjs)
                sbj_inds = np.arange(n_subjs)
                curr_test_ind = test_sbj_all[test_sbj_count]
                sbj_inds_all_test.append(np.asarray([sbj_inds[curr_test_ind]]))
                if n_val == 1:
                    curr_val_ind = val_sbj_all[test_sbj_count]
                    sbj_inds_all_val.append(
                        np.asarray([sbj_inds[curr_val_ind]]))
                    sbj_inds = np.delete(
                        sbj_inds, np.array([curr_test_ind, curr_val_ind])
                    )
                    np.random.shuffle(sbj_inds)
                    sbj_inds_all_train.append(sbj_inds[:][:n_train])
                else:
                    sbj_inds = np.delete(sbj_inds, curr_test_ind)
                    np.random.shuffle(sbj_inds)
                    sbj_inds_all_val.append(sbj_inds[:n_val])
                    sbj_inds_all_train.append(sbj_inds[n_val:][:n_train])
                test_sbj_count += 1
            else:
                sbj_inds = np.arange(n_subjs)
                np.random.shuffle(sbj_inds)
                sbj_inds_all_test.append(sbj_inds[:n_test])
                sbj_inds_all_val.append(sbj_inds[n_test: (n_val + n_test)])
                sbj_inds_all_train.append(
                    sbj_inds[(n_val + n_test):][:n_train])

        return sbj_inds_all_train, sbj_inds_all_val, sbj_inds_all_test

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

        # pdb.set_trace()
        # I think lowering the threshold here would get more ROIs
        # But that doesn't seem to be the whole story
        dipole_densities = dipole_densities / len(patient_ids)
        if custom_roi_inds is None:
            good_ROIs = np.nonzero(np.asarray(
                dipole_densities) > dipole_dens_thresh)[0]
        else:
            good_ROIs = custom_roi_inds.copy()

        # Now create projection matrix output (patients x roi x chans)
        n_pats = len(patient_ids)
        proj_mat_out = np.zeros([n_pats, len(good_ROIs), n_chans_all])
        # pdb.set_trace()
        chan_ind_vals_all = []
        for s, patient in enumerate(patient_ids):
            # pdb.set_trace()
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

    def generate_sine_data(self, time_stamps, freq, sbj_data_samples):
        """
        Creates fake sine data of channels x time, same shape as ECoG data
        """
        data_samples = sbj_data_samples * len(self.pats_ids_in)

        tmp_sbj_order = np.asarray(
            [
                [sbj for i in range(sbj_data_samples)]
                for sbj in range(len(self.pats_ids_in))
            ]
        )
        tmp_sbj_order = np.reshape(tmp_sbj_order, data_samples)

        all_sine_data = np.zeros((data_samples, self.n_chans_all, time_stamps))
        all_sine_labels = np.zeros((data_samples))
        t = np.linspace(-1, 1, time_stamps)  # time coordinates
        for i in range(data_samples):
            sine_by_chan = np.zeros((self.n_chans_all, time_stamps))
            # rand_amp = np.random.uniform(low=1.0, high=10.0, size=(1))
            rand_amp = 2.0
            rand_phase = np.random.uniform(low=-1.0, high=1.0, size=(1))
            for ch in range(self.n_chans_all):
                noise_sample = np.random.uniform(
                    low=-0.2, high=0.2, size=(time_stamps))
                # sine_wave = ( (np.sin((2*np.pi)*freq*t)) * rand_amp ) + noise_sample #for predicting amplitude
                sine_wave = (
                    (np.sin((2 * np.pi) * freq * t + (rand_phase * np.pi))) * rand_amp
                ) + noise_sample  # for predicting phase/angle

                sine_by_chan[ch] = sine_wave

            all_sine_data[i] = sine_by_chan
            # all_sine_labels[i] = rand_amp
            all_sine_labels[i] = rand_phase * np.pi

        return all_sine_data, all_sine_labels, tmp_sbj_order

    # TO DO: add comments about expected shape
    def emg_get_mav(self, x):
        return np.mean(np.absolute(x), axis=-1)

    def emg_get_logvar(self, x):
        return np.log(np.var(x, axis=-1))

    def emg_get_wl(self, x):
        return np.sum(np.absolute(np.diff(x, axis=-1)), axis=-1)

    def emg_get_ar(self, x):
        emg_ar = []
        for sample in range(x.shape[0]):
            emg_sample = x[sample]
            chan_ar = []
            for chan in range(emg_sample.shape[0]):
                emg_chan_sample = emg_sample[chan]

                # get_ar_feat from pyEMG author
                tmp_emg_ar = get_ar_feat(emg_chan_sample)
                chan_ar.append(np.squeeze(tmp_emg_ar))

            emg_ar.append(chan_ar)

        return np.array(emg_ar)

    def remove_zero_channels(self, x):
        dp = x[0, :, :]
        # BUG: axis needs to be whatever the timepoint axis is? Probably depends on the data
        # pdb.set_trace()
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

    def emg_get_lr_feats(self, x_train, acc_train, x_test, acc_test):
        # remove zeroed channels before tranforming
        x_train = self.remove_zero_channels(x_train)
        x_test = self.remove_zero_channels(x_test)

        train_mav = self.emg_get_mav(x_train)
        train_logvar = self.emg_get_logvar(x_train)
        train_wl = self.emg_get_wl(x_train)
        train_ar = self.emg_get_ar(x_train)

        # x_train_fts = np.concatenate((train_mav, train_logvar, train_wl, acc_train,
        #                             train_ar[:,:,0], train_ar[:,:,1], train_ar[:,:,2], train_ar[:,:,3]), axis = 1)
        # Use line below for removing features for albation testing
        x_train_fts = np.concatenate(
            (
                train_mav,
                train_logvar,
                train_wl,
                train_ar[:, :, 0],
                train_ar[:, :, 1],
                train_ar[:, :, 2],
                train_ar[:, :, 3],
            ),
            axis=1,
        )

        test_mav = self.emg_get_mav(x_test)
        test_logvar = self.emg_get_logvar(x_test)
        test_wl = self.emg_get_wl(x_test)
        test_ar = self.emg_get_ar(x_test)

        # x_test_fts = np.concatenate((test_mav, test_logvar, test_wl, acc_test,
        #                             test_ar[:,:,0], test_ar[:,:,1], test_ar[:,:,2], test_ar[:,:,3]), axis = 1)
        # Use line below for removing features for albation testing
        x_test_fts = np.concatenate(
            (
                test_mav,
                test_logvar,
                test_wl,
                test_ar[:, :, 0],
                test_ar[:, :, 1],
                test_ar[:, :, 2],
                test_ar[:, :, 3],
            ),
            axis=1,
        )

        return x_train_fts, x_test_fts

    # Bandpass filtering code from
    # https://scipy-cookbook.readthedocs.io/items/ButterworthBandpass.html
    def butter_bandpass(self, lowcut, highcut, fs, order=4):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        sos = butter(order, [low, high], btype="band", output="sos")
        return sos

    def butter_bandpass_filter(self, data, lowcut, highcut, fs, order=4):
        sos = self.butter_bandpass(lowcut, highcut, fs, order=order)
        y = sosfiltfilt(sos, data, axis=0)
        return y

    def emg_remove_rest_events(
        self, emg_data, X, y, acc_data, glove_data, event_marker, rep_marker
    ):
        rest_indicies = np.where(emg_data["stimulus"] == 0)[0]
        X_rest_removed = np.delete(X, rest_indicies, axis=0)
        y_rr = np.delete(y, rest_indicies)
        acc_data_rr = np.delete(acc_data, rest_indicies, axis=0)
        acc_data_rr = acc_data_rr.transpose()
        glove_data_rr = np.delete(glove_data, rest_indicies, axis=0)
        glove_data_rr = glove_data_rr.transpose()

        event_marker_rr = np.delete(event_marker, rest_indicies)
        rep_marker_rr = np.delete(rep_marker, rest_indicies)

        return (
            X_rest_removed,
            y_rr,
            acc_data_rr,
            glove_data_rr,
            event_marker_rr,
            rep_marker_rr,
        )

    # Why did I not end up using this function??
    def get_ecog_freq_bands(self, ecog_data):
        # want to extract all of these freq bands
        # TO DO: get this info into JSON file
        freq_bands = {
            "LFO": [0.5, 4],
            "Alpha": [8, 14],
            "Beta": [14, 30],
            "Low Gamma": [30, 60],
            "Gamma": [60, 100],
            "High Gamma": [100, 200],
        }
        total_freqs = len(freq_bands)

        data_samps = ecog_data.shape[0]
        chans = ecog_data.shape[1]
        time_samps = ecog_data.shape[2]

        sbj_filt_ECoG = np.zeros((total_freqs, data_samps, chans, time_samps))
        envelope_sbj_filt_ECoG = np.zeros(
            (total_freqs, data_samps, chans, time_samps))

        for f, cur_freqs in enumerate(freq_bands.values()):
            print("Freq calcs for ", cur_freqs)
            low_f = cur_freqs[0]
            high_f = cur_freqs[1]

            # first do a 4th order butterworth filter
            tmp_filt_data = self.butter_bandpass_filter(
                ecog_data.T, low_f, high_f, self.ecog_srate, order=4
            )
            # print(tmp_filt_data.shape)
            sbj_filt_ECoG[f, ...] = tmp_filt_data.T
            # print(tmp_filt_data.T.shape)

            # print("Calc Hilbert")
            # then use the hilbert transform to extract the envelope
            # need to loop over each trial
            analytic_sbj_filt_ECoG = hilbert(sbj_filt_ECoG[f, ...])
            # print(analytic_sbj_filt_ECoG.shape)
            envelope_sbj_filt_ECoG[f, ...] = np.abs(analytic_sbj_filt_ECoG)
            # print(envelope_sbj_filt_ECoG[f,...].shape)

        freq_labels = envelope_sbj_filt_ECoG.mean(axis=3).mean(axis=2)
        # print(freq_labels.shape)

        return freq_labels.T

    def load_emg_data(self, pats_ids_in, lp):
        """
        Loads in the EMG data, performs bandpass filtering, and splits into train and test
        """

        X_filt_wind, X_test = [], []
        sbj_order, sbj_order_test = [], []
        y_avged, y_test = [], []
        acc_avged, acc_test = [], []
        glove_avged, glove_test = [], []

        for sbj in tqdm(range(len(pats_ids_in))):
            # load in the emg data
            # sbj = 0
            emg_file_name = (
                "DB2_"
                + pats_ids_in[sbj].lower()
                + "/"
                + pats_ids_in[sbj]
                + "_E2_A1.mat"
            )
            emg_data = scipy.io.loadmat(lp + emg_file_name)

            # pull columns from emg mat file and set filter params
            X = emg_data["emg"]
            y = emg_data["glove"][:, -2]
            acc_data = emg_data["acc"]
            glove_data = emg_data["glove"]
            event_marker = emg_data["stimulus"]
            rep_marker = emg_data["repetition"]

            fs = 2000
            lowcut = 20
            highcut = 500

            # remove rest
            (
                X_rest_removed,
                y_rr,
                acc_data_rr,
                glove_data_rr,
                event_marker_rr,
                rep_marker_rr,
            ) = self.emg_remove_rest_events(
                emg_data, X, y, acc_data, glove_data, event_marker, rep_marker
            )

            # do the bandpass filter
            X_bfilter = self.butter_bandpass_filter(
                X_rest_removed, lowcut, highcut, fs)
            X_bfilter = X_bfilter.transpose()

            # Set the params for windowing the data, and create containers
            window = 512
            increment = 200
            samples = int(X_bfilter.shape[1] / increment) - 2
            chans = X_bfilter.shape[0]

            # loop through the events and repetitions to split data
            events = np.unique(emg_data["stimulus"])[1:]
            for e in events:
                event_indices = np.where(event_marker_rr == e)
                event_indices = event_indices[0]
                event_reps = rep_marker_rr[event_indices]

                X_for_event = X_bfilter[:, event_indices]
                y_for_event = y_rr[event_indices]
                acc_for_event = acc_data_rr[:, event_indices]
                glove_for_event = glove_data_rr[:, event_indices]

                # pick a random repetition to become test data
                test_ind = np.random.randint(1, 8)
                for r in range(1, 7):
                    event_rep_indices = np.where(event_reps == r)
                    event_rep_indices = event_rep_indices[0]

                    X_event_rep = X_for_event[:, event_rep_indices]
                    y_event_rep = y_for_event[event_rep_indices]
                    acc_event_rep = acc_for_event[:, event_rep_indices]
                    glove_event_rep = glove_for_event[:, event_rep_indices]

                    for i in range(0, X_event_rep.shape[1], increment):
                        if i + window < X_event_rep.shape[1]:
                            if r == test_ind:
                                X_test.append(X_event_rep[:, i: i + window])
                                y_test.append(
                                    y_event_rep[i: i + window].mean())
                                acc_test.append(
                                    acc_event_rep[:, i: i +
                                                  window].mean(axis=1)
                                )
                                glove_test.append(
                                    glove_event_rep[:, i: i +
                                                    window].mean(axis=1)
                                )
                                sbj_order_test.append(sbj)
                            else:
                                X_filt_wind.append(
                                    X_event_rep[:, i: i + window])
                                y_avged.append(
                                    y_event_rep[i: i + window].mean())
                                acc_avged.append(
                                    acc_event_rep[:, i: i +
                                                  window].mean(axis=1)
                                )
                                glove_avged.append(
                                    glove_event_rep[:, i: i +
                                                    window].mean(axis=1)
                                )
                                sbj_order.append(sbj)

        return (
            np.array(X_filt_wind),
            np.array(y_avged),
            np.array(X_test),
            np.array(y_test),
            np.array(acc_avged),
            np.array(glove_avged),
            np.array(acc_test),
            np.array(glove_test),
            np.array(sbj_order),
            np.array(sbj_order_test),
        )

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
        # ZSH: correct pat_ids_in type if needed
        if not isinstance(pats_ids_in, list):
            pats_ids_in = [pats_ids_in]
        if len(test_day) == 1:
            test_day = test_day[0]
        sbj_order, sbj_order_test = [], []
        X_test_subj, y_test_subj = [], []  # placeholder vals

        regr_sbjs = [
            "subj_01",
            "subj_02",
            "subj_03",
            "subj_04",
            "subj_05",
            "subj_06",
            "subj_07",
            "subj_08",
            "subj_09",
            "subj_10",
            "subj_11",
            "subj_12",
        ]
        sbj_map = {pats_ids_in[i]: regr_sbjs[i]
                   for i in range(len(pats_ids_in))}
        # sbj_map = {
        #     "EC01": "subj_01",
        #     "EC02": "subj_02",
        #     "EC03": "subj_03",
        #     "EC04": "subj_04",
        #     "EC05": "subj_05",
        #     "EC06": "subj_06",
        #     "EC07": "subj_07",
        #     "EC08": "subj_08",
        #     "EC09": "subj_09",
        #     "EC10": "subj_10",
        #     "EC11": "subj_11",
        #     "EC12": "subj_12",
        # }

        # Gather each subjects data, and concatenate all days
        for j in tqdm(range(len(pats_ids_in))):
            pat_curr = pats_ids_in[j]

            # ZSH: load in ecog data and metatdata for current sbj
            if "steve_xr" in self.data_format:
                print("loading steve data")
                (
                    ep_data_in,
                    ep_metadata_in,
                    time_inds,
                    n_ecog_chans,
                ) = self.load_steve_ecog(
                    lp, pat_curr, exp_params["tlim"], feat_dat
                )
                # print(np.unique(ep_data_in["events"]))
            elif "zeynep" in lp:
                print("loading in zeynep data")
                # load in cont data from zeynep
                (
                    ep_data_in,
                    ep_metadata_in,
                    time_inds,
                    n_ecog_chans,
                ) = self.load_zeynep_ecog(lp, pat_curr, exp_params["tlim"])
            elif (("experimental_ecog_data" in lp) and ('pose' not in lp)) or ("fingerflex" in lp):
                print("loading in fingerflex data")
                (
                    ep_data_in,
                    ep_metadata_in,
                    time_inds,
                    n_ecog_chans,
                ) = self.load_fingerflex_data(lp, pat_curr, exp_params["tlim"])
            elif "zsteineh" in lp:
                print("loading zoe data")
                (
                    ep_data_in,
                    ep_metadata_in,
                    time_inds,
                    n_ecog_chans,
                ) = self.load_zoe_pose(
                    lp, pat_curr, exp_params["tlim"], feat_dat, sbj_map
                )

            else:
                print("loading sat data")
                # loading in from sat data lockers if not steve or zeynep
                (
                    ep_data_in,
                    ep_metadata_in,
                    time_inds,
                    n_ecog_chans,
                ) = self.load_sat_ecog(lp, pat_curr, exp_params["tlim"], feat_dat)

            # print(ep_data_in)
            # ZSH: determine right number of channels
            if n_chans_all < n_ecog_chans:
                n_chans_curr = n_chans_all
            else:
                n_chans_curr = n_ecog_chans
            # print("num chans", n_chans_curr)

            # Next few lines: Split into train and test data
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
            dat_train, labels_train, dat_test, labels_test = self.dl_check_balance(
                dat_train, labels_train, dat_test, labels_test, exp_params
            )

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

    def load_steve_ecog(self, lp, pat_curr, tlim, feat_dat):
        # ZSH: what is the datatype of ep_data_in???
        # ZSH: xarray object
        if self.dataset == "pose":
            ep_data_in = xr.open_dataset(lp + pat_curr + "_pose_data.nc")
        else:
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

    def load_zoe_pose(self, lp, pat_curr, tlim, feat_dat, sbj_map):
        # TO DO: Finish this and make work
        pose_data_in = xr.open_dataset(lp + pat_curr + "_pose_data.nc")
        print(pose_data_in)
        # pdb.set_trace()
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

    def load_zeynep_ecog(self, lp, pat_curr, tlim, crop_val=0):
        # code from zeynep
        print("loading from: ", lp + pat_curr + "-epo.fif")
        f_load = natsort.natsorted(glob.glob(lp + pat_curr + "-epo.fif"))[0]
        print("for sbj", pat_curr, "f_load is", f_load)

        if crop_val > 0:
            all_epochs = mne.read_epochs(f_load).crop(
                tmin=-1 * crop_val, tmax=crop_val, include_tmax=True
            )
        else:
            all_epochs = mne.read_epochs(f_load)

        dat = all_epochs.get_data()
        n_ecog_chans = dat.shape[1]
        time_inds = np.array(range(dat.shape[2]))

        if self.feat_dat == "ssl_domain_fts":
            freq_labels = self.get_ecog_freq_bands(dat)

        # convert to xarray for consistency
        xr_days = np.zeros(dat.shape[0]) + 4
        ecog_xr = xr.Dataset(
            {"ecog_data": (["events", "channels", "time"], dat)},
            coords={
                "events": (["events"], xr_days),
                "time": (
                    ["time"],
                    np.linspace(tlim[0], tlim[1], np.array(dat).shape[2]),
                ),
                "channels": (["channels"], np.array(range(n_ecog_chans))),
            },
        )

        return ecog_xr, freq_labels, time_inds, n_ecog_chans

    def load_sat_ecog(self, lp, pat_curr, tlim, feat_dat):
        # helper functions for Sat data
        def _pull_time(row):
            timestamp_str = row["event_timestamp"]
            return timestamp_str[7:]

        def _get_eventspans_from_events_ajile(
            events, window_before=0.2, window_after=0.8
        ):
            """
            From satpreet-analysis repo
            Eventspan: window_before (sec) --[Event]-- window_after (sec)
            """
            eventspans = events.copy()
            window_before = pd.Timedelta(window_before, unit="s")
            window_after = pd.Timedelta(window_after, unit="s")
            eventspans["start_time"] = eventspans["time"] - window_before
            eventspans["end_time"] = eventspans["time"] + window_after
            eventspans["event_timespan"] = (
                eventspans["end_time"] - eventspans["start_time"]
            )
            return eventspans

        ecog_sp = "/home/zsteineh/research_projects/neural_manifolds/data/"
        region_to_int = {"C" + str(i): i for i in range(1, 9)}
        # sbj_ids = {"subj_01": "a0f66459"}

        if os.path.isfile(ecog_sp + "reach_region/" + pat_curr + "_region_ecog.nc"):
            ecog_xr = xr.open_dataset(
                ecog_sp + "reach_region/" + pat_curr + "_region_ecog.nc"
            )
            ep_times = np.asarray(ecog_xr.time)
            time_inds = np.nonzero(
                np.logical_and(ep_times >= tlim[0], ep_times <= tlim[1])
            )[0]
            n_ecog_chans = len(ecog_xr.channels)
        else:
            # get the movement feature data loaded in
            ecog_data_filename = (
                "events_enhanced_multitrack_r_wrist_move_{}_2019*.csv".format(
                    pat_curr)
            )
            ecog_data_file = glob.glob(
                lp + ecog_data_filename)[0].split("/")[-1]

            mvmt_fts_df = pd.read_csv(lp + ecog_data_file)
            mvmt_fts_df = mvmt_fts_df[pd.isna(mvmt_fts_df["blacklist"])]
            mvmt_fts_df["time"] = mvmt_fts_df.apply(
                lambda row: _pull_time(row), axis=1)
            colnames = [
                "day",
                "time",
                "event_timestamp",
                "mvmt",
                "region_start",
                "region_end",
            ]
            mvmt_fts_df = mvmt_fts_df[colnames].query('mvmt != "mv_0"')
            mvmt_fts_df = _get_eventspans_from_events_ajile(
                mvmt_fts_df, window_before=(-1 * tlim[0]), window_after=tlim[1]
            )

            print(mvmt_fts_df.head())

            # # balance the data
            # mvmt_fts_df, region_to_int = self.balance_data(mvmt_fts_df, region_to_int, label_max=100)
            # print("label distrib after balancing")
            # print(mvmt_fts_df.head())
            # print(self.get_class_distrib(np.array(list(map(region_to_int.get, mvmt_fts_df['region_start'])))))

            patient_days = mvmt_fts_df["day"].unique()
            patient_days.sort()
            patient_days = patient_days[1:]
            Xs = []
            ys = []
            xr_days = []
            # now go find the coresponding ecog data at the times in the feature dataframe
            for day in tqdm(patient_days):
                mvmt_fts_day_df = mvmt_fts_df.loc[mvmt_fts_df["day"] == day]
                ecog_d, f_ecog, ecog_offset, metadata_dict = ecogu.load_ajile_ecog_data(
                    pat_curr, day, verbose=True
                )
                print("ecog_d.shape", ecog_d.shape, "f_ecog:", f_ecog)

                cur_chans = ecog_d.shape[0]
                for idx, row in mvmt_fts_day_df.iterrows():
                    start_time = row["start_time"]
                    end_time = row["end_time"]

                    one_sec = pd.Timedelta(1, unit="s")
                    idxs = np.arange(
                        (start_time / one_sec) * f_ecog,
                        (end_time / one_sec) * f_ecog + 1,
                        dtype="int",
                    )
                    idxs -= ecog_offset
                    event_ecog_data = ecog_d[:, idxs]
                    num_samps = (f_ecog * (tlim[1] - tlim[0])) + 1
                    if (
                        event_ecog_data.shape[0] != cur_chans
                        or event_ecog_data.shape[1] != num_samps
                    ):
                        # pdb.set_trace()
                        mvmt_fts_day_df.drop(labels=idx, inplace=True)
                        continue
                    Xs.append(event_ecog_data)
                    ys.append(row[feat_dat])
                xr_days += [day] * len(mvmt_fts_day_df)

            # convert to xarray for consistency with Steve's data
            ys_int = list(map(region_to_int.get, ys))
            time_inds = np.array(range(np.array(Xs).shape[2]))
            n_ecog_chans = np.array(Xs).shape[1]
            combined_Xs_ys = np.insert(
                np.array(Xs),
                n_ecog_chans,
                np.tile(np.array(ys_int), (np.array(Xs).shape[2], 1)).T,
                axis=1,
            )
            ecog_xr = xr.Dataset(
                {"ecog_data": (["events", "channels", "time"],
                               combined_Xs_ys)},
                coords={
                    "events": (["events"], xr_days),
                    "time": (
                        ["time"],
                        np.linspace(tlim[0], tlim[1], np.array(Xs).shape[2]),
                    ),
                    "channels": (["channels"], np.array(range(n_ecog_chans + 1))),
                },
            )
            print(ecog_xr)

            ecog_xr.to_netcdf(ecog_sp + "reach_region/" +
                              pat_curr + "_region_ecog.nc")

        return ecog_xr, None, time_inds, n_ecog_chans

    def get_metadata_labels(self, ep_data, ep_metadata_in):
        htnet_gen_labels = ep_data.to_array()[:, :, -1, 0]
        htnet_gen_labels = np.squeeze(np.array(htnet_gen_labels))
        # well fuck...
        # event_dict = {'rest':1,'move':2}
        move_indicies = np.where(htnet_gen_labels == 2)[0]
        rest_indicies = np.where(htnet_gen_labels == 1)[0]
        assert (
            move_indicies.shape[0] == rest_indicies.shape[0]
        ), "does not contain the same number of move and rest labels"
        # pdb.set_trace()
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

    # there could be a way to refactor and combine these two fns
    def align_m_v_r_angle(self, ep_data, ep_metadata_in):
        htnet_gen_labels = ep_data.to_array()[:, :, -1, 0]
        htnet_gen_labels = np.squeeze(np.array(htnet_gen_labels))
        # well fuck...
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
            days_test_inds = []
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
        if self.feat_dat == "ssl_domain_fts":
            labels_train = ep_metadata_in.copy()
        elif self.feat_dat == "mvr_quads":
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

    def dl_check_balance(
        self, dat_train, labels_train, dat_test, labels_test, exp_params
    ):
        if exp_params["quadrants"]:
            labels_train = self.get_quadrants(labels_train)
            labels_test = self.get_quadrants(labels_test)
            print("changed to quads:", labels_train)
            # exp_params['class_or_reg'] = 'class'
            # self.class_or_reg = 'class'

            # balance here
            # TO DO: make a variable that controls this
            dat_train, labels_train = self.balance_quads(
                dat_train, labels_train)
            dat_test, labels_test = self.balance_quads(dat_test, labels_test)

            print("label distribution after train test split")
            print(self.get_class_distrib(labels_train))
            print(self.get_class_distrib(labels_test))

        return dat_train, labels_train, dat_test, labels_test

    def dl_pad_electrodes(self, pats_ids_in, n_chans_all, n_ecog_chans, data):
        # pdb.set_trace()
        if (len(pats_ids_in) > 1) and (n_chans_all > n_ecog_chans):
            dat_sh = list(data.shape)
            dat_sh[1] = n_chans_all
            # Create dataset padded with zeros if less than n_chans_all, or cut down to n_chans_all
            X_pad = np.zeros(dat_sh)
            X_pad[:, :n_ecog_chans, ...] = data
            data = X_pad.copy()

        return data

    def balance_data(self, mvmt_fts_df, region_to_int, label_max):
        int_labels = list(map(region_to_int.get, mvmt_fts_df["region_start"]))
        unique_labels, label_counts = self.get_class_distrib(
            np.array(int_labels))
        labels_picked = [
            0 if c < label_max else label_max for c in label_counts]

        # update region int based on the labels that didn't make the cut
        region_to_int = {}
        nxt_label = 1
        for i, l_amt in enumerate(labels_picked):
            if l_amt != 0:
                region_to_int["C" + str(i + 1)] = nxt_label
                nxt_label += 1
        # pdb.set_trace()

        balanced_mvmt_df = pd.DataFrame(
            columns=[
                "day",
                "time",
                "event_timestamp",
                "mvmt",
                "region_start",
                "region_end",
                "start_time",
                "end_time",
                "event_timespan",
            ]
        )

        for l, label in enumerate(unique_labels):
            indices_for_label = np.where(np.array(int_labels) == label)[0]
            indices_to_keep = np.random.choice(
                indices_for_label, size=labels_picked[l], replace=False
            )
            balanced_mvmt_df = balanced_mvmt_df.append(
                mvmt_fts_df.iloc[indices_to_keep]
            )

        return balanced_mvmt_df, region_to_int

    def get_class_distrib(self, y_data):
        return np.unique(y_data, return_counts=True)
