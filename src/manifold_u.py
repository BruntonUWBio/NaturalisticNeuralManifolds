from cmath import nan
from re import T
from typing import Tuple
import numpy as np
import xarray as xr
from numpy.lib.format import numpy
import pandas as pd
# import mat73
from scipy.signal import butter, sosfiltfilt, hilbert
from scipy import stats, io
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
# from scipy.cluster import hierarchy
# from scipy.spatial.distance import squareform

# from optht import optht
from tqdm import tqdm
import sys
import os
import pdb
from functools import reduce

from src.data_utils import ECoG_Data

import matplotlib.pyplot as plt
# from mpl_toolkits import mplot3d
from nilearn import plotting as ni_plt
import seaborn as sns
import pickle

# --------------------------------#
### ECOG DATA PREPROCESSING FNS ###
# --------------------------------#


def preprocess_freq_sbj_data(
    exp_params: dict,
    sbj_sp: str,
    freq_bands: dict,
    freq_name: str,
    class_dict: dict,
    manifold_ECoG_data: ECoG_Data,
) -> Tuple[np.ndarray, list, list, int, np.ndarray]:
    """
    Preprocess data for a single participant on a
    single frequency band
    Parameters
    ----------
    exp_params : dict
        Dictionary of experiment parameters
    sbj_sp : str
        Path to where subject results are saved
    freq_bands : dict
        Dictionary of frequency bands, to get freq names and band values
    freq_name : str
        Name of current frequency band to preprocess
    class_dict : dict
        Dictionary of class labels, to get class names
    manifold_ECoG_data : ECoG_Data
        Contains the data class for all participants
        Should already have the current participant's data in sbj_X and sbj_X_test

    Returns
    -------
    cur_classes : np.ndarray
        List of unique class labels for current participant
    trial_dim : list
        List of the number of trials for each class
    sr_dim : list
        List of the number of samples per trial for each class
    chan_dim : int
        Number of channels in the data
    norm_concat_ECoG_data : np.ndarray
        (n_chan, n_trials * n_time)
        Contains the data for the current participant,
        filtered to the current frequency, and then trial concatenated and normalized
    """

    envelope_freq_sbj_X_test, cur_freq_sbj_X_test = extract_analytic_signal(
        freq_bands,
        freq_name,
        exp_params["ecog_srate"],
        manifold_ECoG_data.sbj_X_test,
    )

    if cur_freq_sbj_X_test.tolist() != []:
        plot_raw_and_freq(freq_name, sbj_sp,
                          manifold_ECoG_data, cur_freq_sbj_X_test)
        # do i need this function after adding padding to filter?
        envelope_freq_sbj_X_test = trim_filtered_data(
            [-1, 1], exp_params["ecog_srate"], cur_freq_sbj_X_test
        )

        split_sbj_eLFO_ECoG_data, cur_classes = split_into_classes(
            class_dict,
            manifold_ECoG_data.sbj_y_test[:, 0],
            envelope_freq_sbj_X_test,
        )

        # check that all trials distinct here first
        assert check_repeated_trials(
            split_sbj_eLFO_ECoG_data, cur_classes
        ), "Trials repeated somewhere"
        # plot_trials_before_concat(
        #     freq_name, sbj_sp, class_dict, cur_classes, split_sbj_eLFO_ECoG_data
        # )
        # cool, so looks like the trials are distinct at this point

        print("Concatenating Trials")
        concat_ECoG_data, trial_dim, sr_dim, chan_dim = concat_trials(
            split_sbj_eLFO_ECoG_data
        )
        assert check_repeated_trials(
            concat_ECoG_data, cur_classes, trial_dim, sr_dim, chan_dim
        ), "Trials repeated somewhere"

        print("Normalizing Data")
        norm_concat_ECoG_data = normalize_data(concat_ECoG_data)
        assert check_repeated_trials(
            norm_concat_ECoG_data, cur_classes, trial_dim, sr_dim, chan_dim
        ), "Trials repeated somewhere"
        # need visual inspection of data after normalizing
        plot_after_norm(sbj_sp, freq_name, concat_ECoG_data,
                        norm_concat_ECoG_data)
    else:
        cur_classes = []
        trial_dim = []
        sr_dim = []
        chan_dim = []
        norm_concat_ECoG_data = []

    return cur_classes, trial_dim, sr_dim, chan_dim, norm_concat_ECoG_data


def preprocess_TME_data(
    exp_params: dict,
    sbj_sp: str,
    freq_bands: dict,
    freq_name: str,
    class_dict: dict,
    manifold_ECoG_data: ECoG_Data,
    save_flg: bool = False,
):
    """
    Preprocess data for a single participant and frequency to save
    and send to TME for generating null data
    Returns the average of the data for each class
    Parameters
    ----------
    exp_params : dict
        Dictionary of experiment parameters
    sbj_sp : str
        Path to where subject results are saved, if the save_flg is True
    freq_bands : dict
        Dictionary of frequency bands, to get freq names and band values
    freq_name : str
        Name of current frequency band to preprocess
    class_dict : dict
        Dictionary of class labels, to get class names
    manifold_ECoG_data : ECoG_Data
        Contains the data class for all participants
        Should already have the current participant's data in sbj_X and sbj_X_test
    save_flg : bool
        If True, save the data to the sbj_sp
        Default is False

    Returns
    -------
        export_data : np.ndarray
            (n_time, n_chans, n_classes) ie. (T, N, C) per TME specifications
            The trial average for each class
    """
    envelope_freq_sbj_X_test, cur_freq_sbj_X_test = extract_analytic_signal(
        freq_bands,
        freq_name,
        exp_params["ecog_srate"],
        manifold_ECoG_data.sbj_X_test,
    )

    trim_data = trim_filtered_data(
        [-1, 1], exp_params["ecog_srate"], cur_freq_sbj_X_test
    )
    split_sbj_eLFO_ECoG_data, cur_classes = split_into_classes(
        class_dict, manifold_ECoG_data.sbj_y_test[:, 0], trim_data
    )

    # can update to downsample trials later if that seems better
    # for now, using the average across trials
    avg_sbj_mvmt_data = get_trial_avg(split_sbj_eLFO_ECoG_data)

    # concat_ECoG_data, trial_dim, sr_dim, chan_dim = concat_trials(
    #     avg_sbj_mvmt_data)

    export_data = np.array(avg_sbj_mvmt_data)

    if save_flg:
        print("Saving Data")
        io.savemat(
            sbj_sp + "avg_" + freq_name +
            "_tme_data.mat", {"mydata": export_data}
        )
    else:
        return export_data


# Bandpass filtering code from
# https://scipy-cookbook.readthedocs.io/items/ButterworthBandpass.html
def butter_bandpass(
    lowcut: float, highcut: float, fs: float, order: int = 4
) -> np.ndarray:
    """
    Butterworth bandpass filter, using scipy.signal.butter
    Parameters
    ----------
    lowcut : float
        Low cutoff frequency
    highcut : float
        High cutoff frequency
    fs : float
        Sampling frequency
    order : int, optional
        Order of the filter, by default 4

    Returns
    -------
    sos: np.ndarray
        Second-order sections representation of the IIR filter
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    sos = butter(order, [low, high], btype="band", output="sos")
    return sos


def butter_bandpass_filter(
    data: np.ndarray, lowcut: float, highcut: float, fs: float, order: int = 4
) -> np.ndarray:
    """
    Apply a Butterworth bandpass filter to input data
    Parameters
    ----------
    data : np.ndarray
        Input data to filter
    lowcut : float
        Low cutoff frequency
    highcut : float
        High cutoff frequency
    fs : float
        Sampling frequency
    order : int, optional
        Order of the filter, by default 4

    Returns
    -------
    y : np.ndarray
        Data bandpass filtered by the give frequency range
    """
    sos = butter_bandpass(lowcut, highcut, fs, order=order)
    y = sosfiltfilt(sos, data, padlen=int(fs / 3))
    return y


def extract_analytic_signal(
    freq_bands: dict, freq_name: str, ecog_srate: float, ecog_data: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract analytic signal from one participants ECoG data, on one specific frequency band
    Parameters
    ----------
    freq_bands : dict
        Dictionary of frequency bands, to get freq names and band values
    freq_name : str
        Name of current frequency band to preprocess
    ecog_srate : float
        Sampling frequency of the input ECoG data
    ecog_data : np.ndarray
        (n_trials, n_channels, n_time)
        Input ECoG data for one participant

    Returns
    -------
    envelope_freq_sbj_X_test : np.ndarray
        (n_trials, n_channels, n_time)
        Envelope of the analytic signal
    cur_freq_sbj_X_test : np.ndarray
        (n_trials, n_channels, n_time)
        Analytic signal of the current frequency band
        Gives better results because it includes negative values
    """
    if freq_name == "pose":
        # if pose, just use the data as is
        cur_freq_sbj_X_test = ecog_data
        analytic = ecog_data
    else:
        cur_freqs = freq_bands[freq_name]
        low_f = cur_freqs[0]
        high_f = cur_freqs[1]
        cur_freq_sbj_X_test = butter_bandpass_filter(
            ecog_data,
            low_f,
            high_f,
            ecog_srate,
            order=4,
        )
        # cur_freq_sbj_X_test = cur_freq_sbj_X_test.T
        assert cur_freq_sbj_X_test.shape == ecog_data.shape

        # then use the hilbert transform to extract the envelope
        analytic = hilbert(cur_freq_sbj_X_test)
        envelope = np.abs(analytic)
        assert analytic.shape == ecog_data.shape
        assert envelope.shape == ecog_data.shape
        # TODO: is there a way to write a test that will essentially do what I was looking for in plot?

    return analytic, cur_freq_sbj_X_test


def trim_filtered_data(tlim: list, srate: float, data: np.ndarray) -> np.ndarray:
    """
    Trim data to a specific time range, to avoid edge artifacts
    Parameters
    ----------
    tlim : list
        [start_time, end_time]
        times to trim to
    srate : float
        Sampling frequency of the input data
    data : np.ndarray
        (n_trials, n_channels, n_time)
        Input data to trim
    Returns
    -------
    trimmed_data : np.ndarray
        (n_trials, n_channels, start_time:end_time)
        Trimmed data
    """
    cur_time_samps = data.shape[2]
    trim_time_samps = (tlim[1] - tlim[0]) * srate
    assert cur_time_samps >= trim_time_samps, "cannot trim to a bigger time"

    samps_to_remove = cur_time_samps - trim_time_samps
    first_samp = int(samps_to_remove / 2)
    end_samp = cur_time_samps - int(samps_to_remove / 2)
    # need to trim from front and end
    trimmed_data = data[:, :, first_samp:end_samp]
    return trimmed_data


def split_into_classes(
    class_dict: dict, sbj_y_test: np.ndarray, data: np.ndarray
) -> Tuple[list, np.ndarray]:
    """
    Split data by classes, so classes can be analyzed separately
    Parameters
    ----------
    class_dict : dict
        Dictionary of classes, to get class names
    sbj_y_test : np.ndarray
        (n_trials, 2)
        Labels for each trial, first column is the class,
        second column is additional metadata (can be adjusted in ECoG_Data)
    data : np.ndarray
        (n_trials, n_channels, n_time)
        Input ecog data for current participant to split by classes
    Returns
    -------
    class_data : list
        List of (n_trials, n_channels, n_time) for each class
    class_labels : np.ndarray
        list of unique classes for current participant
    """
    # Split into the different classes
    class_data = []
    assert type(sbj_y_test) == np.ndarray

    for i, c in enumerate(class_dict):
        class_y_indices = np.where(sbj_y_test == c)[0]
        if len(class_y_indices) > 0:
            class_data.append(data[class_y_indices, ...])
        else:
            # if there are no trials of this class, add an empty array
            class_data.append([])
            continue
        # print("class ", c)
        # print(class_data[i].shape)
        assert class_data[i].shape[0] == len(class_y_indices)
        assert class_data[i].shape[1] == data.shape[1]
        assert class_data[i].shape[2] == data.shape[2]

    cur_classes = np.unique(sbj_y_test)

    return class_data, cur_classes


def get_trial_avg(split_sbj_eLFO_ECoG_data: list) -> list:
    """
    Get average of all trials for each class
    Parameters
    ----------
    split_sbj_eLFO_ECoG_data : list
        List of (n_trials, n_channels, n_time) for each class
    Returns
    -------
    avg_sbj_eLFO_ECoG_data : list
        List of (1, n_channels, n_time) for each class
    """
    avg_sbj_mvmt_data = []
    for m, mvmt_data in enumerate(split_sbj_eLFO_ECoG_data):
        if type(mvmt_data) == list:
            avg_sbj_mvmt_data.append([])
            continue
        rand_mvmt_data = np.expand_dims(mvmt_data.mean(axis=0), axis=0)
        avg_sbj_mvmt_data.append(rand_mvmt_data)

    return avg_sbj_mvmt_data


def concat_trials(split_sbj_eLFO_ECoG_data: list) -> Tuple[list, list, list, int]:
    """
    Concatenate trials for each class
    Parameters
    ----------
    split_sbj_eLFO_ECoG_data : list
        List of (n_trials, n_channels, n_time) for each class
    Returns
    -------
    concat_sbj_eLFO_ECoG_data : list
        List of (n_trials * n_time, n_channels) for each class
    trial_dim : list
        List of n_trials for each class
    sr_dim : list
        List of n_time for each class
        Should be the same for all classes
    chan_dim : int
        Number of channels for each class
        Is the same for all classes
    """
    concat_ECoG_data = []
    trial_dim = []
    sr_dim = []

    for i, c_data in enumerate(split_sbj_eLFO_ECoG_data):
        if len(c_data) == 0:
            concat_ECoG_data.append([])
            trial_dim.append(0)
            sr_dim.append(0)
            continue
        # print(c_data.shape)
        chan_dim = c_data.shape[1]
        trial_dim.append(c_data.shape[0])
        sr_dim.append(c_data.shape[2])
        # (chan, trial, samps) after next line
        c_data_swap = np.swapaxes(c_data, 0, 1)
        concat_data = np.reshape(
            c_data_swap, (chan_dim, trial_dim[i] * sr_dim[i])).T
        # comes out (trial*samps, chan)
        concat_ECoG_data.append(concat_data)
        assert np.all(
            np.all(c_data[0, :, :] == concat_data[0: sr_dim[i], :].T)
        ), "trials no longer match for class " + str(i)

    return concat_ECoG_data, trial_dim, sr_dim, chan_dim


def normalize_data(concat_ECoG_data: list) -> list:
    """
    Normalize data for each channel
    Parameters
    ----------
    concat_ECoG_data : list
        List of (n_trials * n_time, n_channels) for each class
    Returns
    -------
    norm_ECoG_data : list
        List of (n_trials * n_time, n_channels) for each class, normalized
    """
    # really just needs to have visual inspection for data
    std_scaler = StandardScaler()

    norm_concat_ECoG_data = []
    for i, data in enumerate(concat_ECoG_data):
        if len(data) == 0:
            norm_concat_ECoG_data.append([])
            continue
        std_data = std_scaler.fit_transform(data)
        norm_concat_ECoG_data.append(std_data)
        assert norm_concat_ECoG_data[i].shape == std_data.shape

    return norm_concat_ECoG_data


def unroll_concat_data(
    trial_dim: list,
    sr_dim: list,
    chan_dim: int,
    cur_classes: list,
    class_concat_data: list,
) -> list:
    """
    Unroll the concatenated data into list of 3d data: (n_trials, n_channels, n_samples)
    Parameters
    ----------
    trial_dim : list
        List of trial dimensions
    sr_dim : list
        List of sample dimensions
    chan_dim : int
        Number of channels
    cur_classes : list
        List of class labels
    class_concat_data : list
        List of concatenated data for each class
    Returns
    -------
    unrolled_data : list
        List of 3d data for each class: (n_trials, n_channels, n_samples)
    """
    unrolled_data = []
    for class_i, class_n in enumerate(cur_classes):
        class_data = class_concat_data[class_i]
        if len(class_data) == 0:
            unrolled_data.append([])
            continue
        # make it so data is in (n_trials * n_samples, n_channels)
        if class_data.shape[0] == chan_dim:
            class_data = np.swapaxes(class_data, 0, 1)
        reshaped_data = np.reshape(
            class_data,
            (trial_dim[class_i], sr_dim[class_i], chan_dim),
        )
        reshaped_data = np.swapaxes(reshaped_data, 1, 2)
        unrolled_data.append(reshaped_data)

    return unrolled_data


def check_repeated_trials(
    trial_data: np.ndarray,
    cur_classes: list,
    trial_dim: list = None,
    sr_dim: list = None,
    chan_dim: int = None,
) -> bool:
    """
    Check if there are repeated trials in the data
    Returns False if all trials in the data match for any class
    Parameters
    ----------
    trial_data : list of np.ndarray for each class
        either (n_trials, n_channels, n_samples) or (n_trials * n_samples, n_channels)
    cur_classes : list
        List of class labels
    trial_dim : list
        List of trial dimensions, only used if trial_data is 2d
    sr_dim : list
        List of sample dimensions, only used if trial_data is 2d
    chan_dim : int
        Number of channels, only used if trial_data is 2d
    Returns
    -------
    repeated_trials : bool
        True if there are repeated trials, False otherwise
    """
    # need to unroll if the trials are concatenated
    if len(trial_data[0].shape) == 2:
        trial_data = unroll_concat_data(
            trial_dim, sr_dim, chan_dim, cur_classes, trial_data
        )
    trials_diff = True
    for class_i, class_n in enumerate(cur_classes):
        class_data = trial_data[class_i]
        # if there is no data, ignore
        if len(class_data) == 1 or len(class_data) == 0:
            continue
        first_trial = class_data[0, ...]
        # will not cover case of one repeat of a trial, but I don't think I'm worried about that
        if np.all(class_data == first_trial):
            trials_diff = False

    # checking that logic works if we know trials the same
    # same_trial = np.repeat(
    #     np.expand_dims(trial_data[0][0, ...], axis=0), trial_dim[0], axis=0
    # )
    # print(np.all(same_trial == trial_data[0][0,...]))

    return trials_diff


def roi_proj(
    X_in: np.ndarray, sbj_ord: list, nROIs: int, proj_mat_out: np.ndarray
) -> np.ndarray:
    """
    For all participants, project spectral power from
    electrodes to ROI's for common mapping
    Parameters
    ----------
    X_in : np.ndarray
        (n_sbj * n_trials, n_chan, n_time)
        Contains the data for all participants
    sbj_ord : list
        List of indices of participants, for indexing X_in
    nROIs : int
        Number of ROIs to project to
    proj_mat_out : np.ndarray
        (n_sbj, n_chan, n_ROIs)
        Contains the projection matrix for all participants
        Used to convert from electrodes to ROIs

    Returns
    -------
    X_in_proj : np.ndarray
        (n_sbj * n_trials, n_ROIs, n_time)
        Contains the data for all participants, projected to ROIs
    """
    # Project to ROIs using matrix multiply
    X_in_sh = list(X_in.shape)
    X_in_sh[1] = nROIs
    X_in_proj = np.zeros(X_in_sh)
    for s in range(X_in.shape[0]):
        X_in_proj[s, ...] = proj_mat_out[sbj_ord[s], ...] @ X_in[s, ...]
    del X_in

    return X_in_proj


# # ---------------------------------------------#
# ### QUESTION 1 PRINCIPAL ANGLE ALIGNMENT FNS ###
# # ---------------------------------------------#


def calc_class_pca(
    trial_dim: list,
    sr_dim: list,
    chan_dim: int,
    sbj_sp: str,
    freq_name: str,
    norm_concat_ECoG_data: list,
) -> Tuple[list, list]:
    """
    Calculate PCA within the channel dim for each class
    Parameters
    ----------
    trial_dim : list
        List of n_trials for each class
    sr_dim : list
        List of n_time for each class
    chan_dim : int
        Number of channels for each class
    sbj_sp : str
        Path to where subject results are saved
    freq_name : str
        Name of current frequency band
    norm_concat_ECoG_data : list
        List of (n_trials * n_time, n_channels) for each class, normalized for each channel
    Returns
    -------
    class_pca : list
        List of PCA() objects for each class
    reduced_class_ECoG_data : list
        List of (n_trials * n_time, n_channels) for each class, mapped to PCA space
    """
    class_pca = []
    reduced_class_ECoG_data = []
    for i, n_data in enumerate(norm_concat_ECoG_data):
        if len(n_data) == 0:
            class_pca.append([])
            reduced_class_ECoG_data.append([])
            continue
        # print(i)
        class_pca.append(PCA())
        if n_data.shape[1] != chan_dim:
            n_data = n_data.T
        reduced_class_ECoG_data.append(class_pca[i].fit_transform(n_data))

        concat_dim = trial_dim[i] * sr_dim[i]
        assert (
            class_pca[i].components_.shape[0] == chan_dim
        ), "pca did not result in expected dimensions (chan x chan)"
        assert (
            class_pca[i].components_.shape[1] == chan_dim
        ), "pca did not result in expected dimensions (chan x chan)"
        np.testing.assert_almost_equal(
            sum(class_pca[i].explained_variance_ratio_),
            1.0,
            err_msg="pca explained variance did not add to 1",
        )
        assert (
            reduced_class_ECoG_data[i].shape[0] == concat_dim
        ), "pca oriented data has incorrect 1st dim (concatenated trials)"
        assert (
            reduced_class_ECoG_data[i].shape[1] == chan_dim
        ), "pca oriented data has incorrect 1st dim (channels)"

    # print("Save pca info")
    with open(sbj_sp + freq_name + "_class_pca", "wb") as fp:
        pickle.dump(class_pca, fp)
    np.save(
        sbj_sp + freq_name +
        "_pca_reduced_data.npy", np.array(reduced_class_ECoG_data)
    )

    return class_pca, reduced_class_ECoG_data


def extract_explained_var(
    class_dict: dict, freq_bands: dict, pats_ids_in: list, all_sbjs_pca: list
) -> np.ndarray:
    """
    Extract explained variance for each participant, frequency and class
    Parameters
    ----------
    class_dict : dict
        Dictionary of class names
    freq_bands : dict
        Dictionary of frequency bands
    pats_ids_in : list
        List of patient IDs
    all_sbjs_pca : list
        List of PCA() objects. Now the exact shape of the list does not matter
    Returns
    -------
    pca_manifolds_VAF : np.ndarray
        (n_freqs, n_sbjs, n_mvmts, n_components)
        Array of explained variance for each participant, frequency and class and day if included
    """
    num_components = calc_min_dim(
        freq_bands, pats_ids_in, class_dict, all_sbjs_pca)
    pca_shape = list(all_sbjs_pca.shape)
    pca_shape.append(num_components)
    pca_manifolds_VAF = np.zeros(pca_shape)

    # runs into issues with Sat's data because inconsistent num of classes
    # I believe this should be fixed now
    arr_it = np.nditer(all_sbjs_pca, ["multi_index", "refs_ok"])
    for x in arr_it:
        if len(arr_it.multi_index) == 4:
            f, i, d, m = arr_it.multi_index
        else:
            f, i, m = arr_it.multi_index
        cur_pca = all_sbjs_pca[arr_it.multi_index]

        if type(cur_pca) == list:
            for v in range(num_components):
                if len(arr_it.multi_index) == 4:
                    pca_manifolds_VAF[f, i, d, m, v] = float("nan")
                else:
                    pca_manifolds_VAF[f, i, m, v] = float("nan")
            continue

        for v in range(num_components):
            if len(arr_it.multi_index) == 4:
                pca_manifolds_VAF[f, i, d, m,
                                  v] = cur_pca.explained_variance_ratio_[v]
            elif len(arr_it.multi_index) == 3:
                pca_manifolds_VAF[f, i, m,
                                  v] = cur_pca.explained_variance_ratio_[v]

    return pca_manifolds_VAF


def get_pa_per_pat(
    first_dim_lst: list,
    second_dim_lst: list,
    comparing_dim: list,
    dim_red_lst: list,
    all_sbjs_pca: list,
    include_same_same: bool = False,
) -> Tuple[np.ndarray, list]:
    """
    Calculate principal angles for each
    pairwise comparison of each participant and frequency
    Parameters
    ----------
    first_dim_lst : list
        The list of elements in the first dimension of the pca data
        Typically is freq_bands, as a dictionary of frequency bands
    second_dim_lst : list
        The list of elements in the second dimension of the pca data
        Typically is pats_ids_in, as a list of patient IDs
    comparing_dim : list
        The list of elements in the comparing dimension of the pca data
        Typically is class_dict, as a dictionary of class names
    dim_red_lst : list
        List of dimensions to reduce to
        Must be the same length as the first dimension
        Typically is for each frequnecy
        Based on 80% variance explained
    all_sbjs_pca : list
        List of PCA() objects for each frequency, participant and class
        In shape (first_dim_lst, second_dim_lst, comparing_dim)
        Typically (freq, sbj, class)
    Returns
    -------
    grass_dist : np.ndarray
        (n_freq, n_sbjs, n_compare)
        Array of grassmannian distance for each pairwise comparison
    pa_by_freq : list
        List of (n_sbjs, n_compare_above, freq_red_dim[f]) for each frequency
        So overall shape (n_freq, n_sbjs, n_compare_above, freq_red_dim[f])
        Contains the principal angles between manifolds for each pairwise comparison
    """
    n_compare, n_compare_above = get_num_pa_comparisons(comparing_dim)
    num_classes = len(comparing_dim)
    if include_same_same:
        n_compare_above = n_compare
    grass_dist = np.empty((len(first_dim_lst), len(second_dim_lst), n_compare))
    grass_dist[:] = np.nan
    pa_by_freq = [[] for f in first_dim_lst]

    for f, f_dim in enumerate(first_dim_lst):
        principal_angles = np.empty(
            (len(second_dim_lst), n_compare_above, dim_red_lst[f])
        )
        principal_angles[:] = np.nan
        for s, s_dim in enumerate(second_dim_lst):
            sbj_freq_pca_manifolds = all_sbjs_pca[f][s]
            principal_angles[s, ...] = calc_comp_dim_pas(
                comparing_dim, sbj_freq_pca_manifolds, dim_red_lst[f], include_same_same)

        pa_by_freq[f] = principal_angles

    return grass_dist, pa_by_freq


def calc_comp_dim_pas(comp_dim_lst, pca_manifolds, red_dim, include_same_same=False):
    n_compare, n_compare_above = get_num_pa_comparisons(comp_dim_lst)
    if include_same_same:
        n_compare_above = n_compare
    n_comp_dim = len(comp_dim_lst)
    m_ind = 0
    upper_diag_ind = 0
    principal_angles = np.empty((n_compare_above, red_dim))
    principal_angles[:] = np.nan
    for m_a in range(n_comp_dim):
        for m_b in range(m_a, n_comp_dim):
            # means one of the comparisons doesn't have data
            if (
                type(pca_manifolds[m_a]) == list
                or type(pca_manifolds[m_b]) == list
            ):
                # will leave behind nan if no data there
                m_ind += 1
                if not include_same_same:
                    if m_a != m_b:
                        upper_diag_ind += 1
                continue
            # print(pca_manifolds[f][s][m_a].components_.shape)
            if (
                pca_manifolds[m_a] == pca_manifolds[m_b]
                and m_a != m_b
            ):
                print("WE YU WE YU")
                print("the PCA objects are the same")
                print("the class inds are: ", m_a, m_b)
                raise ValueError

            a_components = pca_manifolds[m_a].components_[
                0: red_dim, :
            ]
            b_components = pca_manifolds[m_b].components_[
                0: red_dim, :
            ]
            theta_vals, cur_dist = calc_principal_angles(
                a_components, b_components
            )

            if not include_same_same:
                if m_a != m_b:
                    principal_angles[upper_diag_ind, ...] = np.degrees(
                        theta_vals
                    )
                    upper_diag_ind += 1
            else:
                principal_angles[m_ind, ...] = np.degrees(
                    theta_vals)
            m_ind += 1

    return principal_angles


def get_summed_pas_df(
    freq_bands: dict,
    pats_ids_in: list,
    comparison_list: list,
    pa_by_freq: list,
    red_dim: int,
    null_data_sbjs_freqs: list = None,
) -> pd.DataFrame:
    """
    Get the summed pas as a dataframe

    Parameters
    ----------
    freq_bands : dict
        Dictionary of frequency bands
    pats_ids_in : list
        List of patient IDs
    comparison_list : list
        List of values in the comparison dimension
    pa_by_freq : list
        List of (n_sbjs, n_compare_above, freq_red_dim[f]) for each frequency
        So overall shape (n_freq, n_sbjs, n_compare_above, freq_red_dim[f])
        Contains the principal angles between manifolds for each pairwise comparison
    null_data_sbjs_freqs : list - Default none
        List of (n_sbjs, n_null_samps, n_compare_above, freq_red_dim[f]) for each frequency
        So overall shape (n_freq, n_sbjs, n_compare_above, freq_red_dim[f])
        Contains the principal angles between manifolds for each pairwise comparison
        for the null data

    Returns
    -------
    summed_pas_df : pd.DataFrame
        Dataframe of summed pas
    """
    # real_summed_pas, null_summed_pas = get_summed_pas_real_null(
    #     pa_by_freq, null_data_sbjs_freqs
    # )
    real_summed_pas = calc_norm_sum_pa(red_dim, pa_by_freq)
    if null_data_sbjs_freqs is not None:
        null_summed_pas = calc_norm_sum_pa(red_dim, null_data_sbjs_freqs)
        null_summed_pas = np.squeeze(null_summed_pas)
        if len(freq_bands) == 1:
            null_summed_pas = np.expand_dims(null_summed_pas, axis=0)

    # now put the summed PA data for both real and null data into a dataframe
    # the dataframe will have the following columns:
    # 1) frequency band
    # 2) subject id
    # 3) Movement Comparison
    # 4) Summed PA
    summed_pas = []
    comp_names = get_pa_comparison_names(comparison_list)
    # for real data

    for f, cur_freq in enumerate(freq_bands):
        for s, cur_sbj in enumerate(pats_ids_in):
            for c, cur_comp in enumerate(comp_names):
                summed_pas.append(
                    [cur_freq, cur_sbj, comp_names[c], real_summed_pas[f][s][c]]
                )

    # for null data
    if null_data_sbjs_freqs is not None:
        for f, cur_freq in enumerate(freq_bands):
            for s, cur_sbj in enumerate(pats_ids_in):
                for i, samp in enumerate(null_summed_pas[f][s]):
                    for c, cur_comp in enumerate(comp_names):
                        summed_pas.append(
                            ["Null", "Null", comp_names[c],
                                null_summed_pas[f][s][i][c]]
                        )

    # now make into dataframe
    summed_pas_df = pd.DataFrame(
        summed_pas,
        columns=[
            "Frequency",
            "Participant",
            "Movement Comparison",
            "Neural Dissimilarity",
        ],
    )
    return summed_pas_df


def one_freq_get_summed_pas_df(pa, red_dim, cur_freq, pats_ids_in, days_tested, comparison_list, null_pa=None):
    real_summed_pas = calc_norm_sum_pa(red_dim, pa)
    # now put the summed PA data for both real and null data into a dataframe
    # the dataframe will have the following columns:
    # 1) frequency band
    # 2) subject id
    # 3) Day
    # 4) Movement Comparison
    # 5) Summed PA (Neural dissim)
    summed_pas = []
    comp_names = get_pa_comparison_names(comparison_list)
    # for real data
    for s, cur_sbj in enumerate(pats_ids_in):
        for d, day in enumerate(days_tested):
            for c, cur_comp in enumerate(comp_names):
                summed_pas.append(
                    [cur_freq, cur_sbj, day, comp_names[c], real_summed_pas[s][d][c]]
                )

    # for null data
    if null_pa is not None:
        null_summed_pas = calc_norm_sum_pa(red_dim, null_pa)
        null_summed_pas = np.squeeze(null_summed_pas)
        for s, cur_sbj in enumerate(pats_ids_in):
            for i, samp in enumerate(null_summed_pas[s]):
                for c, cur_comp in enumerate(comp_names):
                    summed_pas.append(
                        [cur_freq, "Null", "Null", comp_names[c],
                            null_summed_pas[s][i][c]]
                    )

    # now make into dataframe
    summed_pas_df = pd.DataFrame(
        summed_pas,
        columns=[
            "Frequency",
            "Participant",
            "Day",
            "Movement Comparison",
            "Neural Dissimilarity",
        ],
    )
    return summed_pas_df


def calc_norm_sum_pa(red_dim: int, data_pas: list) -> np.ndarray:
    """
    Calculate the sum of the principal angles for each sbj and freq
    Parameters
    ----------
    red_dim : int
        Dimensionality of the reduced data
    data_pas : list
        List of principal angles for the data
        Assumes that the last dimension contains the principal angles
        Also assumes that the pairwise comparisons have already been done
    Returns
    -------
    summed_pas : np.array
        (num_sbjs, num_freqs, *maybe num_null_samples*, 1)
    """
    data_pas = data_pas[..., 0:red_dim]
    summed_pas = calc_sum_pa(data_pas)

    # not sure how well this will work across diff freqs with different red_dims
    max_pas = np.zeros((red_dim)) + 90
    normalized_max = np.sum(max_pas)
    # print(normalized_max)
    return summed_pas / normalized_max


def choose_one_freq_dimensionality(
    class_dict: dict,
    freq_band: str,
    pats_ids_in: list,
    all_sbjs_pca: list,
    percent_threshold: float = 0.8,
) -> int:
    """
    Choose the dimensionality of the PCA data based on the variance accounted for by each component
    Asssumes that I want 80% VAF
    Just does it for one frequency
    Assumes that the PCA data comes in as (sbj, days, classes)
    Parameters
    ----------
    class_dict : dict
        Dictionary of the classes
    freq_bands : dict
        Dictionary of the frequency bands
    pats_ids_in : list
        List of the patient ids
    all_sbjs_pca : list
        List of the PCA objects for each freq and sbj
    percent_threshold : float, optional
        Percent of variance explained to threshold at, by default 0.8
    Returns
    -------
    freq_red_dim : int
        the 80% VAF dimensionality for the given frequency
    """

    pca_manifolds_VAF = extract_explained_var(
        class_dict, freq_band, pats_ids_in, all_sbjs_pca
    )
    # pca_manifolds_VAF shape of (1, num_sbj, num_days, num_mvmts, num_components)

    # get average for the frequency
    avg_components = np.squeeze(np.nanmean(pca_manifolds_VAF, axis=(3, 2, 1)))

    # now find the dim
    cum_VAF = 0
    for j, cur_VAF in enumerate(avg_components):
        cum_VAF += cur_VAF
        if cum_VAF >= percent_threshold:
            freq_red_dim = j + 1
            return freq_red_dim

    raise ValueError("No cutoff dimensionality found")


# # HELPER FUNCTIONS FOR PRINCIPAL ANGLES ANALYSIS #
def calc_principal_angles(
    a_components: np.ndarray, b_components: np.ndarray
) -> Tuple[np.ndarray, int]:
    """
    Calculate the principal angles between two sets of components
    Parameters
    ----------
    a_components : np.ndarray
        (num_components, num_features)
    b_components : np.ndarray
        (num_components, num_features)
    Returns
    -------
    theta_vals : np.ndarray
        (num_components, 1)
    cur_dist : int
        The distance between the two sets of components
    """
    # assumes that components come in with shape (red_dim x n_fts)
    a_components = a_components.T
    b_components = b_components.T

    orth_combined = np.dot(a_components.T, b_components)
    U, Sigma, VT = np.linalg.svd(orth_combined)
    Sigma = np.clip(Sigma, -1, 1)
    theta_vals = np.arccos(Sigma)
    # print("Theta values, in degrees")
    # print(np.degrees(theta_vals))
    cur_dist = np.sum(np.square(theta_vals)) ** (1 / 2)

    return theta_vals, cur_dist


def calc_sum_pa(data_pas: list) -> np.ndarray:
    """
    Calculate the sum of the principal angles for each sbj and freq
    Parameters
    ----------
    data_pas : list
        List of principal angles for each sbj and freq
        Assumes that the data is coming in as a list of frequencies, then sbj
        Also assumes that the pairwise comparisons have already been done
    Returns
    -------
    summed_pas : np.array
        (num_sbjs, num_freqs, *maybe num_null_samples*, num_comparisons)
    """

    return np.array(data_pas).sum(axis=-1)


def calc_min_dim(
    freq_bands: dict, pats_ids_in: list, class_dict: dict, all_sbjs_pca: list
) -> int:
    """
    Calculate minimum number of components shared across participants
    Mostly useful if participant data hasn't been projected to the same rois
    Parameters
    ----------
    freq_bands : dict
        Dictionary of frequency bands
    pats_ids_in : list
        List of patient IDs
    class_dict : dict
        Dictionary of class names
    all_sbjs_pca : list
        List of PCA() objects for each frequency, participant and class
        shape (freq, sbj, class)
    Returns
    -------
    num_components : int
        Minimum number of components shared across participants
    """
    min_dim = float("inf")
    arr_it = np.nditer(all_sbjs_pca, flags=["multi_index", "refs_ok"])
    for x in arr_it:
        cur_pca = all_sbjs_pca[arr_it.multi_index]
        if type(cur_pca) == list:
            continue
        cur_dim = len(cur_pca.explained_variance_ratio_)
        if cur_dim < min_dim:
            min_dim = cur_dim

    return min_dim


def get_num_pa_comparisons(comparing_dim: dict) -> Tuple[int, int]:
    """
    Calculate number of pairwise comparisons based on comparing_dim
    Assumes that the comparing dim is provided as a list or dict
    Parameters
    ----------
    comparing_dim : dict
        Dictionary or list for things to compare
    Returns
    -------
    n_compare : int
        Number of pairwise comparisons, including self-comparisons
    n_compare_above : int
        Number of pairwise comparisons, excluding self-comparisons (ie. just the upper triangle)
    """

    num_dim = len(comparing_dim)
    n_compare = 0
    for i in range(0, num_dim):
        n_compare += num_dim - i
    n_compare_above = n_compare - num_dim

    return n_compare, n_compare_above


def get_pa_comparison_names(
    comparing_dim: dict, include_same_same: bool = False
) -> dict:
    """
    Calculate names of pairwise comparisons based on comparing_dim
    ie. left vs right, left vs left, right vs right, etc.
    Assumes that the comparing dim is provided as a list or dict
    Parameters
    ----------
    comparing_dim : dict
        Dictionary or list for things to compare
    Returns
    -------
    pa_comparison_names : dict
        Dictionary of the names of pairwise comparisons, excluding self-comparisons
    """

    # if dict, starts at 0
    num_dim = len(comparing_dim)
    vs_dict = {}
    compare_i = 0
    for i in range(num_dim):
        for j in range(i, num_dim):
            if i == j and not include_same_same:
                continue
            vs_dict[compare_i] = str(comparing_dim[i]) + \
                " vs " + str(comparing_dim[j])
            compare_i += 1

    return vs_dict


def calc_sumed_significance(
    summed_pas_df: pd.DataFrame, null_col: str = "Participant"
) -> float:
    """
    Calculate the significance value of the summed principal angles
    Parameters
    ----------
    summed_pas_df : pd.DataFrame
        Dataframe of summed principal angles
    Returns
    -------
    summed_significance : float
        Significance value threshold of the summed principal angles
    """
    # get the value out for 0.01 of null
    null_vals = summed_pas_df[summed_pas_df[null_col] == "Null"]
    null_vals = null_vals["Neural Dissimilarity"]
    null_vals = null_vals.values
    null_vals = null_vals[~np.isnan(null_vals)]
    sigf_val = np.percentile(null_vals, 1, axis=0)
    return sigf_val


# # -----------------------#
# ###    NULL DATA FNS   ###
# # -----------------------#

def load_null_data(null_data_sp: str, freq_name: str) -> np.ndarray:
    """
    Get null data for a given frequency and participant
    Parameters
    ----------
    null_data_sp : str
        Path to null data
    freq_name : str
        Frequency name
    Returns
    -------
    null_data : np.ndarray
        Null data for the given frequency and participant
        Should be of shape (num_null_samples, time, chans, conditions)
    """
    # freq_name = freq_name.replace(" ", "_")
    # null_data_mat = mat73.loadmat(null_data_sp + "null_data_" + freq_name + ".mat")
    # null_data = null_data_mat["all_surr_data"]
    null_data = np.load(null_data_sp + "TME_null_" + freq_name + ".npy")

    return null_data


# use if you need to load in the null data
def get_null_data_sbjs_freqs(
    null_data_sp: str,
    proj_mat_sp: str,
    class_dict: dict,
    freq_bands: dict,
    pats_ids_in: list,
    freq_red_dim: list,
) -> list:
    """
    Get null data for each sbj and freq
    Use if you need to load in the data
    Will calculate the principal angles for each sbj and freq
    Parameters
    ----------
    null_data_sp : str
        Path to null data
    proj_mat_sp : str
        Path to where the pca info will get saved
    class_dict : dict
        Dictionary of class names
    freq_bands : dict
        Dictionary of frequency bands
    pats_ids_in : list
        List of participant ids
    freq_red_dim : list
        List of the number of components to reduce to for each freq
    Returns
    -------
    null_data_sbjs_freqs : list
        List of 1000 samples of null data for each sbj and freq
        Should be shape (num_freqs, num_sbjs, num_null_samples, time, conditions, chans)
    """
    num_classes = len(class_dict)
    null_data_sbjs_freqs = []
    for f, freq_name in enumerate(freq_bands):
        print("Getting null data for freq: " + str(freq_name))
        null_data_sbjs_freqs.append([])
        for s, pat_id_curr in enumerate(pats_ids_in):
            print("Getting null data for sbj: " + str(pat_id_curr))
            sbj_sp = proj_mat_sp + str(pat_id_curr) + "/"
            cur_null_sp = null_data_sp + str(pat_id_curr) + "/"
            null_data = load_null_data(cur_null_sp, freq_name)
            # reshape null data to match expected shape
            # (num_samples, sr_dim, num_classes, chans)
            if null_data.shape[2] != num_classes:
                null_data = np.swapaxes(null_data, 2, 3)
            # null_data_sbjs_freqs[f].append(null_data)
            null_data_sbjs_freqs[f].append(
                get_null_data_pa(
                    null_data,
                    sbj_sp,
                    class_dict,
                    cur_dim=freq_red_dim[f],
                )
            )

    return null_data_sbjs_freqs


def get_null_data_pa(
    null_data: list, sbj_sp: str, condition_lst: list, cur_dim: int
) -> np.ndarray:
    """
    Calculate principal angles of null data for a given frequency and participant
    This will tell how much the null data is different from the real data
    Parameters
    ----------
    null_data : list
        List of 1000 samples of null data for current sbj, freq condition
        Should be shape (num_null_samples, time, conds, channels)
    sbj_sp : str
        Path to subject data savepoint
    condition_lst : list
        List of items in the condition dim
        Typically, class_dict
    cur_dim : int
        Dimension to reduce to, based on the 80% variance explained of real data
    Returns
    -------
    null_pa : np.ndarray
        (n_samples, n_compare_above, cur_dim)
        Array of principal angles for each pairwise comparison of the null data
    """
    # check that sbj_sp exists
    if not os.path.exists(sbj_sp):
        os.makedirs(sbj_sp)

    assert (
        len(condition_lst) == null_data.shape[2]
    ), "Condition list must match null data"
    null_pa = []
    # for all of the samples in the null data
    for null_t in tqdm(range(null_data.shape[0])):
        cur_null_data = null_data[null_t, ...]
        cur_null_data = np.swapaxes(cur_null_data, 0, 1)
        cur_null_data = np.swapaxes(cur_null_data, 1, 2)
        # reshape into (conds, chans, time)
        # print(cur_surr_data.shape)

        # create list of all the conditions
        null_data_list = []
        for m in range(cur_null_data.shape[0]):
            null_norm = normalize_data(cur_null_data[m, ...].T)
            null_data_list.append(null_norm)

        trial_dim = [1 for i in range(null_data.shape[2])]
        sr_dim = [null_data.shape[1] for i in range(null_data.shape[2])]
        chan_dim = null_data.shape[3]
        class_pca, reduced_surr_data = calc_class_pca(
            trial_dim, sr_dim, chan_dim, sbj_sp, "null_distrib", null_norm
        )

        freq_red_dim = cur_dim

        grass_dist, pa_by_freq = get_pa_per_pat(
            ["Null"], ["Null"], condition_lst, [freq_red_dim], [[class_pca]]
        )
        null_pa.append(pa_by_freq)

    return np.array(null_pa)


# # --------------------------#
# ###     Cross-Pat FNS    ###
# # --------------------------#

def calc_elec_overlap(
    exp_params: dict,
    comparison: str = "subjects",
    custom_rois: bool = False,
) -> float:
    """
    Calculate the overlap between electrodes of different participants
    Or of participant overlap with motor areas
    Depends on the comparison parameter
    Parameters
    ----------
    exp_params : dict
        Dictionary of experiment parameters
    comparison : str
        Type of comparison to make
        Either "subjects" or "motor"
    custom_rois : bool
        Whether to use custom ROIs
    Returns
    -------
    overlap : float
        Percentage overlap between electrodes
    """
    dipole_dens_thresh = exp_params["dipole_dens_thresh"]
    dipole_dens = proj_compute_dipdens(
        exp_params["pats_ids_in"], exp_params["ecog_roi_proj_lp"]
    )
    sbj_comparisons = 0
    sbjs_numerator = []
    sbjs_denominator = []
    num_sbjs = len(exp_params["pats_ids_in"])

    # make the subject arrays for comparing
    if comparison == "subjects":
        for m_a in range(num_sbjs):
            for m_b in range(m_a, num_sbjs):
                sbjs_numerator.append([m_a])
                sbjs_denominator.append([m_b])
                sbj_comparisons += 1

    elif comparison == "motor":
        for sbj in range(num_sbjs):
            sbj_comparisons += 1
            sbjs_numerator.append([sbj])
            sbjs_denominator.append([sbj])

    sbjs_numerator = np.array(sbjs_numerator)
    sbjs_denominator = np.array(sbjs_denominator)
    # print(sbjs_numerator.shape)
    # print(sbjs_numerator)
    # print(sbjs_denominator)

    # now can actually run the calculation
    frac_overlap = []
    for i in range(sbj_comparisons):
        mean_dips1 = dipole_dens[sbjs_numerator[i, :], :].mean(axis=0)
        inds_thresh1 = np.nonzero(mean_dips1 >= dipole_dens_thresh)[0]

        if comparison == "subjects":
            mean_dips2 = dipole_dens[sbjs_denominator[i, :], :].mean(axis=0)
            inds_thresh2 = np.nonzero(mean_dips2 >= dipole_dens_thresh)[0]

        elif comparison == "motor":
            custom_roi_inds_compare = get_custom_motor_rois()
            inds_thresh2 = custom_roi_inds_compare.copy()

        if custom_rois:
            custom_roi_inds = get_custom_motor_rois()
            frac_num = len(
                reduce(np.intersect1d, (inds_thresh1,
                       inds_thresh2, custom_roi_inds))
            )
            frac_denom = len(
                reduce(np.intersect1d, (inds_thresh2, custom_roi_inds)))
        else:
            frac_num = len(
                reduce(np.intersect1d, (inds_thresh1, inds_thresh2)))
            frac_denom = len(inds_thresh2)

        frac_overlap.append(frac_num / frac_denom)
    return frac_overlap


def proj_compute_dipdens(
    patient_ids: list, roi_proj_loadpath: str, atlas: str = "none"
) -> np.ndarray:
    """
    Loads projection matrix for each subject and extracts dipole densities (top row)

    Inputs:
            patient_ids : which participants to get projection matrix from
            roi_proj_loadpath : where to load projection matrix CSV files
            atlas : ROI projection atlas to use (aal, loni, brodmann, or none)
    Outputs:
            dipole_densities : dipole densities for each subject
    """
    # Find good ROIs first
    dipole_densities = []
    for s, patient in enumerate(patient_ids):
        df = pd.read_csv(roi_proj_loadpath + atlas +
                         "_" + patient + "_elecs2ROI.csv")
        dip_dens = df.iloc[0]
        dipole_densities.append(dip_dens)

    return np.asarray(dipole_densities)


def get_custom_motor_rois(
    regions: list = ["precentral", "postcentral", "parietal_inf"]
) -> list:
    """
    Returns ROI indices for those within the precentral, postcentral, and inferior parietal regions (accoring to AAL2)
    """
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

    #     custom_roi_inds = np.union1d(np.union1d(precentral_inds,postcentral_inds),parietal_inf_inds) #select for sensorimotor ROIs
    custom_roi_inds = []
    for val in regions:
        eval("custom_roi_inds.extend(" + val + "_inds)")
    return custom_roi_inds


# # -------------------------------#
# ### VISUALIZATIONS AND FIGURES ###
# # -------------------------------#


def get_all_colors():
    # color dicts for all slices of data
    # Q1 - Movements
    movement_colors = {
        "rest": "black",
        "Null": "grey",
        "up": "#e5876f",
        "down": "#8a99b9",
        "left": "#d587b6",
        "right": "#98c364",
    }

    # Q3 - Days
    dc = sns.color_palette("Set2", n_colors=5)
    day_colors = {
        "Null": "grey",
        "Day 1": dc[0],
        "Day 2": dc[1],
        "Day 3": dc[2],
        "Day 4": dc[3],
        "Day 5": dc[4],
    }

    # Q4 - Participants
    participant_colors = {
        "Null": "grey",
        "P01": "#f46284",
        "P02": "#e68a55",
        "P03": "#c5aa64",
        "P04": "#a4b05d",
        "P05": "#5eb64e",
        "P06": "#3eaf84",
        "P07": "#4eb4a8",
        "P08": "#6cbdca",
        "P09": "#5eafe9",
        "P10": "#9a7fec",
        "P11": "#e558ec",
        "P12": "#f67ec8",
    }

    # Q5 - Experimental Movements
    cc = sns.color_palette("pastel", n_colors=5)
    exp_class_colors = {
        "Null": "grey",
        "Thumb": cc[0],
        "Index": cc[1],
        "Middle": cc[2],
        "Ring": cc[3],
        "Pinky": cc[4],
    }
    pc = sns.color_palette("rainbow", n_colors=5)
    exp_participant_colors = {
        "Null": "grey",
        "E01": pc[0],
        "E02": pc[1],
        "E03": pc[2],
        "E04": pc[3],
        "E05": pc[4]
    }

    # frequency colors
    fc = sns.color_palette("Set2", n_colors=6)
    freq_colors = {
        "Null": "grey",
        "LFO": fc[0],
        "Alpha": fc[1],
        "Beta": fc[2],
        "Low Gamma": fc[3],
        "Gamma": fc[4],
        "High Gamma": fc[5],
    }

    return (
        movement_colors,
        day_colors,
        participant_colors,
        exp_class_colors,
        exp_participant_colors,
        freq_colors,
    )


# # --------------------------#
# ### PREPROCESSING FIGURES ###
# # --------------------------#


def plot_raw_and_freq(
    freq_name: str,
    sbj_sp: str,
    manifold_ECoG_data: ECoG_Data,
    cur_freq_sbj_X_test: np.ndarray,
):
    """
    Plots raw ECoG data and frequency data for a given subject and frequency band
    This is to verify that the data is being processed correctly during the bandpass
    filtering step
    Parameters
    ----------
    freq_name : str
        Name of frequency band
    sbj_sp : str
        Where to save the figure
    manifold_ECoG_data : ECoG_Data
        ECoG_Data object
    cur_freq_sbj_X_test : np.ndarray
        Frequency data for a given subject and frequency band
    Returns
    -------
    None.
    Plots and saves in sbj directory
    """
    plt.plot(
        manifold_ECoG_data.sbj_X_test[0, 0, :], label="Unfiltered", c="purple")
    plt.plot(cur_freq_sbj_X_test[0, 0, :],
             label=freq_name + " filter", c="green")
    plt.legend()
    plt.title("Unfiltered and " + freq_name + " filtered data")
    plt.savefig(sbj_sp + "unfilt_and_" + freq_name + "_filt_data.png")
    plt.clf()


def plot_after_norm(
    sbj_sp: str, freq_name: str, concat_ECoG_data: list, norm_concat_ECoG_data: list
):
    """
    Plots the filtered ECoG data for first trial after concatenation and normalization
    This is to make sure that normalizing the data looks good
    Parameters
    ----------
    sbj_sp : str
        Where to save the figure
    freq_name : str
        Name of frequency band
    concat_ECoG_data : list
        List of concatenated ecog data, assumed in (n_trials * n_time, n_channels) shape
    norm_concat_ECoG_data : list
        List of normalized concatenated ecog data, assumed in (n_trials * n_time, n_channels) shape
    Returns
    -------
    None.
    Plots and saves in sbj directory
    """
    plt.plot(concat_ECoG_data[0][0:500, 0:3], c="purple", label="Original")
    plt.plot(norm_concat_ECoG_data[0][0:500, 0:3],
             c="green", label="Normalized")
    plt.legend()
    plt.title("Data Before and After Normalization")
    plt.savefig(sbj_sp + freq_name + "_before_after_norm.png")
    plt.clf()


# # ---------------#
# ### Q1 FIGURES ###
# # ---------------#

def make_pca_plots(
    sbj_sp,
    i,
    freq_name,
    class_dict,
    cur_classes,
    class_color,
    trial_dim,
    sr_dim,
    roi_centroids,
    class_pca,
    reduced_class_ECoG_data,
):
    """
    Plot all the combos of PCA plots for the current PCA data:
    explained variance, average trial manifold trajectory,
    all trial manifold trajectories, channel contributions

    Args:
        sbj_sp (str): where to save the figures
        i (int): which subject to plot channel contributions
        freq_name (str): name of frequency band to plot
        class_dict (dict): dictionary of classes
        cur_classes (list): list of current classes to plot
        class_color (dict): dictionary of class colors for the plots
        trial_dim (list): list of trial dimensions of the PCA data
        sr_dim (list): list of time length dimensions of the PCA data
        roi_centroids (list): list of ROI centroids, if not None
        class_pca (list): list of PCA data for each class
        reduced_class_ECoG_data (list): List of (n_trials * n_time, n_channels)
            for each class, mapped to PCA space
    """
    if reduced_class_ECoG_data != []:
        plot_explained_variance(
            class_dict, cur_classes, sbj_sp, freq_name, class_color, class_pca
        )
        # pdb.set_trace()
        plot_manifold_trajectories(
            class_dict,
            cur_classes,
            sbj_sp,
            freq_name,
            class_color,
            trial_dim,
            sr_dim,
            reduced_class_ECoG_data,
        )
        plot_all_manifold_trials(
            class_dict,
            cur_classes,
            class_color,
            sbj_sp,
            freq_name,
            reduced_class_ECoG_data,
        )
        plot_chan_contribs(
            i,
            class_dict,
            cur_classes,
            sbj_sp,
            freq_name,
            class_pca,
            roi_centroids,
        )


def plot_explained_variance(
    class_dict, cur_classes, sbj_sp, freq_name, class_color, class_pca
):
    """
    Plot the cumulative explained variance for the current pca data

    Args:
        class_dict (dict): dictionary mapping class numbers to class names
        cur_classes (list): list of classes to plot
        sbj_sp (str): where to save the figure
        freq_name (str): name of the current frequency band for plot
        class_color (dict): dictionary mapping class names to colors for the plot
        class_pca (list of PCA objects): list of PCA objects for each class,
            or whatever comparison you want to make for PCA
    """
    for class_i, class_n in enumerate(cur_classes):
        c_pca = class_pca[class_i]
        if type(c_pca) == list:
            continue
        cum_VAF = 0
        for j, cur_VAF in enumerate(c_pca.explained_variance_ratio_):
            cum_VAF += cur_VAF
            plt.scatter(j + 1, cum_VAF, marker="+",
                        color=class_color[class_n][0])

        plt.xlabel("Number of PCs")
        plt.ylabel("Neural Variance Accounted For (%)")
        plt.ylim(0, 1.05)
        plt.title(class_dict[class_n] + " Data Variance Accounted For")
        plt.savefig(
            sbj_sp + freq_name + "_" +
            class_dict[class_n] + "_manifold_VAF.png"
        )
        plt.clf()


def plot_manifold_trajectories(
    class_dict,
    cur_classes,
    sbj_sp,
    freq_name,
    class_color,
    trial_dim,
    sr_dim,
    reduced_class_ECoG_data,
):
    """
    Plot the average trajectory of the current pca data

    Args:
        class_dict (dict): dictionary mapping class numbers to class names
        cur_classes (list): list of classes to plot
        sbj_sp (str): where to save the figure
        freq_name (str): name of the current frequency band for plot
        class_color (dict): dictionary mapping class names to colors for the plot
        trial_dim (list): list of the trial dimensions for each class
        sr_dim (list): list of the time dimension for each class
        reduced_class_ECoG_data (list): List of (n_trials * n_time, n_channels)
            for each class, mapped to PCA space
    """
    plot_components = 3
    unroll_data = unroll_concat_data(
        trial_dim,
        sr_dim,
        reduced_class_ECoG_data[0].shape[1],
        cur_classes,
        reduced_class_ECoG_data,
    )

    for class_i, class_n in enumerate(cur_classes):
        print(class_dict[class_n] + " data average trajectory calcs")
        if len(unroll_data[class_i]) == 0:
            continue
        # TODO: add asserts here
        # print(reduced_data.shape)
        # reshape and then calc mean
        # avg_reduced_ECoG_data = np.reshape(
        #     reduced_data[:, 0:plot_components],
        #     (trial_dim[class_i], sr_dim[class_i], plot_components),
        # )
        # print(avg_reduced_ECoG_data.shape)
        avg_reduced_ECoG_data = unroll_data[class_i][:, 0:plot_components, :].mean(
            axis=0
        )
        avg_reduced_ECoG_data = avg_reduced_ECoG_data.T
        # print(avg_reduced_ECoG_data.shape)

        plt.scatter(
            avg_reduced_ECoG_data[:, 0],
            avg_reduced_ECoG_data[:, 1],
            c=class_color[class_n][0],
        )
        plt.plot(
            avg_reduced_ECoG_data[:, 0],
            avg_reduced_ECoG_data[:, 1],
            c=class_color[class_n][0],
        )
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.title("First 2 PCs for " + class_dict[class_n])
        plt.savefig(
            sbj_sp
            + freq_name
            + "_"
            + class_dict[class_n]
            + "_2d_manifold_trajectory.png"
        )
        plt.clf()

        fig = plt.figure()
        ax = plt.axes(projection="3d")

        ax.plot3D(
            avg_reduced_ECoG_data[:, 0],
            avg_reduced_ECoG_data[:, 1],
            avg_reduced_ECoG_data[:, 2],
            "gray",
        )
        ax.scatter3D(
            avg_reduced_ECoG_data[:, 0],
            avg_reduced_ECoG_data[:, 1],
            avg_reduced_ECoG_data[:, 2],
            c=np.arange(sr_dim[class_i]),
            cmap=class_color[class_n][1],
        )
        # plt.xlim([-0.6, 0.6])
        # plt.ylim([-0.3, 0.3])
        # ax.set_zlim(-0.4, 0.4)

        plt.title("First 3 PCs for " + class_dict[class_n])
        plt.savefig(
            sbj_sp
            + freq_name
            + "_"
            + class_dict[class_n]
            + "_3d_manifold_trajectory.png"
        )
        plt.close()


def plot_all_manifold_trials(
    class_dict, cur_classes, class_color, sbj_sp, freq_name, reduced_class_ECoG_data
):
    """
    Plot all trial trajectories in the current pca data

    Args:
        class_dict (dict): dictionary mapping class numbers to class names
        cur_classes (list): list of classes to plot
        class_color (dict): dictionary mapping class names to colors for the plot
        sbj_sp (str): where to save the figure
        freq_name (str): name of the current frequency band for plot
        reduced_class_ECoG_data (list): List of (n_trials * n_time, n_channels)
            for each class, mapped to PCA space
    """
    for class_i, class_n in enumerate(cur_classes):
        # for i, reduced_data in enumerate(reduced_class_ECoG_data):
        reduced_data = reduced_class_ECoG_data[class_i]
        if len(reduced_data) == 0:
            continue
        fig = plt.figure()
        ax = plt.axes(projection="3d")
        # ax.plot3D(reduced_data[:, 0], reduced_data[:, 1], reduced_data[:, 2], "gray")
        ax.scatter3D(
            reduced_data[:, 0],
            reduced_data[:, 1],
            reduced_data[:, 2],
            c=np.arange(reduced_data.shape[0]),
            cmap=class_color[class_n][1],
        )
        plt.title("First 3 PCs for " + class_dict[class_n])
        plt.savefig(
            sbj_sp
            + freq_name
            + "_"
            + class_dict[class_n]
            + "_3d_manifold_all_trials.png"
        )
        plt.close()


def plot_chan_contribs(
    sbj, class_dict, cur_classes, sbj_sp, freq_name, class_pca, roi_centroids=None
):
    """
    Plot the contribution of each channel/ROI to the first PC

    Args:
        sbj (int): int determining which subject to plot
        class_dict (dict): dictionary mapping class numbers to class names
        cur_classes (list): list of classes to plot
        sbj_sp (str): where to save the figure
        freq_name (str): name of the current frequency band for plot
        class_pca (list): list of PCA objects for each class
        roi_centroids (list, optional): List of centroid locations, if not None.
            Defaults to None.
    """
    if roi_centroids is None:
        mni_elec_lp = "/nas/ecog_project/derived/electrode_mni_locations/"
        # need a dict here to modify and pull in the correct electrode info
        elec_sbj_name_dict = {
            1: "a0f66459",
            2: "c95c1e82",
            3: "cb46fd46",
            4: "fcb01f7a",
            5: "ffb52f92",
            6: "b4ac1726",
            7: "f3b79359",
            8: "ec761078",
            9: "f0bbc9a9",
            10: "abdb496b",
            11: "ec168864",
            12: "b45e3f7b",
        }
        sbj_name = elec_sbj_name_dict[sbj + 1]
        mni_elec_file = sbj_name + "_MNI_atlasRegions.xlsx"
        if not os.path.isfile((mni_elec_lp + mni_elec_file)):
            print("No electrode file, skip electrode contrib plots")
            return

        df_elec_pos = pd.read_excel(mni_elec_lp + mni_elec_file)
    else:
        df_elec_pos = roi_centroids[sbj]

    node_size = 50
    sides_2_display = ["x", "y", "z"]
    # for i, c_pca in enumerate(class_pca):
    for class_i, class_n in enumerate(cur_classes):
        c_pca = class_pca[class_i]
        if type(c_pca) == list:
            continue
        print("Channel Contributions for " + class_dict[class_n])
        component_dim = c_pca.components_[0, :].shape[0]
        df_dim = len(df_elec_pos[["X coor", "Y coor", "Z coor"]])
        # pdb.set_trace()
        if component_dim != df_dim:
            print("not matching dimensions, trimming!")
            min_dim = min(component_dim, df_dim)
            pca_components = c_pca.components_[0, 0:min_dim]
            elec_pos = df_elec_pos[["X coor", "Y coor", "Z coor"]][0:min_dim]
        else:
            pca_components = c_pca.components_[0, :]
            elec_pos = df_elec_pos[["X coor", "Y coor", "Z coor"]][
                0: pca_components.shape[0]
            ]
        for disp in sides_2_display:
            fig, ax = plt.subplots(1, 1, figsize=(7.5, 5))
            #     fig.set_cmap = 'viridis'
            # ax.set_title('Regions of Interest (ROIs) in Sensori-motor areas \n ROI' + str(ROIs_of_interest[0]) + 'highlighted')
            #     ni_plt.plot_connectome(np.eye(df_elec_pos.shape[0]), df_elec_pos[["X coor", "Y coor", "Z coor"]], output_file=None,
            #                                    node_kwargs={'alpha': 1, 'edgecolors': 'silver','linewidths':.5,'marker': 'o'},
            #                                    node_size=node_size, node_color=c_pca.components_[0,:], display_mode=disp, axes=ax, colorbar=True)

            ni_plt.plot_markers(
                node_values=abs(pca_components),
                node_coords=elec_pos,
                node_size=node_size,
                display_mode=disp,
                axes=ax,
                node_kwargs={
                    "alpha": 1,
                    "edgecolors": "silver",
                    "linewidths": 0.5,
                    "marker": "o",
                },
            )
            plt.savefig(
                sbj_sp
                + freq_name
                + "_"
                + class_dict[class_n]
                + "_"
                + disp
                + "_slice_channel_contribs.png"
            )
            plt.close()
