import mne
import pandas as pd
import numpy as np
import json
from tqdm import tqdm

import matplotlib.pyplot as plt
import seaborn as sns
from nilearn import plotting as ni_plt

from src.data_utils import ECoG_Data
import src.manifold_u as mu

filepath = '/home/zsteineh/research_projects/NaturalisticNeuralManifolds/experiment_params/'
file_name = 'exp_params_nat_pca.json'
try:
    json_filename = filepath + file_name
except IndexError:
    raise SystemExit(
        f"Usage: {sys.argv[0]} <json file of experiment parameters>")
with open(json_filename) as f:
    exp_params = json.load(f)
print(exp_params['comment'])

freq_bands = exp_params["freq_bands"]
class_dict = exp_params["class_dict"]
class_dict = {int(cur_key): val for cur_key, val in class_dict.items()}
class_color = exp_params["class_color"]
class_color = {int(cur_key): val for cur_key, val in class_color.items()}

proj_mat_sp = exp_params["sp"] + \
    exp_params["dataset"] + exp_params["experiment_folder"]
print(proj_mat_sp)

percent_threshold = 0.8
pats_ids_in = exp_params["pats_ids_in"]
test_days = exp_params["test_day"]
# convert ids to new varition that starts with P
new_pats_ids = {sbj: "N"+sbj[-2:] for sbj in pats_ids_in}

roi_regions = pd.read_csv(
    '/data2/users/zsteineh/nm_paper_data/naturalistic_ecog_data/elec_and_roi/nat_roi_regions.csv')
print(roi_regions.head())


# freq_bands = ['LFO']
roi_contrib_df = pd.DataFrame(columns=['Frequency',
                                       'Participant',
                                       'Day',
                                       'Movement',
                                       'PC Dimension',
                                       'ROI Number',
                                       'ROI Region',
                                       'Contribution Weight'])
for f, freq in enumerate(freq_bands):
    print("Calculating for frequency band: " + freq)
    # get the PCA components loaded in
    all_sbjs_pca = np.load(proj_mat_sp + freq +
                           "_pca_objects.npy", allow_pickle=True)
    # print(np.array(all_sbjs_pca).shape)
    # of shape (pats, days, mvmts)

    freq_red_dim = mu.choose_one_freq_dimensionality(class_dict,
                                                     freq,
                                                     pats_ids_in,
                                                     np.expand_dims(
                                                         all_sbjs_pca, axis=0)
                                                     )
    for p, pat in tqdm(enumerate(new_pats_ids)):
        for d, day in enumerate(test_days):
            for m, mvmt in enumerate(class_dict):
                # load in this specific PCA
                cur_pca = all_sbjs_pca[p][d][m]
                if cur_pca == []:
                    print("no data")
                    continue
                # get the top 15 components
                top_dims = cur_pca.components_[0:freq_red_dim, :]
                for pc_i, pc in enumerate(top_dims):
                    for r, roi_contrib in enumerate(pc):
                        # print(abs(roi_contrib))
                        # add to the dataframe
                        roi_contrib_df = roi_contrib_df.append({'Frequency': freq,
                                                                'Participant': pat,
                                                                'Day': day,
                                                                'Movement': mvmt,
                                                                'PC Dimension': pc_i,
                                                                'ROI Number': r,
                                                                'ROI Region': roi_regions.iloc[r]['NN Labels'],
                                                                'Contribution Weight': abs(roi_contrib)},
                                                               ignore_index=True)

print(roi_contrib_df.head())
roi_contrib_df.to_csv(proj_mat_sp + 'roi_region_contrib_df.csv')
