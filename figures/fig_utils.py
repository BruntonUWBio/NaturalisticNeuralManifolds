import numpy as np
import scipy as sci
import matplotlib.pyplot as plt
import seaborn as sns
from brokenaxes import brokenaxes
from matplotlib.gridspec import GridSpec
import matplotlib as mpl
from nilearn import plotting as ni_plt

import src.manifold_u as mu


def subplot_VAF(avg_components,
                std_components,
                var_of_interest,
                var_of_interest_name,
                var_of_interest_colors,
                ax,
                var_of_interest_icon=None,
                percent_threshold=0.8,
                x_label=True,
                y_label=True,
                legend_loc='below'):
    # avg_components should be of the shape (len(var_of_interest), num_components)
    if len(avg_components) != len(var_of_interest):
        return "Error with plot"
    # var of interest should be a list with the vairable names
    bax = brokenaxes(
        xlims=((0, 30), (300, avg_components.shape[1])), hspace=0.05, subplot_spec=ax)
    for i, var in enumerate(var_of_interest):
        cum_VAF = 0
        lst_VAF = []
        percent_cross = None
        for j, cur_VAF in enumerate(avg_components[i]):
            cum_VAF += cur_VAF
            lst_VAF.append(cum_VAF)
            if percent_cross is None and cum_VAF > percent_threshold:
                percent_cross = j

        bax.plot(
            lst_VAF, c=var_of_interest_colors[var], label=var +
            " dim = " + str(percent_cross)
        )
        bax.axvline(percent_cross, linestyle="--",
                    c=var_of_interest_colors[var])

        # add in the std
        bax.fill_between(
            np.arange(len(lst_VAF)),
            lst_VAF - std_components[i, :],
            lst_VAF + std_components[i, :],
            alpha=0.2,
            color=var_of_interest_colors[var],
        )

    if var_of_interest_icon is not None:
        with mpl.cbook.get_sample_data(var_of_interest_icon) as file:
            arr_image = plt.imread(file)
        im_ax = bax.inset_axes([0.3, 0.95, 0.47, 0.47])
        im_ax[1].imshow(arr_image)
        for a in im_ax:
            a.axis('off')
    bax.set_ylim([0, 1.05])
    bax.axhline(percent_threshold, c="k", linestyle="--")
    if x_label:
        bax.set_xlabel("Number of PCs", labelpad=-20)
    if y_label:
        bax.set_ylabel("Neural Variance\nAccounted For (\%)")
    if len(var_of_interest) > 9:
        n_cols = 6
        bax.legend(loc="lower right", ncol=n_cols,
                   frameon=True, bbox_to_anchor=(1.18, -0.5))
    elif len(var_of_interest) >= 6 and legend_loc == 'below':
        n_cols = 2
        bax.legend(loc="lower right", ncol=n_cols,
                   frameon=True, bbox_to_anchor=(1.6, -0.85))
    else:
        n_cols = 1
        # , bbox_to_anchor=(1.18, 0.95))
        bax.legend(loc="lower right", ncol=n_cols, frameon=True)
    # bax.set_title(var_of_interest_name + " VAF", loc = 'left', y = 0.99)


def plot_PAs(ax,
             class_dict,
             comps_to_plot,
             all_day_pas,
             null_data_pa,
             red_dim,
             null_sig_pas=None,
             null_all=False,
             const_label=False,
             color_black=False,
             fill_between=False,
             legend_on=True,
             participant=0,
             frequency=0):
    # assume all_days_pas is of shape (days, freqs, pats, mvmts comparisons, manifold dim)
    # assume null_data_pa is of shape (freqs, pats, reruns, 1, 1, mvmts comparisons, manifold dim)
    class_vs_dict = mu.get_pa_comparison_names(class_dict)
    evenly_spaced_interval = np.linspace(0, 1, len(comps_to_plot))
    # may want a better cm later
    colors = [plt.cm.cool(x) for x in evenly_spaced_interval]
    # upper_diag_ind = 0
    # c = 0
    for c, comp in enumerate(comps_to_plot):
        # gets the average over the days
        theta_vals = np.nanmean(all_day_pas, axis=0)[
            frequency, participant, comp, ...]
        theta_std = np.nanstd(all_day_pas, axis=0)[
            frequency, participant, comp, ...]
        if const_label:
            cur_label = 'Movement Comparison'
        else:
            cur_label = class_vs_dict[comp]
        if color_black:
            color = "black"
        else:
            color = colors[c]
        ax.plot(theta_vals, label=cur_label, color=color)
        if fill_between:
            ax.fill_between(
                np.arange(len(theta_vals)),
                theta_vals - theta_std,
                theta_vals + theta_std,
                alpha=0.3,
                color=colors[c],
            )

    if null_data_pa is not None and null_all:
        null_pa = np.array(null_data_pa[frequency][participant])
        # turn to shape (1000 runs, comps, manifold dim)
        # this will plot all the null samples for the whole distrubition
        null_pa = np.squeeze(null_pa)
        for s in range(len(null_pa)):
            ax.plot(null_pa[s, 6], color="grey", alpha=0.1)
            if s == 1:
                ax.plot(null_pa[s, 6], color="grey", label="Null Samples")

    if null_sig_pas is not None:
        ax.plot(null_sig_pas, linestyle="--", color="black", label="Null 1%")

    ax.set_ylim(0, 95)
    ax.set_yticks([30, 60, 90])
    ax.set_xlabel("Manifold Dimension")
    ax.set_ylabel("Principal Angle (deg)")
    # ax.set_xticks([0, 4, 9, 14])
    # ax.set_xticklabels([1, 5, 10, 15])
    ax.set_xlim([0, red_dim])
    if legend_on:
        ax.legend(loc="center", frameon=True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_neural_dissimilarity(ax,
                              neur_dis_df,
                              sigf_val,
                              order,
                              colors,
                              x="Participant",
                              y="Neural Dissimilarity",
                              var_of_interest_icon=None):
    sns.violinplot(
        x=x,
        y=y,
        data=neur_dis_df,
        inner="quart",
        order=order,
        ax=ax,
        palette=colors,
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xticklabels(labels=order, rotation=45)
    ax.set_ylim([-0.05, 1])
    ax.set_yticks([0, 0.5, 1])
    ax.hlines(
        sigf_val, -0.5, len(order), linestyles="dashed", color="black"
    )
    if var_of_interest_icon is not None:
        im_ax = ax.inset_axes([0.7, 0.7, 0.47, 0.47])
        im_ax.imshow(var_of_interest_icon)
        im_ax.axis('off')


def plot_electrodes(ax,
                    df_elec_pos,
                    sbj_num,
                    electrode_color,
                    data_type='N',
                    side_2_display="x",
                    node_size=50,
                    alpha=1,
                    edgecolors="silver",
                    node_cmap="Greys",
                    linewidths=0.5,
                    marker="o",
                    colorbar=True,
                    node_vmax=None,
                    node_vmin=None,):
    # Include asterisk for RH patients
    average_xpos_sign = np.mean(np.asarray(df_elec_pos['X coor']))
    if average_xpos_sign > 0:
        suff = '*'
    else:
        suff = ''

    df_dim = len(df_elec_pos[["X coor", "Y coor", "Z coor"]])
    elec_pos = df_elec_pos[["X coor", "Y coor", "Z coor"]]
    ni_plt.plot_markers(
        node_values=electrode_color,
        node_coords=elec_pos,
        node_size=node_size,
        display_mode=side_2_display,
        axes=ax,
        node_kwargs={
            "alpha": alpha,
            "edgecolors": edgecolors,
            "linewidths": linewidths,
            "marker": marker,
        },
        node_cmap=node_cmap,
        node_vmax=node_vmax,
        node_vmin=node_vmin,
        colorbar=colorbar,
        # node_cmap="binary_r"
    )
    # format sbj_num to include leading 0
    # print_num = "%02d" % (sbj_num + 1)
    ax.set_title(f"{data_type}{sbj_num + 1:02d}{suff}", fontsize=24)


def plot_roi_contribs(ax,
                      first_component,
                      df_elec_pos,
                      side_2_display="l",
                      node_size=50,
                      alpha=1,
                      edgecolors="silver",
                      linewidths=0.5,
                      marker="o",
                      colorbar=True,
                      node_vmax=None,
                      node_vmin=None,):
    component_dim = first_component.shape[0]
    df_dim = len(df_elec_pos[["X coor", "Y coor", "Z coor"]])

    if component_dim != df_dim:
        print("not matching dimensions!")
        return
    else:
        elec_pos = df_elec_pos[["X coor", "Y coor", "Z coor"]][
            0: first_component.shape[0]
        ]
    ni_plt.plot_markers(
        node_values=abs(first_component),
        node_coords=elec_pos,
        node_size=node_size,
        display_mode=side_2_display,
        axes=ax,
        node_kwargs={
            "alpha": alpha,
            "edgecolors": edgecolors,
            "linewidths": linewidths,
            "marker": marker,
        },
        node_vmax=node_vmax,
        node_vmin=node_vmin,
        colorbar=colorbar,
    )
