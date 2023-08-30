import numpy as np
import scipy as sci
import matplotlib.pyplot as plt
import seaborn as sns
from brokenaxes import brokenaxes
from matplotlib.gridspec import GridSpec
import matplotlib as mpl


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
    ax.hlines(
        sigf_val, -0.5, len(order), linestyles="dashed", color="black"
    )
    if var_of_interest_icon is not None:
        im_ax = ax.inset_axes([0.7, 0.7, 0.47, 0.47])
        im_ax.imshow(var_of_interest_icon)
        im_ax.axis('off')
