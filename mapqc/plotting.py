import pandas as pd
import numpy as np
import scanpy as sc
import matplotlib.pyplot as plt
import seaborn as sns

def plot_ref_vs_q_dists_per_nhood(mapqc_results):
    if not mapqc_results._sample_dists_to_ref_calculated:
        return ValueError(
            "Plot can only be generated after running calculate_sample_dists_to_ref_samples()."
        )
    if not mapqc_results._results_prepped_for_boxplot:
        _convert_nhood_dists_to_plottable_dataframe(mapqc_results)
    fig = plt.figure(figsize=(24, 8))
    ax = fig.add_subplot()
    nhood_order = mapqc_results.nhood_stats.sort_values(by="pval_corr").index.tolist()
    y = "mean_dist_to_ref"
    hue = "sample_r_or_q"
    sns.boxplot(
        mapqc_results._boxplot_df_mean_sample_dists_to_ref,
        x="nhood",
        y=y,
        order=nhood_order,
        hue=hue,
        ax=ax,
        saturation=0.2,
    )
    sns.stripplot(
        mapqc_results._boxplot_df_mean_sample_dists_to_ref,
        x="nhood",
        y=y,
        order=nhood_order,
        hue=hue,
        dodge=True,
        ax=ax,
        size=1,
        palette="dark:black",
    )
    ax.tick_params(axis="x", rotation=90)


def plot_umap_significance(mapqc_results, title="neighborhood significance"):
    if not mapqc_results._adata_prepped_for_plotting:
        _prepare_adata_for_plotting(mapqc_results)
        sc.set_figure_params(figsize=(5, 5))
    sc.pl.umap(
        mapqc_results._adata_ordered,
        color="mapqc_sign",
        size=mapqc_results._adata_ordered.obs.mapqc_dotsize,
        cmap="Reds",
        sort_order=False,
        title=title,
        vmin=0,
        vmax=1,
    )


def _convert_nhood_dists_to_plottable_dataframe(mapqc_results):
    """Creates a DataFrame of with ... and ... as multi-level index, and
    ... as columns."""
    # convert to pandas dataframe
    mean_s2ref_dist_per_nhood_df_for_boxplot = (
        pd.DataFrame(
            mapqc_results.mean_sample_dists_to_ref,
            index=mapqc_results.ref_q_samples_all,
            columns=mapqc_results.center_cells,
        )
        .unstack()
        .reset_index()
        .rename(
            columns={"level_0": "nhood", "level_1": "sample", 0: "mean_dist_to_ref"}
        )
    )
    # remove rows with nans
    mean_s2ref_dist_per_nhood_df_for_boxplot.dropna(how="any", axis=0, inplace=True)
    # add info on whether sample is from ref or query
    sample_from_r_or_q_df = mapqc_results.adata.obs.groupby(
        mapqc_results.sample_key
    ).agg({mapqc_results.ref_q_key: "first"})
    sample_from_r_or_q_dict = {
        s: rvsq
        for s, rvsq in zip(
            sample_from_r_or_q_df.index, sample_from_r_or_q_df[mapqc_results.ref_q_key]
        )
    }
    mean_s2ref_dist_per_nhood_df_for_boxplot[
        "sample_r_or_q"
    ] = mean_s2ref_dist_per_nhood_df_for_boxplot["sample"].map(sample_from_r_or_q_dict)
    mapqc_results._boxplot_df_mean_sample_dists_to_ref = (
        mean_s2ref_dist_per_nhood_df_for_boxplot
    )


def _prepare_adata_for_plotting(mapqc_results, scale=1):
    if not mapqc_results._nhood_stats_calculated:
        raise ValueError(
            "Calculate result statistics first using the 'calculate_nhood_statistics' function."
        )
    for stat_to_copy in ["nhood_weight", "pval_corr", "sign"]:
        mapqc_results.adata.obs[f"mapqc_{stat_to_copy}"] = np.nan
        mapqc_results.adata.obs.loc[
            mapqc_results.nhood_stats.index, f"mapqc_{stat_to_copy}"
        ] = mapqc_results.nhood_stats[stat_to_copy]
    # set sizes for plotting:
    mapqc_results.adata.obs["mapqc_dotsize"] = scale * 1
    mapqc_results.adata.obs.loc[mapqc_results.nhood_stats.index, "mapqc_dotsize"] = (
        scale * 20
    )
    # order cells so that index cells are plotted on top
    cells_ordered = (
        mapqc_results.adata.obs.index[
            ~mapqc_results.adata.obs.index.isin(mapqc_results.nhood_stats.index)
        ].tolist()
        + mapqc_results.nhood_stats.index.tolist()
    )
    # store ordered version of adata
    # TO DO: we're storing adata twice now, which is very memory inefficient
    # but I don't want to re-order the original AnnData (it will mess
    # with all the code above which relies on the adata row order). Think
    # of a better alternative here...
    mapqc_results._adata_ordered = mapqc_results.adata[cells_ordered, :].copy()
    # remember that adata has been prepped for plotting:
    mapqc_results._adata_prepped_for_plotting = True
