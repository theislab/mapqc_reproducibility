import pandas as pd
import numpy as np
import scanpy as sc
import matplotlib.pyplot as plt
import seaborn as sns
from mapqc import mapQC_results


def plot_boxplot_of_ref_vs_q_dists_per_nhood(mapqc_results: mapQC_results):
    """
    Plot a split boxplot of, for each neighborhood, reference sample mean distances
    to all other reference samples, and query sample mean distances to all reference
    samples.

    Parameters
    ----------
    mapQC_results
        Instance of mapQC results class for which sample distance to reference samples
        have already been calculated using mapQC_results.calculate_sample_dists_to_ref_samples().

    Returns
    -------
    Plots boxplot.

    TO DO:
    add option to return figure as variable, add other plotting options.
    """
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


def plot_umap_significance(
    mapqc_results: mapQC_results, title: str = "neighborhood significance"
):
    """
    Plot UMAP of adata in mapqc_results instance, colored grey for all non-index
    cells (i.e. all cells that were not the center of a tested neighborhood),
    white and large for all index cells that passed filtering and for which
    reference-to-reference sample distances were not smaller than reference-
    to-query sample distances, and red for index cells for which reference-
    to-reference distances were significantly smaller than reference-to-query
    samples.

    Parameters
    ---------
    mapqc_results
        Instance of mapQC_results class, for which the adata has a UMAP pre-calculated
        and for which mapQC statistics have been calculated using
        mapQC_results.calculate_nhood_statistics

    title
        Title for UMAP.

    Returns
    -------
    Plots UMAP.

    TO DO: add option to return figure as variable and more plotting options.
    """
    if not "X_umap" in mapqc_results.adata.obsm.keys():
        raise ValueError("No UMAP calculated for this AnnData object.")
    if not mapqc_results._nhood_stats_calculated:
        raise ValueError(
            "Results have not yet been calculated, run mapqc_results.calculate_nhood_statistics first."
        )
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


def _convert_nhood_dists_to_plottable_dataframe(mapqc_results: mapQC_results):
    """Creates a DataFrame of intersample distances per neighborhood to be used
    for split boxplot.

    Parameters
    ----------
    mapqc_results
        Instance of mapQC_results class for which intersample distances have
        been calculated.

    Returns
    -------
    pandas.DataFrame stored under mapqc_results._boxplot_df_mean_sample_dists_to_ref
    with multiple rows per neighborhood (neighborhood name, i.e.
    center cell barcode, specified in the dataframe), with each neighborhood row
    representing a single sample in the neighborhood that passed filtering, for
    which its mean distance to the reference samples in that neighborhood can be
    found under 'mean_dist_to_ref'. sample_ref_or_q specifies whether the sample
    is from the reference or the query.
    """
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


def _prepare_adata_for_plotting(mapqc_results: mapQC_results, scale: int = 1):
    """
    Prepares AnnData of mapqc_results for plotting by adding the calculated
    statistics to the mapQC_results.adata instance, calculating the dot size
    such that neighborhood center cells are plotted bigger than the other cells,
    and orders cells such that center cells are plotted on top.

    Parameters
    ----------
    mapqc_results
        Instance of mapQC_results class for which neighborhood statistics have
        already been calculated using the mapQC_results.calculate_nhood_statistics
        function
    scale
        Scale with which to change overall dot size in UMAP.

    Returns
    -------
    Adds statistics to mapqc_results.adata, and creates a mapqc_results._adata_ordered
    to which statistics have also been added and which has been ordered such
    that neighborhood center cells are plotted on top.
    The information added to the adata object is stored under columns whose name
    starts with "mapqc", e.g. "mapqc_nhood_weight" or "mapqc_dotsize".
    """
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
