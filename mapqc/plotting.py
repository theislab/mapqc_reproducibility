import pandas as pd
import numpy as np
import random
import scanpy as sc
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import rgb2hex
import seaborn as sns
from mapqc import mapQC_results


def plot_boxplot_of_ref_vs_q_dists_per_nhood(
    mapqc_results: mapQC_results,
    use_zscored_dists: bool = False,
    neighborhood: str = None,
    max_neighborhoods_to_plot: int = 80,
    seed: int = 0
):
    """
    Plot a split boxplot of, for each neighborhood, reference sample mean distances
    to all other reference samples, and query sample mean distances to all reference
    samples.

    Parameters
    ----------
    mapQC_results
        Instance of mapQC results class for which sample distance to reference samples
        have already been calculated using mapQC_results.calculate_sample_dists_to_ref_samples().
    use_zscored_dists
        Whether to plot raw euclidean distances, or distances zscored so that the mean
        reference distance to ref is 0, and the standard deviation 1.
    neighborhood
        Set to name of neighborhood (center cell) if the results for one specific neighborhood
        should be shown.
    max_neighborhoods_to_plot
        Maximum number of neighborhoods to plot (to keep figure legible)
    seed
        Random seed for subsetting neighborhoods

    Returns
    -------
    Plots boxplot as described above.

    TO DO:
    add option to return figure as variable, add other plotting options.
    """
    if not mapqc_results._sample_dists_to_ref_calculated:
        return ValueError(
            "Plot can only be generated after running calculate_sample_dists_to_ref_samples()."
        )
    # check if neighborhood is in center cells if only results for a specific neighborhood
    # should be shown:
    if not isinstance(neighborhood, type(None)):
        if not neighborhood in mapqc_results.center_cells:
            raise ValueError("Your neighborhood is not part of the center cells.")
    if not mapqc_results._results_prepped_for_boxplot:
        _convert_nhood_dists_to_plottable_dataframe(mapqc_results)
    
    if use_zscored_dists:
        plotting_df = mapqc_results._boxplot_df_mean_sample_zscored_dists_to_ref
        y = "mean_zscored_dist_to_ref"
    else:
        plotting_df = mapqc_results._boxplot_df_mean_sample_dists_to_ref
        y = "mean_dist_to_ref"
    # check if only results for a specific neighborhood should be plotted and subset if so:
    if not isinstance(neighborhood, type(None)):
        plotting_df = plotting_df.loc[plotting_df.nhood == neighborhood, :].copy()
        nhood_order = [neighborhood]
        figsize = (1, 4)
    else:
        # subset to lower number of neighborhoods if needed:
        if plotting_df.shape[0] > max_neighborhoods_to_plot:
            random.seed(seed)
            random_nhoods = random.sample(mapqc_results.nhood_stats.index.tolist(), k=max_neighborhoods_to_plot)
            nhood_order = mapqc_results.nhood_stats.loc[random_nhoods,:].sort_values(by="pval_corr").index.tolist()
        else:
            nhood_order = mapqc_results.nhood_stats.sort_values(by="pval_corr").index.tolist()
        figsize = (24, 8)
    hue = "sample_r_or_q"
    # palette
    palette = {mapqc_results.ref_cat: "tab:orange", mapqc_results.query_cat: "tab:blue"}
    # palette = {mapqc_results.ref_cat:"lightsteelblue", mapqc_results.query_cat:"blue"}
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot()
    sns.boxplot(
        plotting_df,
        x="nhood",
        y=y,
        order=nhood_order,
        hue=hue,
        ax=ax,
        saturation=0.4,
        palette=palette,
    )
    sns.stripplot(
        plotting_df,
        x="nhood",
        y=y,
        order=nhood_order,
        hue=hue,
        dodge=True,
        ax=ax,
        size=3,
        palette="dark:black",
    )
    leg = plt.legend(loc=(1.01, 0.4), frameon=False, title="")
    leg._legend_box.align = "left"
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
    # prepare color palette:
    palette = {
        "significant": "red",
        "not significant": "antiquewhite",
        "excluded": "black",
    }
    # plot
    sc.pl.umap(
        mapqc_results._adata_ordered,
        color="mapqc_sign",
        size=mapqc_results._adata_ordered.obs.mapqc_dotsize,
        palette=palette,
        sort_order=False,
        title=title,
        na_color="darkgrey",
        frameon=False,
    )


def plot_umap_zscored_q_dist_to_ref(
    mapqc_results: mapQC_results, title: str = None, scale: int = 1
):
    """
    Plot UMAP of adata in mapqc_results instance, coloring neighborhoods
    by the mean zscored query sample distance to reference sample distance
    (zscoring using the mean and standard deviation of the ref sample
    distance to the reference)

    Parameters
    ---------
    mapqc_results
        Instance of mapQC_results class, for which the adata has a UMAP pre-calculated
        and for which mapQC statistics have been calculated using
        mapQC_results.calculate_nhood_statistics

    scale
        scaling of UMAP dot size

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
    if isinstance(title, type(None)):
        title = f"Mean Z-scored distance of query samples to reference samples"
    # plot
    sc.pl.umap(
        mapqc_results._adata_ordered,
        color="mapqc_mean_zscored_q_dist_to_ref",
        size=mapqc_results._adata_ordered.obs.mapqc_dotsize,
        sort_order=False,
        cmap="coolwarm",
        title=title,
        vmin=-3,
        vmax=3,
        na_color="grey",
        frameon=False,
    )


def plot_umap_neighborhood(
    mapqc_results: mapQC_results, neighborhood: str, title: str = None, scale: int = 1
):
    """
    Plot UMAP of adata in mapqc_results instance, coloring the cells
    from a specific neighborhood

    Parameters
    ---------
    mapqc_results
        Instance of mapQC_results class, for which the adata has a UMAP pre-calculated
        and for which mapQC statistics have been calculated using
        mapQC_results.calculate_nhood_statistics

    neighborhood
        name of the neighborhood (i.e. its center cell)

    scale
        scaling of UMAP dot size

    title
        Title for UMAP.

    Returns
    -------
    Plots UMAP.

    TO DO: add option to return figure as variable and more plotting options.
    """
    if not "X_umap" in mapqc_results.adata.obsm.keys():
        raise ValueError("No UMAP calculated for this AnnData object.")
    if not mapqc_results._sample_dists_to_ref_calculated:
        raise ValueError(
            "Center cell neighbors have not yet been calculated, run mapqc_results.calculate_sample_dists_to_ref_samples first."
        )
    if not mapqc_results._adata_prepped_for_plotting:
        _prepare_adata_for_plotting(mapqc_results)
        sc.set_figure_params(figsize=(5, 5))
    # add neighborhood info to adata object
    nhood_idx = np.where(mapqc_results.center_cells == neighborhood)[0]
    if len(nhood_idx) == 0:
        raise ValueError(
            f"Neighborhood center cell {neighborhood} not found in adata. Check your neighborhood argument setting for typos."
        )
    mapqc_results.adata.obs[f"mapqc_nhood_cells_{neighborhood}"] = "Other"
    mapqc_results.adata.obs[f"mapqc_nhood_cells_{neighborhood}"].iloc[
        mapqc_results.nhood_cells[nhood_idx[0]]
    ] = neighborhood
    if isinstance(title, type(None)):
        title = f"Neighborhood (k={mapqc_results.k})"
    palette = {"Other": "lightgrey", neighborhood: "black"}
    dotsizes = mapqc_results.adata.obs[f"mapqc_nhood_cells_{neighborhood}"].map(
        {"Other": scale, neighborhood: scale * 6}
    )
    # prepare color palette:
    sc.pl.umap(
        mapqc_results.adata,
        color=f"mapqc_nhood_cells_{neighborhood}",
        size=dotsizes,
        palette=palette,
        sort_order=False,
        title=title,
        frameon=False,
    )


def plot_umap_q_ref_sample_shaded(
    mapqc_results: mapQC_results,
    include_legend: bool = False,
    neighborhood: str = None,
    reference_only: bool = False,
    seed: int = None,
    dotsize: int = None,
):
    """
    Plot UMAP of mapqc_results adata, colored by reference versus query, but
    shading those colors by sample (i.e. reference samples have different
    shades of orange, query samples different shades of blue). This helps get
    an impression of how well samples are mixing in different regions of the
    embedding. Can be done for a single neighborhood as well using the
    neighborhood argument.

    Parameters
    ----------
    mapqc_results
        mapqc_results instance
    include_legend
        whether or not to include a legend in the figure
    neighborhood
        Name of neighborhood, if only the cells of a specific neighborhood
        should be plotted
    reference_only
        Whether to only plot cells from the reference
    dotsize
        Size of dots in UMAP

    Returns
    -------
    Plots UMAP as described above
    """
    # check a few things of only a specific neighborhood should be shown,
    # and subset to relevant cells:
    if not isinstance(neighborhood, type(None)):
        # check if neighborhood is in center cells:
        if not neighborhood in mapqc_results.center_cells:
            raise ValueError("Your neighborhood is not part of the center cells.")
        # check if k nearest neighbors were already calculated:
        if not mapqc_results._sample_dists_to_ref_calculated:
            raise ValueError(
                "Center cell neighbors have not yet been calculated, run mapqc_results.calculate_sample_dists_to_ref_samples first."
            )
        nhood_idx = np.where(mapqc_results.center_cells == neighborhood)[0][0]
        nhood_cells = mapqc_results.nhood_cells[nhood_idx, :]
        adata_to_plot = mapqc_results.adata[nhood_cells, :].copy()
        title = f"Sample (query versus reference)\n(nhood {neighborhood}"
    else:
        adata_to_plot = mapqc_results.adata
        title = "Sample (query versus reference)"
    if reference_only:
        adata_to_plot = adata_to_plot[
            adata_to_plot.obs[mapqc_results.ref_q_key] == mapqc_results.ref_cat, :
        ].copy()
    # 1. generate colors for reference samples
    cmap_ref = matplotlib.colormaps["Wistia"]
    ref_samples = adata_to_plot.obs.loc[
        adata_to_plot.obs[mapqc_results.ref_q_key] == mapqc_results.ref_cat, :
    ]["sample"].unique()
    n_ref_samples = len(ref_samples)
    sample_floats_ref = np.arange(start=0, stop=n_ref_samples) / n_ref_samples
    sample_hex_ref = [rgb2hex(cmap_ref(sample_fl)) for sample_fl in sample_floats_ref]
    if not isinstance(seed, type(None)):
        np.random.seed(seed)
    np.random.shuffle(
        sample_hex_ref
    )  # randomize order so that not confounded by dataset (sample order is by dataset)
    # create dictionary with samples as keys and hex code of sample colors as values
    ref_sample_to_col = {
        samp: hexcol for samp, hexcol in zip(ref_samples, sample_hex_ref)
    }
    # 2. generate color for query samples
    if not reference_only:
        cmap_q = matplotlib.colormaps["Blues"]
        q_samples = adata_to_plot.obs.loc[
            mapqc_results.adata.obs[mapqc_results.ref_q_key] == mapqc_results.query_cat,
            :,
        ]["sample"].unique()
        n_q_samples = len(q_samples)
        sample_floats_q = np.arange(start=0, stop=n_q_samples) / n_q_samples
        sample_hex_q = [rgb2hex(cmap_q(sample_fl)) for sample_fl in sample_floats_q]
        if not isinstance(seed, type(None)):
            np.random.seed(seed)
        np.random.shuffle(
            sample_hex_q
        )  # randomize order so that not counfounded by dataset (sample order is by dataset)
        # create dictionary with sample to color hex mapping
        q_sample_to_col = {
            samp: hexcol for samp, hexcol in zip(q_samples, sample_hex_q)
        }
    # 3. join color mappings into single dictionary
    sample_to_color = ref_sample_to_col
    if not reference_only:
        sample_to_color.update(q_sample_to_col)
    # 4. plot umap
    if include_legend:
        legend_loc = "right margin"
    else:
        legend_loc = None
    sc.pl.umap(
        adata_to_plot,
        color="sample",
        palette=sample_to_color,
        legend_loc=legend_loc,
        title=title,
        frameon=False,
        size=dotsize,
    )


def _convert_nhood_dists_to_plottable_dataframe(mapqc_results: mapQC_results):
    """
    TO DO: UPDATE DOC WITH ZSCORE INFO
    Creates a DataFrame of intersample distances per neighborhood to be used
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
    # normal distances
    mean_s2ref_dist_per_nhood_df_for_boxplot = (
        pd.DataFrame(
            mapqc_results.mean_sample_dists_to_ref,  # mean_sample_dists_to_ref_zscored,
            index=mapqc_results.ref_q_samples_all,
            columns=mapqc_results.center_cells,
        )
        .unstack()
        .reset_index()
        .rename(
            columns={
                "level_0": "nhood",
                "level_1": "sample",
                0: "mean_dist_to_ref",
            }  # "mean_dist_to_ref"}
        )
    )
    # zscored distances:
    mean_s2ref_zscored_dist_per_nhood_df_for_boxplot = (
        pd.DataFrame(
            mapqc_results.mean_sample_dists_to_ref_zscored,
            index=mapqc_results.ref_q_samples_all,
            columns=mapqc_results.center_cells,
        )
        .unstack()
        .reset_index()
        .rename(
            columns={
                "level_0": "nhood",
                "level_1": "sample",
                0: "mean_zscored_dist_to_ref",
            }  # "mean_dist_to_ref"}
        )
    )
    # remove rows with nans
    mean_s2ref_dist_per_nhood_df_for_boxplot.dropna(how="any", axis=0, inplace=True)
    mean_s2ref_zscored_dist_per_nhood_df_for_boxplot.dropna(
        how="any", axis=0, inplace=True
    )
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
    mean_s2ref_zscored_dist_per_nhood_df_for_boxplot[
        "sample_r_or_q"
    ] = mean_s2ref_zscored_dist_per_nhood_df_for_boxplot["sample"].map(
        sample_from_r_or_q_dict
    )
    mapqc_results._boxplot_df_mean_sample_dists_to_ref = (
        mean_s2ref_dist_per_nhood_df_for_boxplot
    )
    mapqc_results._boxplot_df_mean_sample_zscored_dists_to_ref = (
        mean_s2ref_zscored_dist_per_nhood_df_for_boxplot
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
    for stat_to_copy in ["nhood_weight", "pval_corr", "mean_zscored_q_dist_to_ref"]:
        mapqc_results.adata.obs[f"mapqc_{stat_to_copy}"] = np.nan
        mapqc_results.adata.obs.loc[
            mapqc_results.nhood_stats.index, f"mapqc_{stat_to_copy}"
        ] = mapqc_results.nhood_stats[stat_to_copy]
    # include filtered out nhood information in "mapqc_sign" column:
    mapqc_results.adata.obs["mapqc_sign"] = np.nan
    sign_nhoods = mapqc_results.nhood_stats.loc[mapqc_results.nhood_stats.sign, :].index
    non_sign_nhoods = mapqc_results.nhood_stats.loc[
        ~mapqc_results.nhood_stats.sign, :
    ].index
    filtered_nhoods = [
        nhood
        for nhood in mapqc_results.center_cells
        if nhood not in mapqc_results.nhood_stats.index
    ]
    mapqc_results.adata.obs.loc[sign_nhoods, "mapqc_sign"] = "significant"
    mapqc_results.adata.obs.loc[non_sign_nhoods, "mapqc_sign"] = "not significant"
    mapqc_results.adata.obs.loc[filtered_nhoods, "mapqc_sign"] = "excluded"
    # set sizes for plotting:
    mapqc_results.adata.obs["mapqc_dotsize"] = scale * 1
    mapqc_results.adata.obs.loc[mapqc_results.center_cells, "mapqc_dotsize"] = (
        scale * 30
    )
    # order cells so that index cells are plotted on top
    cells_ordered = (
        mapqc_results.adata.obs.index[
            ~mapqc_results.adata.obs.index.isin(mapqc_results.center_cells)
        ].tolist()
        + mapqc_results.center_cells.tolist()
    )
    # store ordered version of adata
    # TO DO: we're storing adata twice now, which is very memory inefficient
    # but I don't want to re-order the original AnnData (it will mess
    # with all the code above which relies on the adata row order). Think
    # of a better alternative here...
    mapqc_results._adata_ordered = mapqc_results.adata[cells_ordered, :].copy()
    # remember that adata has been prepped for plotting:
    mapqc_results._adata_prepped_for_plotting = True
