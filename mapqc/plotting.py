import pandas as pd
import numpy as np
import random
import scanpy as sc
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import rgb2hex
import seaborn as sns
from mapqc import mapQC_results
from typing import Literal


def plot_boxplot_of_ref_vs_q_dists_per_nhood(
    mapqc_results: mapQC_results,
    use_zscored_dists: bool = False,
    neighborhood: str = None,
    max_neighborhoods_to_plot: int = 80,
    seed: int = 0,
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
    # if not mapqc_results._results_prepped_for_boxplot:
    # _convert_nhood_dists_to_plottable_dataframe(mapqc_results)
    if use_zscored_dists:
        plotting_df = mapqc_results.mean_s2ref_dist_per_nhood_zscored_df
        y = "mean_zscored_dist_to_ref"
    else:
        plotting_df = mapqc_results.mean_s2ref_dist_per_nhood_df
        y = "mean_dist_to_ref"
    # check if only results for a specific neighborhood should be plotted and subset if so:
    if not isinstance(neighborhood, type(None)):
        plotting_df = plotting_df.loc[plotting_df.nhood == neighborhood, :].copy()
        nhood_order = [neighborhood]
        figsize = (1, 4)
    else:
        # subset to lower number of neighborhoods if needed:
        if mapqc_results.nhood_stats.shape[0] > max_neighborhoods_to_plot:
            random.seed(seed)
            random_nhoods = random.sample(
                mapqc_results.nhood_stats.index.tolist(), k=max_neighborhoods_to_plot
            )
        else:
            random_nhoods = mapqc_results.nhood_stats.index.tolist()
        # if use_zscored_dists:
        #     # nhoods_ordered_by_zscore = mapqc_results.nhood_stats.sort_values(by='median_zscored_q_dist_to_ref',ascending=False).index
        #     # nhood_order = [nhood for nhood in nhoods_ordered_by_zscore if nhood in random_nhoods]
        nhood_order = (
            mapqc_results.nhood_stats.loc[random_nhoods, :]
            .sort_values(
                by=f"perc{mapqc_results.q}_zscored_q_dist_to_ref", ascending=False
            )
            .index.tolist()
        )
        # else:
        # nhood_order = (
        #     mapqc_results.nhood_stats.loc[random_nhoods, :]
        #     .sort_values(by="pval_corr")
        #     .index.tolist()
        # )
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
    # if use_zscored_dists:
    ax.set_xticklabels(
        [
            round(
                mapqc_results.nhood_stats.loc[
                    nhood, f"perc{mapqc_results.q}_zscored_q_dist_to_ref"
                ],
                2,
            )
            for nhood in nhood_order
        ]
    )
    ax.set_xlabel(f"nhood (query zscore, {mapqc_results.q}th percentile)")
    # else:
    #     ax.set_xticklabels(
    #         [
    #             round(mapqc_results.nhood_stats.loc[nhood, "pval_corr"], 4)
    #             for nhood in nhood_order
    #         ]
    #     )
    #     ax.set_xlabel("nhood (corr. pval)")
    ax.tick_params(axis="x", rotation=90)


def plot_umap_significance(
    mapqc_results: mapQC_results,
    title: str = "neighborhood significance",
    scale_size_by_nhood_weight: bool = True,
    scale: float = 1,
):
    """
    Plot UMAP of adata in mapqc_results instance, colored grey for all non-index
    cells (i.e. all cells that were not the center of a tested neighborhood),
    white and large for all index cells that passed filtering and for which
    reference-to-reference sample distances were not smaller than reference-
    to-query sample distances, and red for index cells for which reference-
    to-reference distances were significantly smaller than reference-to-query
    samples. Neighborhoods that did not pass filtering are colored in black.
    Use plot_umap_filtering_info to color by exact reason for exclusion.

    Parameters
    ---------
    mapqc_results
        Instance of mapQC_results class, for which the adata has a UMAP pre-calculated
        and for which mapQC statistics have been calculated using
        mapQC_results.calculate_nhood_statistics
    title
        Title for UMAP.
    scale_size_by_nhood_weight
        Whether to scale dotsize of neighborhood centers by neighborhood weight, or to
        give all neighborhoods equal size. Neighborhood weight depends on the size of
        the neighborhoods, as well as on in how many other neighborhoods each cell in
        a given neighborhood appears.
    scale
        scalar to increase or decrease umap dot size

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
    # set dotsize:
    if scale_size_by_nhood_weight:
        size_col = "mapqc_dotsize_by_nhood_weight"
    else:
        size_col = "mapqc_dotsize"
    # plot
    sc.pl.umap(
        mapqc_results._adata_ordered,
        color="mapqc_sign",
        size=scale * mapqc_results._adata_ordered.obs[size_col],
        palette=palette,
        sort_order=False,
        title=title,
        na_color="darkgrey",
        frameon=False,
    )


def plot_umap_filtering_info(
    mapqc_results: mapQC_results, title: str = "neighborhood filtering"
):
    """
    Plot UMAP of adata in mapqc_results instance, colored grey for all non-index
    cells (i.e. all cells that were not the center of a tested neighborhood),
    white and large for all index cells that passed filtering, and in several
    different colors for different reasons why a neighborhood didn't pass filtering
    (see legend).

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
        "pass": "green",
        "too few reference samples from different batches": "salmon",
        "too few reference samples": "teal",
        "too few query samples": "black",
    }
    # plot
    sc.pl.umap(
        mapqc_results._adata_ordered,
        color="mapqc_filter_info",
        size=mapqc_results._adata_ordered.obs.mapqc_dotsize,
        palette=palette,
        sort_order=False,
        title=title,
        na_color="darkgrey",
        frameon=False,
    )


def plot_umap_zscored_q_dist_to_ref(
    mapqc_results: mapQC_results,
    color_by: Literal[
        "mean_zscore", "mean_zscore>2", "median_zscore>2", "percq_zscore>2"
    ] = "mean_zscore",
    title: str = None,
    scale: int = 1,
    samples_to_exclude: list = None,
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

    color_by
        what to color the umap by. Either mean z-score, or a binary specifying whether
        the mean/median/qth percentile zscore across query samples passes 2.

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
    if not isinstance(samples_to_exclude, type(None)):
        # check if samples exist:
        if (
            np.sum(
                [s not in mapqc_results.query_samples_all for s in samples_to_exclude]
            )
            > 0
        ):
            raise ValueError(
                "One or more of the samples in samples_to_exclude are not found among the query samples."
            )
        samples_to_include_boolean = [
            (sample in mapqc_results.query_samples_all)
            and (sample not in samples_to_exclude)
            for sample in mapqc_results.ref_q_samples_all
        ]
        samples_to_exclude_string = "_".join(samples_to_exclude)
    if color_by == "mean_zscore":
        if isinstance(samples_to_exclude, type(None)):
            color_cat = "mapqc_mean_zscored_q_dist_to_ref"
            palette = None
            if isinstance(title, type(None)):
                title = f"Mean Z-scored distance of query samples to reference samples"
        else:
            # recalculate mean zscore, but excluding samples
            mean_zscored_q_dist_to_ref = np.nanmean(
                mapqc_results.mean_sample_dists_to_ref_zscored[
                    samples_to_include_boolean, :
                ],
                axis=0,
            )
            color_cat = (
                f"mapqc_{color_by}_excluding_samples_{samples_to_exclude_string}"
            )
            mapqc_results._adata_ordered.obs[color_cat] = np.nan
            mapqc_results._adata_ordered.obs.loc[
                mapqc_results.center_cells, color_cat
            ] = mean_zscored_q_dist_to_ref
        palette = None
    elif color_by in ["mean_zscore>2", "median_zscore>2", "percq_zscore>2"]:
        if isinstance(samples_to_exclude, type(None)):
            if color_by == "percq_zscore>2":
                color_cat = "mapqc_" + f"perc{mapqc_results.q}_zscore>2"
            else:
                color_cat = "mapqc_" + color_by
        else:
            color_cat = (
                f"mapqc_{color_by}_excluding_samples_{samples_to_exclude_string}"
            )
            # recalculate metric excluding samples in samples_to_exclude list
            if color_by == "mean_zscore>2":
                mean_zscored_q_dist_to_ref = np.nanmean(
                    mapqc_results.mean_sample_dists_to_ref_zscored[
                        samples_to_include_boolean, :
                    ],
                    axis=0,
                )
                mapqc_results._adata_ordered.obs[color_cat] = np.nan
                mapqc_results._adata_ordered.obs.loc[
                    mapqc_results.center_cells, color_cat
                ] = (mean_zscored_q_dist_to_ref > 2)
            elif color_by == "median_zscore>2":
                median_zscored_q_dist_to_ref = np.nanmedian(
                    mapqc_results.mean_sample_dists_to_ref_zscored[
                        samples_to_include_boolean, :
                    ],
                    axis=0,
                )
                mapqc_results._adata_ordered.obs[color_cat] = np.nan
                mapqc_results._adata_ordered.obs.loc[
                    mapqc_results.center_cells, color_cat
                ] = (median_zscored_q_dist_to_ref > 2)
            elif color_by == "percq_zscore>2":
                percq_zscored_q_dist_to_ref = np.nanpercentile(
                    mapqc_results.mean_sample_dists_to_ref_zscored[
                        samples_to_include_boolean, :
                    ],
                    q=mapqc_results.q,
                    axis=0,
                )
                color_cat = f"mapqc_perc{mapqc_results.q}_zscore>2_excluding_samples_{samples_to_exclude_string}"
                mapqc_results._adata_ordered.obs[color_cat] = np.nan
                mapqc_results._adata_ordered.obs.loc[
                    mapqc_results.center_cells, color_cat
                ] = (percq_zscored_q_dist_to_ref > 2)
            # map boolean to strings:
            mapqc_results._adata_ordered.obs[
                color_cat
            ] = mapqc_results._adata_ordered.obs[color_cat].map(
                {True: ">2 stdev above mean", False: "<=2 stdev above mean"}
            )
            # set excluded neighborhoods as such
            mapqc_results._adata_ordered.obs.loc[
                mapqc_results._nhood_filtered.index[
                    mapqc_results._nhood_filtered != "pass"
                ],
                color_cat,
            ] = "excluded"
        palette = {
            ">2 stdev above mean": "red",
            "<=2 stdev above mean": "antiquewhite",
            "excluded": "black",
        }
        if isinstance(title, type(None)):
            title_prefix = color_cat.split(">")[0]
            title = f"{title_prefix}d distance of query to ref, thresholded"
            if not isinstance(samples_to_exclude, type(None)):
                title = title + f"\n(excluding samples {samples_to_exclude}"
    else:
        raise ValueError(
            "color_by can only be set to 'mean_zscore', 'mean_zscore>2','median_zscore>2', or 'percq_zscore>2'."
        )
    # plot
    sc.pl.umap(
        mapqc_results._adata_ordered,
        color=color_cat,
        size=mapqc_results._adata_ordered.obs.mapqc_dotsize * scale,
        sort_order=False,
        cmap="coolwarm",
        palette=palette,
        title=title,
        vmin=-3,
        vmax=3,
        na_color="grey",
        frameon=False,
    )


def plot_umap_sample_specific_zscored_q_dist_to_ref(
    mapqc_results,
    sample_name: str,
    color_by: Literal["zscore", "zscore>2"] = "zscore",
    size: float = 1,
):
    """
    TO DO: ADD DOCUMENTATION
    """
    if sample_name not in mapqc_results.ref_q_samples_all:
        raise ValueError(
            f"sample_name {sample_name} is not part of the samples in your data."
        )
    if not "X_umap" in mapqc_results.adata.obsm.keys():
        raise ValueError("No UMAP found under .obsm['X_umap'] for this AnnData object.")
    if not mapqc_results._nhood_stats_calculated:
        raise ValueError(
            "Results have not yet been calculated, run mapqc_results.calculate_nhood_statistics first."
        )
    if not mapqc_results._adata_prepped_for_plotting:
        _prepare_adata_for_plotting(mapqc_results)

    # extract sample-specific results:
    sample_zscores = mapqc_results.mean_s2ref_dist_per_nhood_zscored_df.loc[
        mapqc_results.mean_s2ref_dist_per_nhood_zscored_df[mapqc_results.sample_key]
        == sample_name,
        ["nhood", "mean_zscored_dist_to_ref"],
    ]
    # check if results exist:
    if sample_zscores.shape[0] == 0:
        print(
            f"No results were calculated for sample {sample_name}, possibly because it had too few cells or you sampled too few neighborhoods."
        )
        return None
    # add sample-specific results to adata:
    mapqc_results._adata_ordered.obs[f"{sample_name}_zscore"] = np.nan
    mapqc_results._adata_ordered.obs.loc[
        sample_zscores.nhood, f"{sample_name}_zscore"
    ] = sample_zscores.mean_zscored_dist_to_ref.values
    # set size:
    dotsizes = [
        size * 1 if np.isnan(zscore) else size * 30
        for zscore in mapqc_results._adata_ordered.obs[f"{sample_name}_zscore"]
    ]
    # if thresholded color_by, add this to adata:
    palette = None
    if color_by == "zscore>2":
        mapqc_results._adata_ordered.obs[f"{sample_name}_zscore>2"] = [
            ">2 stdev above mean" if zscore > 2 else "<=2 stdev above mean"
            for zscore in mapqc_results._adata_ordered.obs[f"{sample_name}_zscore"]
        ]
        mapqc_results._adata_ordered.obs.loc[
            np.isnan(mapqc_results._adata_ordered.obs[f"{sample_name}_zscore"]),
            f"{sample_name}_zscore>2",
        ] = np.nan
        palette = {">2 stdev above mean": "red", "<=2 stdev above mean": "antiquewhite"}
    # plot UMAP
    sc.pl.umap(
        mapqc_results._adata_ordered,
        color=f"{sample_name}_{color_by}",
        frameon=False,
        size=dotsizes,
        cmap="Reds",
        vmin=0,
        palette=palette,
    )


def plot_umap_neighborhood(
    mapqc_results: mapQC_results, neighborhood: str, title: str = None, scale: int = 1, color_by=None
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

    color_by 
        Color by a specific adata.obs. If left to None, neighborhood cells will be 
        plotted in black

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
    nhood_cells = mapqc_results.nhood_cells[nhood_idx[0]]
    nhood_cells = nhood_cells[~np.isnan(nhood_cells)].astype(int)
    mapqc_results.adata.obs[f"mapqc_nhood_cells_{neighborhood}"].iloc[
            nhood_cells
        ] = neighborhood
    
    if isinstance(title, type(None)):
        title = f"Neighborhood (k={mapqc_results.k})"
    if pd.isnull(color_by):
        
        palette = {"Other": "lightgrey", neighborhood: "black"}
    elif color_by not in mapqc_results.adata.obs.columns:
        raise ValueError("color_by argument has to be set to an existing .obs column name.")
    else:
        # set "Other" to np.nan so that color is automatically set to grey,
        # and cells from neighborhood to their value in the color_by column
        mapqc_results.adata.obs[f"mapqc_nhood_cells_{neighborhood}"] = [
            col if val == neighborhood else np.nan for col, val in zip(
                mapqc_results.adata.obs[color_by],
                mapqc_results.adata.obs[f"mapqc_nhood_cells_{neighborhood}"])
        ]
        palette = None
    dotsizes = [scale if ((val == "Other") or pd.isnull(val)) else scale * 6 for val in mapqc_results.adata.obs[f"mapqc_nhood_cells_{neighborhood}"]]
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
        # remove nans and set to integers instead of floats:
        nhood_cells = nhood_cells[~np.isnan(nhood_cells)].astype(int)
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
    ][mapqc_results.sample_key].unique()
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
        ][mapqc_results.sample_key].unique()
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
        color=mapqc_results.sample_key,
        palette=sample_to_color,
        legend_loc=legend_loc,
        title=title,
        frameon=False,
        size=dotsize,
    )


def plot_umap_cell_level_results(
    mapqc_results,
    size=None,
    sample_based=True,
    color_by: Literal["zscore", "zscore>2"] = "zscore>2",
    vmin=None,
    vmax=None,
    title=None
):
    """
    TO DO: add documentation
    TO DO: give specific color to filtered out or unsampled query cells
    """
    if color_by not in ["zscore", "zscore>2"]:
        raise ValueError(
            "color_by argument has to be set either to 'zscore' or 'zscore>2'."
        )
    if sample_based:
        if not mapqc_results._adata_prepped_for_cell_level_res_plotting_sample_based:
            _prepare_adata_for_cell_level_res_plotting(
                mapqc_results, sample_based=sample_based
            )
        if color_by == "zscore>2":
            color = "mapqc_per_cell_mean_zscored_dist_to_ref>2_sample_based"
            sort_order = False # was set to True before
        else:
            color = "mapqc_per_cell_mean_zscored_dist_to_ref_sample_based"
            sort_order = False
    else:
        if (
            not mapqc_results._adata_prepped_for_cell_level_res_plotting_nhood_only_based
        ):
            _prepare_adata_for_cell_level_res_plotting(
                mapqc_results, sample_based=sample_based
            )
        if color_by == "zscore>2":
            color = f"mapqc_per_cell_perc{mapqc_results.q}_zscored_q_dist_to_ref>2_nhood_based"
            sort_order = True
        else:
            color = f"mapqc_per_cell_perc{mapqc_results.q}_zscored_q_dist_to_ref_nhood_based"
            sort_order = False
    # TO DO: FOR ZSCORE, ORDER CELLS SO THAT NANS ARE PLOTTED FIRST/AT THE BOTTOM
    sc.pl.umap(
        mapqc_results._adata_ordered,
        color=color,
        palette={">2": "red", "<=2": "antiquewhite", "filtered out":"black"},
        frameon=False,
        na_color="gray",
        size=size,
        cmap="Reds",
        vmin=vmin,
        vmax=vmax,
        sort_order=sort_order,
        title=title
    )


def plot_sample_distances_to_reference(mapqc_results):
    """
    TO DO: add documentation
    This is at nhood level
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(1, 1, 1)
    per_sample_meandists_query = mapqc_results.mean_s2ref_dist_per_nhood_zscored_df.loc[
        mapqc_results.mean_s2ref_dist_per_nhood_zscored_df.sample_r_or_q
        == mapqc_results.query_cat,
        :,
    ]
    sample_medians = (
        per_sample_meandists_query.groupby("sample")
        .agg({"mean_zscored_dist_to_ref": "median"})
        .sort_values(by="mean_zscored_dist_to_ref")
    )
    palette = {
        sample: "grey" if zscore < 2 else "red"
        for sample, zscore in zip(
            sample_medians.index, sample_medians.mean_zscored_dist_to_ref
        )
    }
    sns.boxplot(
        per_sample_meandists_query,
        x="sample",
        y="mean_zscored_dist_to_ref",
        palette=palette,
        order=sample_medians.index,
        ax=ax,
    )
    sns.stripplot(
        per_sample_meandists_query,
        x="sample",
        y="mean_zscored_dist_to_ref",
        color="black",
        order=sample_medians.index,
        size=1,
        ax=ax,
    )
    ax.tick_params(axis="x", rotation=90)
    plt.axhline(y=2, linestyle="--")
    plt.tight_layout()


def plot_sample_distances_to_reference_cell_level(mapqc_results):
    """
    TO DO: add documentation
    This is at cell level
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(1, 1, 1)
    # TO DO: ADD CONDITIONAL HERE TO CHECK IF CELL LEVEL RESULTS WERE EVEN CALCULATED
    cell_level_results = mapqc_results.cell_results_nhood_sample_based.copy()
    cell_level_results["sample"] = mapqc_results.adata.obs.loc[
        cell_level_results.index, mapqc_results.sample_key
    ]
    sample_medians = (
        cell_level_results.groupby("sample")
        .agg({"mean_zscored_dist_to_ref": "median"})
        .sort_values(by="mean_zscored_dist_to_ref")
    ).dropna(axis=0, how="all")
    palette = {
        sample: "grey" if zscore < 2 else "red"
        for sample, zscore in zip(
            sample_medians.index, sample_medians.mean_zscored_dist_to_ref
        )
    }
    sns.boxplot(
        cell_level_results,
        x="sample",
        y="mean_zscored_dist_to_ref",
        palette=palette,
        order=sample_medians.index,
        ax=ax,
    )
    sns.stripplot(
        cell_level_results,
        x="sample",
        y="mean_zscored_dist_to_ref",
        color="black",
        order=sample_medians.index,
        size=1,
        ax=ax,
    )
    ax.tick_params(axis="x", rotation=90)
    ax.set_ylabel("cell mean zscored\ndist to ref")
    plt.axhline(y=2, linestyle="--")
    plt.tight_layout()


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
    for stat_to_copy in [
        "nhood_weight",
        "pval_corr",
        "mean_zscored_q_dist_to_ref",
        "mean_zscore>2",
        "median_zscore>2",
        f"perc{mapqc_results.q}_zscore>2",
    ]:
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
    # include info on which neighborhoods were excluded in other columns too, and
    # convert boolean too informative string:
    for stat in ["mean_zscore>2", "median_zscore>2", f"perc{mapqc_results.q}_zscore>2"]:
        bool_to_string = {True: ">2 stdev above mean", False: "<=2 stdev above mean"}
        mapqc_results.adata.obs["mapqc_" + stat] = mapqc_results.adata.obs[
            "mapqc_" + stat
        ].map(bool_to_string)
        mapqc_results.adata.obs.loc[filtered_nhoods, "mapqc_" + stat] = "excluded"
    # store filtering info:
    mapqc_results.adata.obs["mapqc_filter_info"] = np.nan
    mapqc_results.adata.obs.loc[
        mapqc_results._nhood_filtered.index, "mapqc_filter_info"
    ] = mapqc_results._nhood_filtered
    # set sizes for plotting:
    mapqc_results.adata.obs["mapqc_dotsize"] = scale * 1
    mapqc_results.adata.obs.loc[mapqc_results.center_cells, "mapqc_dotsize"] = (
        scale * 30
    )
    # set sizes for plotting, but proportional to neighborhood weight now
    mapqc_results.adata.obs[
        "mapqc_dotsize_by_nhood_weight"
    ] = mapqc_results.adata.obs.mapqc_dotsize
    mapqc_results.adata.obs.loc[
        mapqc_results.nhood_stats.index, "mapqc_dotsize_by_nhood_weight"
    ] = (
        mapqc_results.adata.obs.loc[
            mapqc_results.nhood_stats.index, "mapqc_dotsize_by_nhood_weight"
        ]
        * mapqc_results.nhood_stats.nhood_weight
    )
    # order cells so that index cells are plotted on top, then query, then
    # reference
    cells_ordered = (
        mapqc_results.adata.obs.index[
            mapqc_results.adata.obs[mapqc_results.ref_q_key] == mapqc_results.ref_cat
        ].tolist()
        + [
            cell
            for cell, r_or_q in zip(
                mapqc_results.adata.obs.index,
                mapqc_results.adata.obs[mapqc_results.ref_q_key],
            )
            if (
                (r_or_q == mapqc_results.query_cat)
                and (cell not in mapqc_results.center_cells)
            )
        ]
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


def _prepare_adata_for_cell_level_res_plotting(mapqc_results, sample_based=True):
    if sample_based:
        if not mapqc_results._cell_level_stats_sample_based_calculated:
            raise ValueError(
                "Cell-level stats need to be calculated first; run results.calculate_cell_statistics_using_sample_level_results_per_nhood()."
            )
        mapqc_results.adata.obs[
            "mapqc_per_cell_mean_zscored_dist_to_ref_sample_based"
        ] = np.nan
        mapqc_results.adata.obs.loc[
            mapqc_results.cell_results_nhood_sample_based.index,
            "mapqc_per_cell_mean_zscored_dist_to_ref_sample_based",
        ] = mapqc_results.cell_results_nhood_sample_based.mean_zscored_dist_to_ref
        mapqc_results.adata.obs[
            "mapqc_per_cell_mean_zscored_dist_to_ref>2_sample_based"
        ] = np.nan
        mapqc_results.adata.obs.loc[
            mapqc_results.cell_results_nhood_sample_based.index,
            "mapqc_per_cell_mean_zscored_dist_to_ref>2_sample_based",
        ] = mapqc_results.cell_results_nhood_sample_based["mean_zscored_dist_to_ref>2"]
    else:
        if not mapqc_results._cell_level_stats_nhood_only_based_calculated:
            raise ValueError(
                "Cell-level stats need to be calculated first; run results.calculate_cell_statistics_from_nhoods_using_nhood_stats()."
            )
        for stat in [
            f"mapqc_per_cell_perc{mapqc_results.q}_zscored_q_dist_to_ref>2_nhood_based",
            f"mapqc_per_cell_perc{mapqc_results.q}_zscored_q_dist_to_ref_nhood_based",
        ]:
            mapqc_results.adata.obs[stat] = np.nan
            mapqc_results.adata.obs.loc[
                mapqc_results.cell_results_nhood_based.index, stat
            ] = mapqc_results.cell_results_nhood_based[stat]
    # store ordered version of adata for plotting:
    cells_ordered = (
        mapqc_results.adata.obs.index[
            mapqc_results.adata.obs[mapqc_results.ref_q_key] == mapqc_results.ref_cat
        ].tolist()
        + [
            cell
            for cell, r_or_q in zip(
                mapqc_results.adata.obs.index,
                mapqc_results.adata.obs[mapqc_results.ref_q_key],
            )
            if (
                (r_or_q == mapqc_results.query_cat)
                and (cell not in mapqc_results.center_cells)
            )
        ]
        + mapqc_results.center_cells.tolist()
    )
    # store ordered version of adata
    mapqc_results._adata_ordered = mapqc_results.adata[cells_ordered, :].copy()
    # store that results have been calculated successfully:
    if sample_based:
        mapqc_results._adata_prepped_for_cell_level_res_plotting_sample_based = True
    else:
        mapqc_results._adata_prepped_for_cell_level_res_plotting_nhood_only_based = True
