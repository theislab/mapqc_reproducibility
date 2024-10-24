import pandas as pd
import numpy as np
import random
import scanpy as sc
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import rgb2hex
import matplotlib.colors as mcolors
import seaborn as sns
from ..mapqc import MapqcResults
from typing import Literal


def boxplot_sample_scores_per_nhood(
    mapqc_results: MapqcResults,
    use_zscored_dists: bool = True,
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
    mapqc_results
        Instance of mapQC results class for which sample distance to reference samples
        have already been calculated using mapqc_results.calculate_sample_dists_to_ref_samples().
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
        plotting_df = mapqc_results.sample_stats_per_nhood_df
        y = "mapqc_score"
        ylabel = "Sample mapQC score"
    else:
        plotting_df = mapqc_results.sample_stats_per_nhood_df
        y = "mean_dist_to_ref"
        ylabel = "Sample mean absolute distance to reference"
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
            .sort_values(by=f"median_mapqc_score", ascending=False)
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
        fliersize=1,
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
    nhood_numbers = mapqc_results.nhood_stats.loc[nhood_order, "nhood_number"]
    nhood_zscores = round(
        mapqc_results.nhood_stats.loc[nhood_order, f"median_mapqc_score"],
        2,
    )
    ax.set_xticklabels(
        [f"{num} ({score})" for num, score in zip(nhood_numbers, nhood_zscores)]
    )
    ax.set_xlabel(f"Neighborhood number (median query sample mapqcscore)")
    ax.set_ylabel(ylabel)
    # else:
    #     ax.set_xticklabels(
    #         [
    #             round(mapqc_results.nhood_stats.loc[nhood, "pval_corr"], 4)
    #             for nhood in nhood_order
    #         ]
    #     )
    #     ax.set_xlabel("nhood (corr. pval)")
    ax.tick_params(axis="x", rotation=90)


def umap_significance(
    mapqc_results: MapqcResults,
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
        Instance of mapqc_results class, for which the adata has a UMAP pre-calculated
        and for which mapQC statistics have been calculated using
        mapqc_results.calculate_nhood_statistics
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
        color="nhood_mapqc_sign",
        size=scale * mapqc_results._adata_ordered.obs[size_col],
        palette=palette,
        sort_order=False,
        title=title,
        na_color="darkgrey",
        frameon=False,
    )


def umap_filtering_info(
    mapqc_results: MapqcResults, title: str = "neighborhood filtering"
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
        Instance of mapqc_results class, for which the adata has a UMAP pre-calculated
        and for which mapQC statistics have been calculated using
        mapqc_results.calculate_nhood_statistics

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
        "pass": "tab:green",
        "too few reference samples from different batches": "salmon",
        "too few reference samples": "tab:blue",
        "too few query samples": "black",
    }
    # plot
    sc.pl.umap(
        mapqc_results._adata_ordered,
        color="nhood_filter_info",
        size=mapqc_results._adata_ordered.obs.mapqc_dotsize,
        palette=palette,
        sort_order=False,
        title=title,
        na_color="darkgrey",
        frameon=False,
    )


def umap_zscored_q_dist_to_ref(
    mapqc_results: MapqcResults,
    color_by: Literal[
        "mapqc_score",
        "mapqc_score>2",
        "median_zscore>2",
        "percq_zscore>2",
        "mean_t_stat",
    ] = "mapqc_score",
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
        Instance of mapqc_results class, for which the adata has a UMAP pre-calculated
        and for which mapQC statistics have been calculated using
        mapqc_results.calculate_nhood_statistics

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
    if color_by == "mapqc_score":
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
    elif color_by in [
        "mapqc_score>2",
        "median_zscore>2",
        "percq_zscore>2",
        "mean_t_stat",
    ]:
        if isinstance(samples_to_exclude, type(None)):
            if color_by == "percq_zscore>2":
                color_cat = f"perc{mapqc_results.q}_zscore>2"
            else:
                color_cat = color_by
        else:
            color_cat = (
                f"mapqc_{color_by}_excluding_samples_{samples_to_exclude_string}"
            )
            # recalculate metric excluding samples in samples_to_exclude list
            if color_by == "mapqc_score>2":
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
            title = f"{title_prefix} distance of query to ref, thresholded"
            if not isinstance(samples_to_exclude, type(None)):
                title = title + f"\n(excluding samples {samples_to_exclude}"
    else:
        raise ValueError(
            "color_by can only be set to 'mapqc_score', 'mapqc_score>2','median_zscore>2', 'mean_t_stat', or 'percq_zscore>2'."
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


def umap_sample_specific_zscored_q_dist_to_ref(
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
    sample_zscores = mapqc_results.sample_stats_per_nhood_df.loc[
        mapqc_results.sample_stats_per_nhood_df[mapqc_results.sample_key]
        == sample_name,
        ["nhood", "mapqc_score"],
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
    ] = sample_zscores.mapqc_score.values
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


def umap_neighborhood(
    mapqc_results: MapqcResults,
    neighborhood: str,
    title: str = None,
    scale: int = 1,
    color_by=None,
):
    """
    Plot UMAP of adata in mapqc_results instance, coloring the cells
    from a specific neighborhood

    Parameters
    ---------
    mapqc_results
        Instance of mapqc_results class, for which the adata has a UMAP pre-calculated
        and for which mapQC statistics have been calculated using
        mapqc_results.calculate_nhood_statistics

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
        raise ValueError(
            "color_by argument has to be set to an existing .obs column name."
        )
    else:
        # set "Other" to np.nan so that color is automatically set to grey,
        # and cells from neighborhood to their value in the color_by column
        mapqc_results.adata.obs[f"mapqc_nhood_cells_{neighborhood}"] = [
            col if val == neighborhood else np.nan
            for col, val in zip(
                mapqc_results.adata.obs[color_by],
                mapqc_results.adata.obs[f"mapqc_nhood_cells_{neighborhood}"],
            )
        ]
        palette = None
    dotsizes = [
        scale if ((val == "Other") or pd.isnull(val)) else scale * 6
        for val in mapqc_results.adata.obs[f"mapqc_nhood_cells_{neighborhood}"]
    ]
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


def umap_q_ref_sample_shaded(
    mapqc_results: MapqcResults,
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


def umap_mapqc_scores(
    mapqc_results,
    size=None,
    sample_based: bool = True,
    color_by: Literal[
        "mapqc_score", "mapqc_score>2", "mean_pval_corr", "mean_pval_corr_significant"
    ] = "mapqc_score",
    vmin=None,
    vmax=None,
    title=None,
    cmap_name=None,
    figsize=(5, 5),
):
    """
    TO DO: add documentation
    TO DO: give specific color to filtered out or unsampled query cells
    """
    if color_by not in [
        "mapqc_score",
        "mapqc_score>2",
        "mean_pval_corr",
        "mean_pval_corr_significant",
    ]:
        raise ValueError(
            "color_by has to be set to mapqc_score, mapqc_score>2, mean_pval_corr, or mean_pval_corr_significant."
        )
    if sample_based:
        if not mapqc_results._adata_prepped_for_cell_level_res_plotting_sample_based:
            _prepare_adata_for_cell_level_res_plotting(
                mapqc_results, sample_based=sample_based
            )
    else:
        raise ValueError("nhood based is deprecated.")
        # if (
        #     not mapqc_results._adata_prepped_for_cell_level_res_plotting_nhood_only_based
        # ):
        #     _prepare_adata_for_cell_level_res_plotting(
        #         mapqc_results, sample_based=sample_based
        #     )
        # if binarized:
        #     color = f"mapqc_per_cell_perc{mapqc_results.q}_zscored_q_dist_to_ref>2_nhood_based"
        #     sort_order = True
        # else:
        #     color = f"mapqc_per_cell_perc{mapqc_results.q}_zscored_q_dist_to_ref_nhood_based"
        #     sort_order = False
    # create color sequence, such that each cell has a color, and such that
    # query cells are colored based on their mapQC score with a gradient,
    # query cells that are filtered out or not sampled are colored black,
    # and reference cells are colored grey
    # TO DO: FOR ZSCORE, ORDER CELLS SO THAT NANS ARE PLOTTED FIRST/AT THE BOTTOM
    if color_by == "mapqc_score>2":
        # standard scanpy umap suffices
        sc.pl.umap(
            mapqc_results._adata_ordered,
            color=color_by,
            palette={
                ">2": "firebrick",
                "<=2": "cornflowerblue",
                "filtered out": "black",
                "not sampled": "navajowhite",
            },
            frameon=False,
            na_color="gray",
            size=size,
            # sort_order=sort_order,
            title=title,
        )
    elif color_by == "mean_pval_corr_significant":
        sc.pl.umap(
            mapqc_results._adata_ordered,
            color=color_by,
            palette={
                "significant": "firebrick",
                "not significant": "cornflowerblue",
                "filtered out": "black",
                "not sampled": "navajowhite",
            },
            frameon=False,
            na_color="gray",
            size=size,
            # sort_order=sort_order,
            title=title,
        )
    else:
        if vmin is None or vmax is None:
            if color_by == "mean_pval_corr":
                vmin = 0
                vmax = 1
            elif color_by == "mapqc_score":
                vmin = -4
                vmax = 4
        # need to manually change colors, as we have a combination of
        # categoricals (reference grey, nan black) and gradient (mapQC scores)
        fig, ax = plt.subplots(figsize=figsize)
        sc.pl.umap(
            mapqc_results._adata_ordered,
            color=None,
            frameon=False,
            size=size,
            # sort_order=sort_order,
            title=title,
            ax=ax,
            show=False,
        )
        if cmap_name is None:
            if color_by == "mean_pval_corr":
                cmap_name = "Reds_r"
            elif color_by == "mapqc_score":
                cmap_name = "coolwarm"
        colors = _translate_to_colors_rgba(
            point_color_values=mapqc_results._adata_ordered.obs[color_by],
            point_ref_q_values=mapqc_results._adata_ordered.obs[
                mapqc_results.ref_q_key
            ],
            vmin=vmin,
            vmax=vmax,
            cmap_name=cmap_name,
        )
        scatter = ax.collections[0]
        # Set the colors of the scatter plot
        scatter.set_facecolors(colors)
        scatter.set_edgecolors("none")

        # Add a colorbar
        sm = plt.cm.ScalarMappable(
            cmap=plt.get_cmap(cmap_name), norm=plt.Normalize(vmin=vmin, vmax=vmax)
        )
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, fraction=0.04, aspect=30)
        if color_by == "mean_pval_corr":
            label_prefix = "Multiple-testing-corrected p-value"
        elif color_by == "mapqc_score":
            label_prefix = "MapQC score"
        cbar.set_label(
            f"{label_prefix}\n(black: not sampled, or filtered out)", fontsize=10
        )
        plt.show()


# Already covered by boxplot_mapqc_scores_per_category
# def boxplot_sample_distances_to_reference(mapqc_results):
#     """
#     TO DO: add documentation
#     This is at nhood level
#     """
#     fig = plt.figure(figsize=(10, 8))
#     ax = fig.add_subplot(1, 1, 1)
#     per_sample_meandists_query = mapqc_results.sample_stats_per_nhood_df.loc[ # mean_s2ref_dist_per_nhood_zscored_df
#         mapqc_results.sample_stats_per_nhood_df.sample_r_or_q
#         == mapqc_results.query_cat,
#         :,
#     ]
#     sample_medians = (
#         per_sample_meandists_query.groupby("sample")
#         .agg({"mapqc_score": "median"})
#         .sort_values(by="mapqc_score")
#     )
#     palette = {
#         sample: "grey" if zscore < 2 else "red"
#         for sample, zscore in zip(sample_medians.index, sample_medians.mapqc_score)
#     }
#     sns.boxplot(
#         per_sample_meandists_query,
#         x="sample",
#         y="mapqc_score",
#         hue="sample",
#         palette=palette,
#         order=sample_medians.index,
#         ax=ax,
#         fliersize=1
#     )
#     sns.stripplot(
#         per_sample_meandists_query,
#         x="sample",
#         y="mapqc_score",
#         color="black",
#         order=sample_medians.index,
#         size=1,
#         ax=ax,
#     )
#     ax.tick_params(axis="x", rotation=90)
#     ax.set_ylabel("MapQC score per neighborhood")
#     ax.set_xlabel("Sample")
#     plt.axhline(y=2, linestyle="--")
#     plt.tight_layout()


def boxplot_mapqc_scores_per_category(
    mapqc_results,
    box_category: str,
    color_category: str = None,
    palette_stripplot: str = "tab10",
    min_cells_per_cat: int = 10,
):
    """
    TO DO: add documentation
    This is at cell level
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(1, 1, 1)
    # check that category is in adata.obs columns:
    categories_to_include = [box_category]
    if pd.isnull(color_category):
        palette_stripplot = None
    else:
        categories_to_include.append(color_category)
    for cat in categories_to_include:
        if cat not in mapqc_results.adata.obs.columns:
            raise ValueError(
                f"{cat} is not part of the .obs columns of this AnnData object."
            )
    # check that the category has values for the query cells:
    for cat in categories_to_include:
        if pd.isnull(
            mapqc_results.adata.obs.loc[
                mapqc_results.adata.obs[mapqc_results.ref_q_key]
                == mapqc_results.query_cat,
                cat,
            ]
        ).all():
            raise ValueError(f"None of the query cells have values for category {cat}.")
    cell_level_results = mapqc_results.cell_results_nhood_sample_based.copy()
    cell_level_results[box_category] = mapqc_results.adata.obs.loc[
        cell_level_results.index, box_category
    ].tolist()
    cell_level_results.dropna(axis=0, how="any", inplace=True)
    # remove categories with too few cells:
    category_counts = cell_level_results[box_category].value_counts()
    categories_to_keep = category_counts[category_counts >= min_cells_per_cat].index
    cell_level_results = cell_level_results[
        cell_level_results[box_category].isin(categories_to_keep)
    ]
    # TO DO: CHECK HOW NAS ARE DEALT WITH IN ALL OF THESE PLOTS (what is done after the above line?)
    category_medians = (
        cell_level_results.groupby(box_category)
        .agg({"mapqc_score": "median"})
        .sort_values(by="mapqc_score")
    ).dropna(axis=0, how="all")
    palette_boxes = {
        sample: "grey" if zscore < 2 else "red"
        for sample, zscore in zip(category_medians.index, category_medians.mapqc_score)
    }
    if pd.isnull(color_category):
        stripplot_hue = None
    else:
        stripplot_hue = color_category
        cell_level_results[color_category] = mapqc_results.adata.obs.loc[
            cell_level_results.index, color_category
        ].tolist()
    sns.boxplot(
        cell_level_results,
        x=box_category,
        y="mapqc_score",
        hue=box_category,  # this is only so that we can color the boxes > 2 red
        palette=palette_boxes,
        order=category_medians.index,
        ax=ax,
        fliersize=1,
    )
    sns.stripplot(
        cell_level_results,
        x=box_category,
        y="mapqc_score",
        color="black",
        order=category_medians.index,
        hue=stripplot_hue,
        size=1,
        ax=ax,
        palette=palette_stripplot,
    )
    ax.tick_params(axis="x", rotation=90)
    ax.set_ylabel("Cell mapQC scores")
    ax.set_xlabel(box_category)
    plt.axhline(y=2, linestyle="--")
    if not pd.isnull(color_category):
        leg = plt.legend(loc=(1.01, 0.4), frameon=False, title=color_category)
        leg._legend_box.align = "left"
    plt.tight_layout()


def grouped_boxplot_mapqc_scores_per_category(
    mapqc_results_ctr,
    mapqc_results_case,
    box_category: str,
    group_category: str,
    color_category: str = None,
    palette=None,
    min_cells_per_cat: int = 10,
):
    """
    TO DO: add documentation
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(1, 1, 1)
    # check that category is in adata.obs columns:
    categories_to_include = [box_category, group_category]
    if pd.isnull(color_category):
        palette_stripplot = None
    else:
        categories_to_include.append(color_category)
    for cat in categories_to_include:
        if (
            cat not in mapqc_results_ctr.adata.obs.columns
            and cat not in mapqc_results_case.adata.obs.columns
        ):
            raise ValueError(
                f"{cat} is not part of the .obs columns of either of your Results AnnData objects."
            )
    # check that the category has values for the query cells:
    for cat in categories_to_include:
        if (
            pd.isnull(
                mapqc_results_ctr.adata.obs.loc[
                    mapqc_results_ctr.adata.obs[mapqc_results_ctr.ref_q_key]
                    == mapqc_results_ctr.query_cat,
                    cat,
                ]
            ).all()
            and pd.isnull(
                mapqc_results_case.adata.obs.loc[
                    mapqc_results_case.adata.obs[mapqc_results_case.ref_q_key]
                    == mapqc_results_case.query_cat,
                    cat,
                ]
            ).all()
        ):
            raise ValueError(f"None of the query cells have values for category {cat}.")
    # first collect results for query control:
    cell_level_results_ctr = mapqc_results_ctr.cell_results_nhood_sample_based.copy()
    cell_level_results_ctr[categories_to_include] = mapqc_results_ctr.adata.obs[
        categories_to_include
    ].loc[cell_level_results_ctr.index]

    # convert to list so that we can append new values with new categories
    for cat in categories_to_include:
        cell_level_results_ctr[cat] = cell_level_results_ctr[cat].tolist()
    # then concatenate results for query case:
    cell_level_results_case = mapqc_results_case.cell_results_nhood_sample_based.copy()
    cell_level_results_case[categories_to_include] = mapqc_results_case.adata.obs[
        categories_to_include
    ]
    # concatenate results for control and case:
    cell_level_results = pd.concat([cell_level_results_ctr, cell_level_results_case])
    # remove cells without results
    cell_level_results.dropna(axis=0, how="any", inplace=True)
    # remove categories with too few cells:
    category_counts = cell_level_results[box_category].value_counts()
    categories_to_keep = category_counts[category_counts >= min_cells_per_cat].index
    cell_level_results = cell_level_results[
        cell_level_results[box_category].isin(categories_to_keep)
    ]
    # TO DO: CHECK HOW NAS ARE DEALT WITH IN ALL OF THESE PLOTS (what is done after the above line?)
    # order by median score of first group from group_category in results case:
    first_group_cat = cell_level_results_case[group_category].unique()[0]
    category_medians = (
        cell_level_results.loc[cell_level_results[group_category] == first_group_cat, :]
        .groupby(box_category)
        .agg({"mapqc_score": "median"})
        .sort_values(by="mapqc_score")
    ).dropna(axis=0, how="all")
    # palette_boxes = {
    #     sample: "grey" if zscore < 2 else "red"
    #     for sample, zscore in zip(
    #         category_medians.index, category_medians.mapqc_score
    #     )
    # }
    if pd.isnull(color_category):
        stripplot_hue = None
    else:
        stripplot_hue = color_category
    sns.boxplot(
        cell_level_results,
        x=box_category,
        y="mapqc_score",
        hue=group_category,
        palette=palette,
        order=category_medians.index,
        ax=ax,
        fliersize=0.5,
    )
    sns.stripplot(
        cell_level_results,
        x=box_category,
        y="mapqc_score",
        order=category_medians.index,
        hue=group_category,
        dodge=True,
        palette="dark:black",
        size=1,
        ax=ax,
    )
    ax.tick_params(axis="x", rotation=90)
    ax.set_ylabel("Cell mapQC scores")
    ax.set_xlabel(box_category)
    plt.axhline(y=2, linestyle="--")
    leg = plt.legend(loc=(1.05, 0.7), frameon=False, title=color_category)
    leg._legend_box.align = "right"
    plt.tight_layout()


def _translate_to_colors_rgba(
    point_color_values,
    point_ref_q_values,
    vmin=-4,
    vmax=4,
    cmap_name="coolwarm",
):
    cmap = plt.get_cmap(cmap_name)
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    def value_to_color(value, r_or_q):
        if r_or_q == "r":
            return (0.5, 0.5, 0.5, 1.0)  # Grey for reference
        elif value is None or (isinstance(value, float) and np.isnan(value)):
            return (0, 0, 0, 1)  # Black for None or np.nan
        else:
            return cmap(norm(value))

    colors = np.array(
        [
            value_to_color(point, r_or_q)
            for point, r_or_q in zip(point_color_values, point_ref_q_values)
        ]
    )
    return colors
