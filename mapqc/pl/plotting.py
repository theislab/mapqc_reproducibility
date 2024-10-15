import pandas as pd
import numpy as np
import random
import scanpy as sc
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import rgb2hex
import seaborn as sns
from ..mapqc import MapqcResults
from typing import Literal


def boxplot_sample_scores_per_nhood(
    MapqcResults: MapqcResults,
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
    MapqcResults
        Instance of mapQC results class for which sample distance to reference samples
        have already been calculated using MapqcResults.calculate_sample_dists_to_ref_samples().
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
    if not MapqcResults._sample_dists_to_ref_calculated:
        return ValueError(
            "Plot can only be generated after running calculate_sample_dists_to_ref_samples()."
        )
    # check if neighborhood is in center cells if only results for a specific neighborhood
    # should be shown:
    if not isinstance(neighborhood, type(None)):
        if not neighborhood in MapqcResults.center_cells:
            raise ValueError("Your neighborhood is not part of the center cells.")
    # if not MapqcResults._results_prepped_for_boxplot:
    # _convert_nhood_dists_to_plottable_dataframe(MapqcResults)
    if use_zscored_dists:
        plotting_df = MapqcResults.mean_s2ref_dist_per_nhood_zscored_df
        y = "mapQC_score"
        ylabel = "Sample mapQC score"
    else:
        plotting_df = MapqcResults.mean_s2ref_dist_per_nhood_df
        y = "mean_dist_to_ref"
        ylabel = "Sample mean absolute distance to reference"
    # check if only results for a specific neighborhood should be plotted and subset if so:
    if not isinstance(neighborhood, type(None)):
        plotting_df = plotting_df.loc[plotting_df.nhood == neighborhood, :].copy()
        nhood_order = [neighborhood]
        figsize = (1, 4)
    else:
        # subset to lower number of neighborhoods if needed:
        if MapqcResults.nhood_stats.shape[0] > max_neighborhoods_to_plot:
            random.seed(seed)
            random_nhoods = random.sample(
                MapqcResults.nhood_stats.index.tolist(), k=max_neighborhoods_to_plot
            )
        else:
            random_nhoods = MapqcResults.nhood_stats.index.tolist()
        # if use_zscored_dists:
        #     # nhoods_ordered_by_zscore = MapqcResults.nhood_stats.sort_values(by='median_zscored_q_dist_to_ref',ascending=False).index
        #     # nhood_order = [nhood for nhood in nhoods_ordered_by_zscore if nhood in random_nhoods]
        nhood_order = (
            MapqcResults.nhood_stats.loc[random_nhoods, :]
            .sort_values(
                by=f"perc{MapqcResults.q}_zscored_q_dist_to_ref", ascending=False
            )
            .index.tolist()
        )
        # else:
        # nhood_order = (
        #     MapqcResults.nhood_stats.loc[random_nhoods, :]
        #     .sort_values(by="pval_corr")
        #     .index.tolist()
        # )
        figsize = (24, 8)
    hue = "sample_r_or_q"
    # palette
    palette = {MapqcResults.ref_cat: "tab:orange", MapqcResults.query_cat: "tab:blue"}
    # palette = {MapqcResults.ref_cat:"lightsteelblue", MapqcResults.query_cat:"blue"}
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
    nhood_numbers = MapqcResults.nhood_stats.loc[nhood_order, 'nhood_number']
    nhood_zscores = round(MapqcResults.nhood_stats.loc[nhood_order, f'perc{MapqcResults.q}_zscored_q_dist_to_ref'], 2)
    ax.set_xticklabels([f"{num} ({score})" for num, score in zip(nhood_numbers, nhood_zscores)])
    ax.set_xlabel(f"Neighborhood number ({MapqcResults.q}th percentile query score)")
    ax.set_ylabel(ylabel)
    # else:
    #     ax.set_xticklabels(
    #         [
    #             round(MapqcResults.nhood_stats.loc[nhood, "pval_corr"], 4)
    #             for nhood in nhood_order
    #         ]
    #     )
    #     ax.set_xlabel("nhood (corr. pval)")
    ax.tick_params(axis="x", rotation=90)


def umap_significance(
    MapqcResults: MapqcResults,
    title: str = "neighborhood significance",
    scale_size_by_nhood_weight: bool = True,
    scale: float = 1,
):
    """
    Plot UMAP of adata in MapqcResults instance, colored grey for all non-index
    cells (i.e. all cells that were not the center of a tested neighborhood),
    white and large for all index cells that passed filtering and for which
    reference-to-reference sample distances were not smaller than reference-
    to-query sample distances, and red for index cells for which reference-
    to-reference distances were significantly smaller than reference-to-query
    samples. Neighborhoods that did not pass filtering are colored in black.
    Use plot_umap_filtering_info to color by exact reason for exclusion.

    Parameters
    ---------
    MapqcResults
        Instance of MapqcResults class, for which the adata has a UMAP pre-calculated
        and for which mapQC statistics have been calculated using
        MapqcResults.calculate_nhood_statistics
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
    if not "X_umap" in MapqcResults.adata.obsm.keys():
        raise ValueError("No UMAP calculated for this AnnData object.")
    if not MapqcResults._nhood_stats_calculated:
        raise ValueError(
            "Results have not yet been calculated, run MapqcResults.calculate_nhood_statistics first."
        )
    if not MapqcResults._adata_prepped_for_plotting:
        _prepare_adata_for_plotting(MapqcResults)
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
        MapqcResults._adata_ordered,
        color="mapqc_sign",
        size=scale * MapqcResults._adata_ordered.obs[size_col],
        palette=palette,
        sort_order=False,
        title=title,
        na_color="darkgrey",
        frameon=False,
    )


def umap_filtering_info(
    MapqcResults: MapqcResults, title: str = "neighborhood filtering"
):
    """
    Plot UMAP of adata in MapqcResults instance, colored grey for all non-index
    cells (i.e. all cells that were not the center of a tested neighborhood),
    white and large for all index cells that passed filtering, and in several
    different colors for different reasons why a neighborhood didn't pass filtering
    (see legend).

    Parameters
    ---------
    MapqcResults
        Instance of MapqcResults class, for which the adata has a UMAP pre-calculated
        and for which mapQC statistics have been calculated using
        MapqcResults.calculate_nhood_statistics

    title
        Title for UMAP.

    Returns
    -------
    Plots UMAP.

    TO DO: add option to return figure as variable and more plotting options.
    """
    if not "X_umap" in MapqcResults.adata.obsm.keys():
        raise ValueError("No UMAP calculated for this AnnData object.")
    if not MapqcResults._nhood_stats_calculated:
        raise ValueError(
            "Results have not yet been calculated, run MapqcResults.calculate_nhood_statistics first."
        )
    if not MapqcResults._adata_prepped_for_plotting:
        _prepare_adata_for_plotting(MapqcResults)
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
        MapqcResults._adata_ordered,
        color="mapqc_filter_info",
        size=MapqcResults._adata_ordered.obs.mapqc_dotsize,
        palette=palette,
        sort_order=False,
        title=title,
        na_color="darkgrey",
        frameon=False,
    )


def umap_zscored_q_dist_to_ref(
    MapqcResults: MapqcResults,
    color_by: Literal[
        "mapQC_score", "mapQC_score>2", "median_zscore>2", "percq_zscore>2"
    ] = "mapQC_score",
    title: str = None,
    scale: int = 1,
    samples_to_exclude: list = None,
):
    """
    Plot UMAP of adata in MapqcResults instance, coloring neighborhoods
    by the mean zscored query sample distance to reference sample distance
    (zscoring using the mean and standard deviation of the ref sample
    distance to the reference)

    Parameters
    ---------
    MapqcResults
        Instance of MapqcResults class, for which the adata has a UMAP pre-calculated
        and for which mapQC statistics have been calculated using
        MapqcResults.calculate_nhood_statistics

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
    if not "X_umap" in MapqcResults.adata.obsm.keys():
        raise ValueError("No UMAP calculated for this AnnData object.")
    if not MapqcResults._nhood_stats_calculated:
        raise ValueError(
            "Results have not yet been calculated, run MapqcResults.calculate_nhood_statistics first."
        )
    if not MapqcResults._adata_prepped_for_plotting:
        _prepare_adata_for_plotting(MapqcResults)
        sc.set_figure_params(figsize=(5, 5))
    if not isinstance(samples_to_exclude, type(None)):
        # check if samples exist:
        if (
            np.sum(
                [s not in MapqcResults.query_samples_all for s in samples_to_exclude]
            )
            > 0
        ):
            raise ValueError(
                "One or more of the samples in samples_to_exclude are not found among the query samples."
            )
        samples_to_include_boolean = [
            (sample in MapqcResults.query_samples_all)
            and (sample not in samples_to_exclude)
            for sample in MapqcResults.ref_q_samples_all
        ]
        samples_to_exclude_string = "_".join(samples_to_exclude)
    if color_by == "mapQC_score":
        if isinstance(samples_to_exclude, type(None)):
            color_cat = "mapqc_mean_zscored_q_dist_to_ref"
            palette = None
            if isinstance(title, type(None)):
                title = f"Mean Z-scored distance of query samples to reference samples"
        else:
            # recalculate mean zscore, but excluding samples
            mean_zscored_q_dist_to_ref = np.nanmean(
                MapqcResults.mean_sample_dists_to_ref_zscored[
                    samples_to_include_boolean, :
                ],
                axis=0,
            )
            color_cat = (
                f"mapqc_{color_by}_excluding_samples_{samples_to_exclude_string}"
            )
            MapqcResults._adata_ordered.obs[color_cat] = np.nan
            MapqcResults._adata_ordered.obs.loc[
                MapqcResults.center_cells, color_cat
            ] = mean_zscored_q_dist_to_ref
        palette = None
    elif color_by in ["mapQC_score>2", "median_zscore>2", "percq_zscore>2"]:
        if isinstance(samples_to_exclude, type(None)):
            if color_by == "percq_zscore>2":
                color_cat = "mapqc_" + f"perc{MapqcResults.q}_zscore>2"
            else:
                color_cat = "mapqc_" + color_by
        else:
            color_cat = (
                f"mapqc_{color_by}_excluding_samples_{samples_to_exclude_string}"
            )
            # recalculate metric excluding samples in samples_to_exclude list
            if color_by == "mapQC_score>2":
                mean_zscored_q_dist_to_ref = np.nanmean(
                    MapqcResults.mean_sample_dists_to_ref_zscored[
                        samples_to_include_boolean, :
                    ],
                    axis=0,
                )
                MapqcResults._adata_ordered.obs[color_cat] = np.nan
                MapqcResults._adata_ordered.obs.loc[
                    MapqcResults.center_cells, color_cat
                ] = (mean_zscored_q_dist_to_ref > 2)
            elif color_by == "median_zscore>2":
                median_zscored_q_dist_to_ref = np.nanmedian(
                    MapqcResults.mean_sample_dists_to_ref_zscored[
                        samples_to_include_boolean, :
                    ],
                    axis=0,
                )
                MapqcResults._adata_ordered.obs[color_cat] = np.nan
                MapqcResults._adata_ordered.obs.loc[
                    MapqcResults.center_cells, color_cat
                ] = (median_zscored_q_dist_to_ref > 2)
            elif color_by == "percq_zscore>2":
                percq_zscored_q_dist_to_ref = np.nanpercentile(
                    MapqcResults.mean_sample_dists_to_ref_zscored[
                        samples_to_include_boolean, :
                    ],
                    q=MapqcResults.q,
                    axis=0,
                )
                color_cat = f"mapqc_perc{MapqcResults.q}_zscore>2_excluding_samples_{samples_to_exclude_string}"
                MapqcResults._adata_ordered.obs[color_cat] = np.nan
                MapqcResults._adata_ordered.obs.loc[
                    MapqcResults.center_cells, color_cat
                ] = (percq_zscored_q_dist_to_ref > 2)
            # map boolean to strings:
            MapqcResults._adata_ordered.obs[
                color_cat
            ] = MapqcResults._adata_ordered.obs[color_cat].map(
                {True: ">2 stdev above mean", False: "<=2 stdev above mean"}
            )
            # set excluded neighborhoods as such
            MapqcResults._adata_ordered.obs.loc[
                MapqcResults._nhood_filtered.index[
                    MapqcResults._nhood_filtered != "pass"
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
            "color_by can only be set to 'mapQC_score', 'mapQC_score>2','median_zscore>2', or 'percq_zscore>2'."
        )
    # plot
    sc.pl.umap(
        MapqcResults._adata_ordered,
        color=color_cat,
        size=MapqcResults._adata_ordered.obs.mapqc_dotsize * scale,
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
    MapqcResults,
    sample_name: str,
    color_by: Literal["zscore", "zscore>2"] = "zscore",
    size: float = 1,
):
    """
    TO DO: ADD DOCUMENTATION
    """
    if sample_name not in MapqcResults.ref_q_samples_all:
        raise ValueError(
            f"sample_name {sample_name} is not part of the samples in your data."
        )
    if not "X_umap" in MapqcResults.adata.obsm.keys():
        raise ValueError("No UMAP found under .obsm['X_umap'] for this AnnData object.")
    if not MapqcResults._nhood_stats_calculated:
        raise ValueError(
            "Results have not yet been calculated, run MapqcResults.calculate_nhood_statistics first."
        )
    if not MapqcResults._adata_prepped_for_plotting:
        _prepare_adata_for_plotting(MapqcResults)

    # extract sample-specific results:
    sample_zscores = MapqcResults.mean_s2ref_dist_per_nhood_zscored_df.loc[
        MapqcResults.mean_s2ref_dist_per_nhood_zscored_df[MapqcResults.sample_key]
        == sample_name,
        ["nhood", "mapQC_score"],
    ]
    # check if results exist:
    if sample_zscores.shape[0] == 0:
        print(
            f"No results were calculated for sample {sample_name}, possibly because it had too few cells or you sampled too few neighborhoods."
        )
        return None
    # add sample-specific results to adata:
    MapqcResults._adata_ordered.obs[f"{sample_name}_zscore"] = np.nan
    MapqcResults._adata_ordered.obs.loc[
        sample_zscores.nhood, f"{sample_name}_zscore"
    ] = sample_zscores.mapQC_score.values
    # set size:
    dotsizes = [
        size * 1 if np.isnan(zscore) else size * 30
        for zscore in MapqcResults._adata_ordered.obs[f"{sample_name}_zscore"]
    ]
    # if thresholded color_by, add this to adata:
    palette = None
    if color_by == "zscore>2":
        MapqcResults._adata_ordered.obs[f"{sample_name}_zscore>2"] = [
            ">2 stdev above mean" if zscore > 2 else "<=2 stdev above mean"
            for zscore in MapqcResults._adata_ordered.obs[f"{sample_name}_zscore"]
        ]
        MapqcResults._adata_ordered.obs.loc[
            np.isnan(MapqcResults._adata_ordered.obs[f"{sample_name}_zscore"]),
            f"{sample_name}_zscore>2",
        ] = np.nan
        palette = {">2 stdev above mean": "red", "<=2 stdev above mean": "antiquewhite"}
    # plot UMAP
    sc.pl.umap(
        MapqcResults._adata_ordered,
        color=f"{sample_name}_{color_by}",
        frameon=False,
        size=dotsizes,
        cmap="Reds",
        vmin=0,
        palette=palette,
    )


def umap_neighborhood(
    MapqcResults: MapqcResults, neighborhood: str, title: str = None, scale: int = 1, color_by=None
):
    """
    Plot UMAP of adata in MapqcResults instance, coloring the cells
    from a specific neighborhood

    Parameters
    ---------
    MapqcResults
        Instance of MapqcResults class, for which the adata has a UMAP pre-calculated
        and for which mapQC statistics have been calculated using
        MapqcResults.calculate_nhood_statistics

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
    if not "X_umap" in MapqcResults.adata.obsm.keys():
        raise ValueError("No UMAP calculated for this AnnData object.")
    if not MapqcResults._sample_dists_to_ref_calculated:
        raise ValueError(
            "Center cell neighbors have not yet been calculated, run MapqcResults.calculate_sample_dists_to_ref_samples first."
        )
    if not MapqcResults._adata_prepped_for_plotting:
        _prepare_adata_for_plotting(MapqcResults)
        sc.set_figure_params(figsize=(5, 5))
    # add neighborhood info to adata object
    nhood_idx = np.where(MapqcResults.center_cells == neighborhood)[0]
    if len(nhood_idx) == 0:
        raise ValueError(
            f"Neighborhood center cell {neighborhood} not found in adata. Check your neighborhood argument setting for typos."
        )
    MapqcResults.adata.obs[f"mapqc_nhood_cells_{neighborhood}"] = "Other"
    nhood_cells = MapqcResults.nhood_cells[nhood_idx[0]]
    nhood_cells = nhood_cells[~np.isnan(nhood_cells)].astype(int)
    MapqcResults.adata.obs[f"mapqc_nhood_cells_{neighborhood}"].iloc[
            nhood_cells
        ] = neighborhood
    
    if isinstance(title, type(None)):
        title = f"Neighborhood (k={MapqcResults.k})"
    if pd.isnull(color_by):
        
        palette = {"Other": "lightgrey", neighborhood: "black"}
    elif color_by not in MapqcResults.adata.obs.columns:
        raise ValueError("color_by argument has to be set to an existing .obs column name.")
    else:
        # set "Other" to np.nan so that color is automatically set to grey,
        # and cells from neighborhood to their value in the color_by column
        MapqcResults.adata.obs[f"mapqc_nhood_cells_{neighborhood}"] = [
            col if val == neighborhood else np.nan for col, val in zip(
                MapqcResults.adata.obs[color_by],
                MapqcResults.adata.obs[f"mapqc_nhood_cells_{neighborhood}"])
        ]
        palette = None
    dotsizes = [scale if ((val == "Other") or pd.isnull(val)) else scale * 6 for val in MapqcResults.adata.obs[f"mapqc_nhood_cells_{neighborhood}"]]
    # prepare color palette:
    sc.pl.umap(
        MapqcResults.adata,
        color=f"mapqc_nhood_cells_{neighborhood}",
        size=dotsizes,
        palette=palette,
        sort_order=False,
        title=title,
        frameon=False,
    )


def umap_q_ref_sample_shaded(
    MapqcResults: MapqcResults,
    include_legend: bool = False,
    neighborhood: str = None,
    reference_only: bool = False,
    seed: int = None,
    dotsize: int = None,
):
    """
    Plot UMAP of MapqcResults adata, colored by reference versus query, but
    shading those colors by sample (i.e. reference samples have different
    shades of orange, query samples different shades of blue). This helps get
    an impression of how well samples are mixing in different regions of the
    embedding. Can be done for a single neighborhood as well using the
    neighborhood argument.

    Parameters
    ----------
    MapqcResults
        MapqcResults instance
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
        if not neighborhood in MapqcResults.center_cells:
            raise ValueError("Your neighborhood is not part of the center cells.")
        # check if k nearest neighbors were already calculated:
        if not MapqcResults._sample_dists_to_ref_calculated:
            raise ValueError(
                "Center cell neighbors have not yet been calculated, run MapqcResults.calculate_sample_dists_to_ref_samples first."
            )
        nhood_idx = np.where(MapqcResults.center_cells == neighborhood)[0][0]
        nhood_cells = MapqcResults.nhood_cells[nhood_idx, :]
        # remove nans and set to integers instead of floats:
        nhood_cells = nhood_cells[~np.isnan(nhood_cells)].astype(int)
        adata_to_plot = MapqcResults.adata[nhood_cells, :].copy()
        title = f"Sample (query versus reference)\n(nhood {neighborhood}"
    else:
        adata_to_plot = MapqcResults.adata
        title = "Sample (query versus reference)"
    if reference_only:
        adata_to_plot = adata_to_plot[
            adata_to_plot.obs[MapqcResults.ref_q_key] == MapqcResults.ref_cat, :
        ].copy()
    # 1. generate colors for reference samples
    cmap_ref = matplotlib.colormaps["Wistia"]
    ref_samples = adata_to_plot.obs.loc[
        adata_to_plot.obs[MapqcResults.ref_q_key] == MapqcResults.ref_cat, :
    ][MapqcResults.sample_key].unique()
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
            MapqcResults.adata.obs[MapqcResults.ref_q_key] == MapqcResults.query_cat,
            :,
        ][MapqcResults.sample_key].unique()
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
        color=MapqcResults.sample_key,
        palette=sample_to_color,
        legend_loc=legend_loc,
        title=title,
        frameon=False,
        size=dotsize,
    )


def umap_mapQC_scores(
    MapqcResults,
    size=None,
    sample_based: bool =True,
    binarized: bool = False, # Literal["zscore", "zscore>2"] = "zscore>2",
    vmin=None,
    vmax=None,
    title=None
):
    """
    TO DO: add documentation
    TO DO: give specific color to filtered out or unsampled query cells
    """
    if sample_based:
        if not MapqcResults._adata_prepped_for_cell_level_res_plotting_sample_based:
            _prepare_adata_for_cell_level_res_plotting(
                MapqcResults, sample_based=sample_based
            )
        if binarized:
            color = "mapQC_score>2"
            sort_order = False # doesn't seem to matter
        else:
            color = "mapQC_score"
            sort_order = False
    else:
        if (
            not MapqcResults._adata_prepped_for_cell_level_res_plotting_nhood_only_based
        ):
            _prepare_adata_for_cell_level_res_plotting(
                MapqcResults, sample_based=sample_based
            )
        if binarized:
            color = f"mapqc_per_cell_perc{MapqcResults.q}_zscored_q_dist_to_ref>2_nhood_based"
            sort_order = True
        else:
            color = f"mapqc_per_cell_perc{MapqcResults.q}_zscored_q_dist_to_ref_nhood_based"
            sort_order = False
    # TO DO: FOR ZSCORE, ORDER CELLS SO THAT NANS ARE PLOTTED FIRST/AT THE BOTTOM
    sc.pl.umap(
        MapqcResults._adata_ordered,
        color=color,
        palette= {">2": "red", "<=2": "blue", "filtered out":"black"}, # {">2": "red", "<=2": "antiquewhite", "filtered out":"black"},
        frameon=False,
        na_color="gray",
        size=size,
        cmap="Reds",
        vmin=vmin,
        vmax=vmax,
        sort_order=sort_order,
        title=title
    )



def sample_distances_to_reference(MapqcResults):
    """
    TO DO: add documentation
    This is at nhood level
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(1, 1, 1)
    per_sample_meandists_query = MapqcResults.mean_s2ref_dist_per_nhood_zscored_df.loc[
        MapqcResults.mean_s2ref_dist_per_nhood_zscored_df.sample_r_or_q
        == MapqcResults.query_cat,
        :,
    ]
    sample_medians = (
        per_sample_meandists_query.groupby("sample")
        .agg({"mapQC_score": "median"})
        .sort_values(by="mapQC_score")
    )
    palette = {
        sample: "grey" if zscore < 2 else "red"
        for sample, zscore in zip(
            sample_medians.index, sample_medians.mapQC_score
        )
    }
    sns.boxplot(
        per_sample_meandists_query,
        x="sample",
        y="mapQC_score",
        hue="sample",
        palette=palette,
        order=sample_medians.index,
        ax=ax,
    )
    sns.stripplot(
        per_sample_meandists_query,
        x="sample",
        y="mapQC_score",
        color="black",
        order=sample_medians.index,
        size=1,
        ax=ax,
    )
    ax.tick_params(axis="x", rotation=90)
    ax.set_ylabel("MapQC score per neighborhood")
    ax.set_xlabel("Sample")
    plt.axhline(y=2, linestyle="--")
    plt.tight_layout()


def sample_distances_to_reference_cell_level(MapqcResults):
    """
    TO DO: add documentation
    This is at cell level
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(1, 1, 1)
    # TO DO: ADD CONDITIONAL HERE TO CHECK IF CELL LEVEL RESULTS WERE EVEN CALCULATED
    cell_level_results = MapqcResults.cell_results_nhood_sample_based.copy()
    cell_level_results["sample"] = MapqcResults.adata.obs.loc[
        cell_level_results.index, MapqcResults.sample_key
    ].tolist()
    cell_level_results.dropna(axis=0,how='any',inplace=True)
    sample_medians = (
        cell_level_results.groupby("sample")
        .agg({"mapQC_score": "median"})
        .sort_values(by="mapQC_score")
    ).dropna(axis=0, how="all")
    palette = {
        sample: "grey" if zscore < 2 else "red"
        for sample, zscore in zip(
            sample_medians.index, sample_medians.mapQC_score
        )
    }
    sns.boxplot(
        cell_level_results,
        x="sample",
        y="mapQC_score",
        hue="sample",
        palette=palette,
        order=sample_medians.index,
        ax=ax,
    )
    sns.stripplot(
        cell_level_results,
        x="sample",
        y="mapQC_score",
        color="black",
        order=sample_medians.index,
        size=1,
        ax=ax,
    )
    ax.tick_params(axis="x", rotation=90)
    ax.set_ylabel("Cell mapQC scores")
    ax.set_xlabel("Sample")
    plt.axhline(y=2, linestyle="--")
    plt.tight_layout()


def _prepare_adata_for_plotting(MapqcResults: MapqcResults, scale: int = 1):
    """
    Prepares AnnData of MapqcResults for plotting by adding the calculated
    statistics to the MapqcResults.adata instance, calculating the dot size
    such that neighborhood center cells are plotted bigger than the other cells,
    and orders cells such that center cells are plotted on top.

    Parameters
    ----------
    MapqcResults
        Instance of MapqcResults class for which neighborhood statistics have
        already been calculated using the MapqcResults.calculate_nhood_statistics
        function
    scale
        Scale with which to change overall dot size in UMAP.

    Returns
    -------
    Adds statistics to MapqcResults.adata, and creates a MapqcResults._adata_ordered
    to which statistics have also been added and which has been ordered such
    that neighborhood center cells are plotted on top.
    The information added to the adata object is stored under columns whose name
    starts with "mapqc", e.g. "mapqc_nhood_weight" or "mapqc_dotsize".
    """
    if not MapqcResults._nhood_stats_calculated:
        raise ValueError(
            "Calculate result statistics first using the 'calculate_nhood_statistics' function."
        )
    for stat_to_copy in [
        "nhood_weight",
        "pval_corr",
        "mapQC_score",
        "mapQC_score>2",
        "median_zscore>2",
        f"perc{MapqcResults.q}_zscore>2",
    ]:
        MapqcResults.adata.obs[f"mapqc_{stat_to_copy}"] = np.nan
        MapqcResults.adata.obs.loc[
            MapqcResults.nhood_stats.index, f"mapqc_{stat_to_copy}"
        ] = MapqcResults.nhood_stats[stat_to_copy]
    # include filtered out nhood information in "mapqc_sign" column:
    MapqcResults.adata.obs["mapqc_sign"] = np.nan
    sign_nhoods = MapqcResults.nhood_stats.loc[MapqcResults.nhood_stats.sign, :].index
    non_sign_nhoods = MapqcResults.nhood_stats.loc[
        ~MapqcResults.nhood_stats.sign, :
    ].index
    filtered_nhoods = [
        nhood
        for nhood in MapqcResults.center_cells
        if nhood not in MapqcResults.nhood_stats.index
    ]
    MapqcResults.adata.obs.loc[sign_nhoods, "mapqc_sign"] = "significant"
    MapqcResults.adata.obs.loc[non_sign_nhoods, "mapqc_sign"] = "not significant"
    MapqcResults.adata.obs.loc[filtered_nhoods, "mapqc_sign"] = "excluded"
    # include info on which neighborhoods were excluded in other columns too, and
    # convert boolean too informative string:
    for stat in ["mapQC_score>2", "median_zscore>2", f"perc{MapqcResults.q}_zscore>2"]:
        bool_to_string = {True: ">2 stdev above mean", False: "<=2 stdev above mean"}
        MapqcResults.adata.obs["mapqc_" + stat] = MapqcResults.adata.obs[
            "mapqc_" + stat
        ].map(bool_to_string)
        MapqcResults.adata.obs.loc[filtered_nhoods, "mapqc_" + stat] = "excluded"
    # store filtering info:
    MapqcResults.adata.obs["mapqc_filter_info"] = np.nan
    MapqcResults.adata.obs.loc[
        MapqcResults._nhood_filtered.index, "mapqc_filter_info"
    ] = MapqcResults._nhood_filtered
    # set sizes for plotting:
    MapqcResults.adata.obs["mapqc_dotsize"] = scale * 1
    MapqcResults.adata.obs.loc[MapqcResults.center_cells, "mapqc_dotsize"] = (
        scale * 30
    )
    # set sizes for plotting, but proportional to neighborhood weight now
    MapqcResults.adata.obs[
        "mapqc_dotsize_by_nhood_weight"
    ] = MapqcResults.adata.obs.mapqc_dotsize
    MapqcResults.adata.obs.loc[
        MapqcResults.nhood_stats.index, "mapqc_dotsize_by_nhood_weight"
    ] = (
        MapqcResults.adata.obs.loc[
            MapqcResults.nhood_stats.index, "mapqc_dotsize_by_nhood_weight"
        ]
        * MapqcResults.nhood_stats.nhood_weight
    )
    # order cells so that index cells are plotted on top, then query, then
    # reference
    cells_ordered = (
        MapqcResults.adata.obs.index[
            MapqcResults.adata.obs[MapqcResults.ref_q_key] == MapqcResults.ref_cat
        ].tolist()
        + [
            cell
            for cell, r_or_q in zip(
                MapqcResults.adata.obs.index,
                MapqcResults.adata.obs[MapqcResults.ref_q_key],
            )
            if (
                (r_or_q == MapqcResults.query_cat)
                and (cell not in MapqcResults.center_cells)
            )
        ]
        + MapqcResults.center_cells.tolist()
    )
    # store ordered version of adata
    # TO DO: we're storing adata twice now, which is very memory inefficient
    # but I don't want to re-order the original AnnData (it will mess
    # with all the code above which relies on the adata row order). Think
    # of a better alternative here...
    MapqcResults._adata_ordered = MapqcResults.adata[cells_ordered, :].copy()
    # remember that adata has been prepped for plotting:
    MapqcResults._adata_prepped_for_plotting = True


def _prepare_adata_for_cell_level_res_plotting(MapqcResults, sample_based=True):
    if sample_based:
        if not MapqcResults._cell_level_stats_sample_based_calculated:
            raise ValueError(
                "Cell-level stats need to be calculated first; run results.calculate_cell_statistics_using_sample_level_results_per_nhood()."
            )
        MapqcResults.adata.obs[
            "mapQC_score"
        ] = np.nan
        MapqcResults.adata.obs.loc[
            MapqcResults.cell_results_nhood_sample_based.index,
            "mapQC_score",
        ] = MapqcResults.cell_results_nhood_sample_based.mapQC_score
        MapqcResults.adata.obs[
            "mapQC_score>2"
        ] = np.nan
        MapqcResults.adata.obs.loc[
            MapqcResults.cell_results_nhood_sample_based.index,
            "mapQC_score>2",
        ] = MapqcResults.cell_results_nhood_sample_based["mapQC_score>2"]
    else:
        if not MapqcResults._cell_level_stats_nhood_only_based_calculated:
            raise ValueError(
                "Cell-level stats need to be calculated first; run results.calculate_cell_statistics_from_nhoods_using_nhood_stats()."
            )
        for stat in [
            f"mapqc_per_cell_perc{MapqcResults.q}_zscored_q_dist_to_ref>2_nhood_based",
            f"mapqc_per_cell_perc{MapqcResults.q}_zscored_q_dist_to_ref_nhood_based",
        ]:
            MapqcResults.adata.obs[stat] = np.nan
            MapqcResults.adata.obs.loc[
                MapqcResults.cell_results_nhood_based.index, stat
            ] = MapqcResults.cell_results_nhood_based[stat]
    # store ordered version of adata for plotting:
    cells_ordered = (
        MapqcResults.adata.obs.index[
            MapqcResults.adata.obs[MapqcResults.ref_q_key] == MapqcResults.ref_cat
        ].tolist()
        + [
            cell
            for cell, r_or_q in zip(
                MapqcResults.adata.obs.index,
                MapqcResults.adata.obs[MapqcResults.ref_q_key],
            )
            if (
                (r_or_q == MapqcResults.query_cat)
                and (cell not in MapqcResults.center_cells)
            )
        ]
        + MapqcResults.center_cells.tolist()
    )
    # store ordered version of adata
    MapqcResults._adata_ordered = MapqcResults.adata[cells_ordered, :].copy()
    # store that results have been calculated successfully:
    if sample_based:
        MapqcResults._adata_prepped_for_cell_level_res_plotting_sample_based = True
    else:
        MapqcResults._adata_prepped_for_cell_level_res_plotting_nhood_only_based = True
