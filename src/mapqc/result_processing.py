from .mapqc_dists import DistResults
import numpy as np


def add_nhood_stats_to_adata(mapqc_results: DistResults, scale: int = 1):
    """
    Prepares AnnData of mapqc_results for plotting by adding the calculated
    statistics to the mapqc_results.adata instance, calculating the dot size
    such that neighborhood center cells are plotted bigger than the other cells,
    and orders cells such that center cells are plotted on top.

    Parameters
    ----------
    mapqc_results
        Instance of mapqc_results class for which neighborhood statistics have
        already been calculated using the mapqc_results.calculate_nhood_statistics
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
        # "pval_corr",
        # "mapqc_score",
        # "mapqc_score>2",
        # "median_zscore>2",
        # f"perc{mapqc_results.q}_zscore>2",
    ]:
        if not stat_to_copy.startswith("nhood"):
            colname = f"nhood_{stat_to_copy}"
        else:
            colname = stat_to_copy
        mapqc_results.adata.obs[colname] = np.nan
        mapqc_results.adata.obs.loc[
            mapqc_results.nhood_stats.index, colname
        ] = mapqc_results.nhood_stats[stat_to_copy]
    # include filtered out nhood information in "mapqc_sign" column:
    # mapqc_results.adata.obs["nhood_mapqc_sign"] = np.nan
    # sign_nhoods = mapqc_results.nhood_stats.loc[mapqc_results.nhood_stats.sign, :].index
    # non_sign_nhoods = mapqc_results.nhood_stats.loc[
    #     ~mapqc_results.nhood_stats.sign, :
    # ].index
    filtered_nhoods = [
        nhood
        for nhood in mapqc_results.center_cells
        if nhood not in mapqc_results.nhood_stats.index
    ]
    # mapqc_results.adata.obs.loc[sign_nhoods, "nhood_mapqc_sign"] = "significant"
    # mapqc_results.adata.obs.loc[non_sign_nhoods, "nhood_mapqc_sign"] = "not significant"
    # mapqc_results.adata.obs.loc[filtered_nhoods, "nhood_mapqc_sign"] = "excluded"
    # include info on which neighborhoods were excluded in other columns too, and
    # convert boolean too informative string:
    # for stat in ["mapqc_score>2", "median_zscore>2", f"perc{mapqc_results.q}_zscore>2"]:
    #     bool_to_string = {True: ">2 stdev above mean", False: "<=2 stdev above mean"}
    #     mapqc_results.adata.obs["nhood_" + stat] = mapqc_results.adata.obs[
    #         "nhood_" + stat
    #     ].map(bool_to_string)
    #     mapqc_results.adata.obs.loc[filtered_nhoods, "nhood_" + stat] = "excluded"
    # store filtering info:
    mapqc_results.adata.obs["nhood_filter_info"] = np.nan
    mapqc_results.adata.obs.loc[
        mapqc_results._nhood_filtered.index, "nhood_filter_info"
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


def add_cell_level_stats_to_adata(mapqc_results, sample_based=True):
    if sample_based:
        if not mapqc_results._cell_level_stats_sample_based_calculated:
            raise ValueError(
                "Cell-level stats need to be calculated first; run results.calculate_cell_statistics_using_sample_level_results_per_nhood()."
            )
        for stat in ["mapqc_score", "mapqc_score>2", "mean_t_stat", "mean_pval_corr"]:
            mapqc_results.adata.obs[stat] = np.nan
            mapqc_results.adata.obs.loc[
                mapqc_results.cell_results_nhood_sample_based.index,
                stat,
            ] = mapqc_results.cell_results_nhood_sample_based[stat]
        # add boolean specifying if p-value below alpha = 0.05:
        mapqc_results.adata.obs["mean_pval_corr_significant"] = (
            mapqc_results.adata.obs.mean_pval_corr < 0.05
        )
        # set reference cells to np.nan:
        mapqc_results.adata.obs.loc[
            mapqc_results.adata.obs[mapqc_results.ref_q_key] == mapqc_results.ref_cat,
            "mean_pval_corr_significant",
        ] = np.nan
        # map boolean to string:
        mapqc_results.adata.obs["mean_pval_corr_significant"] = mapqc_results.adata.obs[
            "mean_pval_corr_significant"
        ].map({True: "significant", False: "not significant"})
        # copy filtering info over from "mapqc_score>2" column:
        for filter_info in ["filtered out", "not sampled"]:
            mapqc_results.adata.obs.loc[
                mapqc_results.adata.obs[f"mapqc_score>2"] == filter_info,
                "mean_pval_corr_significant",
            ] = filter_info
    else:
        raise ValueError(
            "Using nhood-only based cell-level results for plotting is deprecated. Code has been commented out, uncomment if we decide to re-implement this."
        )
        # if not mapqc_results._cell_level_stats_nhood_only_based_calculated:
        #     raise ValueError(
        #         "Cell-level stats need to be calculated first; run results.calculate_cell_statistics_from_nhoods_using_nhood_stats()."
        #     )
        # for stat in [
        #     f"mapqc_per_cell_perc{mapqc_results.q}_zscored_q_dist_to_ref>2_nhood_based",
        #     f"mapqc_per_cell_perc{mapqc_results.q}_zscored_q_dist_to_ref_nhood_based",
        # ]:
        #     mapqc_results.adata.obs[stat] = np.nan
        #     mapqc_results.adata.obs.loc[
        #         mapqc_results.cell_results_nhood_based.index, stat
        #     ] = mapqc_results.cell_results_nhood_based[stat]
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
    # else:
    #     mapqc_results._adata_prepped_for_cell_level_res_plotting_nhood_only_based = True
