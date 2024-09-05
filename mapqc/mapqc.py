import nhoods
import numpy as np
import pandas as pd
import os
import pickle
from scipy.stats import kstest
from scipy.sparse import csr_matrix
import pval_corr
from anndata import AnnData
from typing import Literal
from tqdm import tqdm


class mapQC_results:
    """
    mapQC results class. Enables calculation of mapQC statistics for a given query
    to reference mapping and plotting using the plotting module.

    Parameters
    ---------
    adata
        AnnData object containing the data of both the reference and query
    center cells
        1-dimensional array containing the barcodes of the query cells for which
        the neighborhoods will be used for calculating the mapQC results
    k
        Number of neighbors to include per neighborhood
    emb_key
        Location of the integrated embedding to which the query was mapped.
        If set to "X", adata.X will be used. Otherwise, a key in obsm should
        be specified.
    distance_type
        Whether to use main pairwise (cell-cell) euclidean distance to quantify
        distance between a pair of samples, or sinkhorn divergence (approximation
        of Wasserstein distance).
    ref_q_key
        The adata.obs column in which is specified for each cell whether it is
        from the reference or the query
    ref_cat
        Name of the reference category in adata.obs[ref_q_key]
    query_cat
        Name of the query category in adata.obs[ref_q_key]
    sample_key
        Name of the adata.obs column in which sample names can be found
    exclude_same_batch_for_ref_samples
        Whether or not to exclude samples from the same batch, when calculating
        inter-sample distances for the reference itself.
    batch_key
        name of column containing batch information. This only needs to be set
        if exclude_same_batch_for_ref_samples is set to True
    min_n_cells_per_sample
        Minimum number of cells of a given sample present in a neighborhood,
        for that sample to be considered present.
    min_n_samples_ref
        Minimum number of reference samples (with at least
        min_n_cells_per_sample) present in a neighborhood for that neighborhood
        to pass filtering and be included in the statistics calculation later
        on.
    min_n_samples_q
        As min_n_samples_ref, but now for query samples
    use_adaptive_k
        If set to True, increases the k for a specific neighborhood if that
        neighborhood does not include sufficient reference samples with the default
        k (k argument above) to pass filtering, until sufficient reference samples
        are present in the neighborhood. For more details, see max_k and
        adaptive_k_margin arguments.
    max_k
        Maximum k to which a neighborhood's size can be increased, if with the
        default k not enough reference samples are found in the neighborhood and
        if use_adaptive_k is set to True.
    adaptive_k_margin
        Given the lowest possible k (minimum k) for a neighborhood to pass filtering,
        a k of adaptive_k_margin * minimum_k will be used as the final k for the
        neighborhood, if use_adaptive_k is set to True.

    TO DO: ADD DOCUMENTATION FOR ADAPTIVE K PARAMETERS
    """

    def __init__(
        self,
        adata: AnnData,
        center_cells: np.ndarray,
        k: int,
        emb_key: str,
        distance_type: Literal["pairwise_euclidean", "sinkhorn"] = "pairwise_euclidean",
        ref_q_key: str = "reference_or_query",
        ref_cat: str = "reference",
        query_cat: str = "query",
        sample_key: str = "sample",
        batch_key: str = None,
        exclude_same_batch_for_ref_samples: bool = True,
        min_n_cells_per_sample: int = 10,
        min_n_samples_ref: int = 3,
        min_n_samples_q: int = 3,
        use_adaptive_k: bool = True,
        max_k: int = None,
        adaptive_k_margin: float = 1.1,
    ):
        # TO DO: build in sanity checks for all parameters here
        self.adata = adata
        self.center_cells = center_cells
        self.k = k
        self.emb_key = emb_key
        self.ref_q_key = ref_q_key
        self.query_cat = query_cat
        self.ref_cat = ref_cat
        self.sample_key = sample_key
        self.exclude_same_batch_for_ref_samples = exclude_same_batch_for_ref_samples
        self.min_n_cells_per_sample = min_n_cells_per_sample
        self.min_n_samples_ref = min_n_samples_ref
        self.min_n_samples_q = min_n_samples_q
        self.q = None
        self.results = dict()
        # Sanity checks:
        # check if adata.obs.index unique:
        if not adata.shape[0] == len(set(adata.obs.index)):
            raise ValueError(
                "Indices of adata.obs are not unique. To make indices unique, run adata.obs_names_make_unique()."
            )
        # exclude_same_batch_for_ref_samples
        if exclude_same_batch_for_ref_samples:
            if isinstance(batch_key, type(None)):
                raise ValueError(
                    "No argument provided for batch_key, even though exclude_same_batch_for_ref_samples is set to true."
                )
            elif batch_key not in adata.obs.columns:
                raise ValueError("batch_key not found in adata.obs columns.")
        self.batch_key = batch_key
        # adaptive_k:
        if use_adaptive_k:
            if isinstance(max_k, type(None)):
                raise ValueError(
                    "max_k has to be specified when use_adaptive_k is set to true."
                )
            if adaptive_k_margin < 1:
                raise ValueError(
                    "adaptive_k_margin should be set to a value of at least 1."
                )
        self.use_adaptive_k = use_adaptive_k
        self.max_k = max_k
        self.adaptive_k_margin = adaptive_k_margin
        # sample_key
        if sample_key not in adata.obs.columns:
            raise ValueError("Your sample key is not present in adata.obs.columns.")
        if not isinstance(self.adata.obs[sample_key].values, pd.Categorical):
            self.adata.obs[sample_key] = pd.Categorical(self.adata.obs[sample_key])
        # distance_type
        if distance_type not in ["pairwise_euclidean", "sinkhorn"]:
            raise ValueError(
                "Argument 'distance_type' should be set to either 'pairwise_euclidiean' or 'sinkhorn'."
            )
        self.distance_type = distance_type
        # embedding key:
        if emb_key != "X" and (emb_key not in adata.obsm.keys()):
            raise ValueError(
                "emb_key should be set to either 'X' or a key in adata.obsm"
            )
        # convert sparse to dense array if embedding is sparse array
        else:
            if emb_key == "X":
                if isinstance(self.adata.X, csr_matrix):
                    self.adata.X = self.adata.X.toarray()
            elif isinstance(self.adata.obsm[emb_key], csr_matrix):
                self.adata.obsm[emb_key] = self.adata.obsm[emb_key].toarray
        # set parameters specifying if specific functions were run or not
        # (TO DO: complete this further)
        self._adata_prepped_for_plotting = False
        self._adata_prepped_for_cell_level_res_plotting_sample_based = False
        self._adata_prepped_for_cell_level_res_plotting_nhood_only_based = False
        self._results_prepped_for_boxplot = False
        self._nhood_stats_calculated = False
        self._cell_level_stats_nhood_only_based_calculated = False
        self._cell_level_stats_sample_based_calculated = False
        self._sample_dists_to_ref_calculated = False
        # initiate Series specifying whether or not a neighborhood passed
        # filtering (including exclusion reason if excluded)
        self._nhood_filtered = pd.Series(index=self.center_cells, dtype="object")
        # and same for final k used:
        self._nhood_k = pd.Series(index=self.center_cells, dtype="int")

    def calculate_sample_dists_to_ref_samples(self):
        """
        Calculate the mean distance to all reference samples for each
        sample in each neighborhood.
        Distances for a given pair of samples are calculated as the mean
        distance between the cells from each sample. Distances are
        Euclidean and calculated in the integrated embedding space.

        Returns
        ------
        Adds np.array of size (n_total_samples(with ref_samples first) x n_nhoods) as
        self.mean_sample_dists_to_ref to class instance, containing mean distances between
        each sample and all reference samples.
        Also adds np.array of the same size as self.mean_sample_dists_to_ref_zscored,
        containing the zscored distances to the reference. For zscoring, the mean
        distance and standard deviation of reference samples to reference are used,
        calculated per neighborhood.
        """
        self._calculate_intersample_distances()
        self.mean_sample_dists_to_ref = np.nanmean(self.intersample_distances, axis=0)
        # TO DO: MOVE THIS TO nhoods.py SCRIPT, I THINK
        # add z-scored distances to reference, setting the mean of the reference
        # sample distances to reference as 0, and their standard deviation to 1.
        # first, calculate mean dist to ref among ref samples (1 per nhood)
        ref_sample_dists_to_ref = self.mean_sample_dists_to_ref[: len(self.ref_samples_all),]
        # NEW: REMOVE OUTLIERS:
        max_n_interquartile_range_from_hinge = 3.5
        nhood_ref_25percentiles = np.nanpercentile(ref_sample_dists_to_ref, q=25, axis=0)
        nhood_ref_75percentiles = np.nanpercentile(ref_sample_dists_to_ref, q=75, axis=0)
        nhood_ref_interquartile_range = nhood_ref_75percentiles - nhood_ref_25percentiles
        upper_threshold_outliers = (
            nhood_ref_75percentiles
            + max_n_interquartile_range_from_hinge * nhood_ref_interquartile_range
        )
        lower_threshold_outliers = (
            nhood_ref_25percentiles
            - max_n_interquartile_range_from_hinge * nhood_ref_interquartile_range
        )
        upper_outliers = ref_sample_dists_to_ref > upper_threshold_outliers
        lower_outliers = ref_sample_dists_to_ref < lower_threshold_outliers
        outliers = upper_outliers | lower_outliers
        ref_sample_dists_to_ref_outliers_removed = ref_sample_dists_to_ref.copy()
        ref_sample_dists_to_ref_outliers_removed[outliers] = np.nan
        # END OF NEW PART, except for now using ref_sample_dists_to_ref_outliers_removed
        # below instead of ref_sample_dists_to_ref
        nhood_ref_mean_dists = np.nanmean(
            ref_sample_dists_to_ref_outliers_removed,
            axis=0,
        )
        # calculate standard deviations of ref dist to ref samples (1 per nhood)
        nhood_ref_stdev = np.nanstd(
            ref_sample_dists_to_ref_outliers_removed,
            axis=0,
        )
        # now zscore all distances and store
        self.mean_sample_dists_to_ref_zscored = (
            self.mean_sample_dists_to_ref - nhood_ref_mean_dists
        ) / nhood_ref_stdev
        # also store a dataframe version of both (useful for plotting and
        # downstream analysis)
        # absolute distances
        mean_s2ref_dists_per_nhood_df = (
            pd.DataFrame(
                self.mean_sample_dists_to_ref,
                index=self.ref_q_samples_all,
                columns=self.center_cells,
            )
            .unstack()
            .reset_index()
            .rename(
                columns={
                    "level_0": "nhood",
                    "level_1": "sample",
                    0: "mean_dist_to_ref",
                }
            )
        )
        # zscored distances:
        mean_s2ref_dists_per_nhood_df_zscored = (
            pd.DataFrame(
                self.mean_sample_dists_to_ref_zscored,
                index=self.ref_q_samples_all,
                columns=self.center_cells,
            )
            .unstack()
            .reset_index()
            .rename(
                columns={
                    "level_0": "nhood",
                    "level_1": "sample",
                    0: "mean_zscored_dist_to_ref",
                }
            )
        )

        mean_s2ref_dists_per_nhood_df.dropna(how="any", axis=0, inplace=True)
        mean_s2ref_dists_per_nhood_df_zscored.dropna(how="any", axis=0, inplace=True)
        # add info on whether sample is from ref or query
        sample_from_r_or_q_df = self.adata.obs.groupby(self.sample_key).agg(
            {self.ref_q_key: "first"}
        )
        sample_from_r_or_q_dict = {
            s: rvsq
            for s, rvsq in zip(
                sample_from_r_or_q_df.index, sample_from_r_or_q_df[self.ref_q_key]
            )
        }
        mean_s2ref_dists_per_nhood_df["sample_r_or_q"] = mean_s2ref_dists_per_nhood_df[
            "sample"
        ].map(sample_from_r_or_q_dict)
        mean_s2ref_dists_per_nhood_df_zscored[
            "sample_r_or_q"
        ] = mean_s2ref_dists_per_nhood_df_zscored["sample"].map(sample_from_r_or_q_dict)
        # store
        self.mean_s2ref_dist_per_nhood_df = mean_s2ref_dists_per_nhood_df
        self.mean_s2ref_dist_per_nhood_zscored_df = (
            mean_s2ref_dists_per_nhood_df_zscored
        )

        # calculate sample-specific performance across neighborhoods
        self._calculate_median_sample_dist_to_ref_across_nhoods()
        # store that this function has been run:
        self._sample_dists_to_ref_calculated = True

    def calculate_nhood_statistics(self, alpha=0.05, q=75, verbose=True):
        """
        Calculates the statistics for testing whether reference-to-reference
        sample distances are smaller than refererence-to-query sample distances.
        Includes FDR-based p-value correction that is weighted based on the overlap
        between neighborhoods, weighing neighborhoods with a lot of overlap less heavily.

        Returns
        -------
        Adds self.nhood_stats to class instance, a pandas.DataFrame with statistics
        per neighborhood. Neighborhoods that did not pass filtering (i.e. did not
        have enough query or reference samples) are left out.
        """
        # first check if any neighborhoods passed filtering:
        if self.perc_of_nhoods_passed_filter == 0:
            raise ValueError(
                "None of the neighborhoods passed filtering, and so no results can be calculated."
            )
        # check if q was set previously, and if it is now changing:
        if not isinstance(self.q, type(None)):
            if self.q != q:
                # make sure that plotting stats will be recalculated (if already calculated)
                # for this new q:
                self._adata_prepped_for_plotting = False
                self._adata_prepped_for_cell_level_res_plotting = False
                self._adata_prepped_for_cell_level_res_plotting_sample_based = False
                self._adata_prepped_for_cell_level_res_plotting_nhood_only_based = False
        self.q = q
        self.nhood_stats = pd.DataFrame(
            index=self.center_cells, columns=["ks_stat", "pval"]
        )
        self._calculate_nhood_ks_statistics()
        # keep only neighborhoods that passed filtering
        self.nhood_stats["nhood_number"] = range(0, len(self.center_cells))
        self.nhood_stats.dropna(axis=0, how="any", inplace=True)
        # store neighborhood k
        self.nhood_stats["k"] = self._nhood_k.loc[self.nhood_stats.index]
        # calculate weight per neighborhood based on its overlap with other
        # nhoods
        (
            self.nhood_stats["nhood_weight"],
            self._neighbor_matrix,
        ) = pval_corr.calculate_cell_and_neighborhood_weights(
            self.nhood_cells[self.nhood_stats.nhood_number, :], self.adata.n_obs
        )
        # perform FDR-based p-value correction using nhood weights
        self.nhood_stats["pval_corr"] = pval_corr.weighted_fdr_corr_pvals(
            self.nhood_stats.pval, self.nhood_stats.nhood_weight
        )
        # add boolean specifying if neighborhood is below significance
        # threshold of alpha=0.05
        self.nhood_stats["sign"] = self.nhood_stats.pval_corr < alpha
        if verbose:
            perc_of_nhoods_sign = round(
                self.nhood_stats.sign.sum() / self.nhood_stats.shape[0] * 100
            )
            print(
                f"{perc_of_nhoods_sign}% of neighborhoods that passed filtering are below the significance threshold alpha={alpha}."
            )
        # calculate the mean, median and qth percentile zscored distance of query
        # samples to reference for each neighborhood:
        mean_zscored_q_dist_to_ref = np.nanmean(
            self.mean_sample_dists_to_ref_zscored[len(self.ref_samples_all) :, :],
            axis=0,
        )
        median_zscored_q_dist_to_ref = np.nanmedian(
            self.mean_sample_dists_to_ref_zscored[len(self.ref_samples_all) :, :],
            axis=0,
        )
        percq_zscored_q_dist_to_ref = np.nanpercentile(
            self.mean_sample_dists_to_ref_zscored[len(self.ref_samples_all) :, :],
            q=q,
            axis=0,
        )
        # take only the nhoods that passed filtering:
        self.nhood_stats["mean_zscored_q_dist_to_ref"] = mean_zscored_q_dist_to_ref[
            self.nhood_stats.nhood_number
        ]
        self.nhood_stats["median_zscored_q_dist_to_ref"] = median_zscored_q_dist_to_ref[
            self.nhood_stats.nhood_number
        ]
        self.nhood_stats[
            f"perc{q}_zscored_q_dist_to_ref"
        ] = percq_zscored_q_dist_to_ref[self.nhood_stats.nhood_number]
        # store which ones pass the 2-standard deviations above the mean threshold:
        self.nhood_stats["mean_zscore>2"] = (
            self.nhood_stats.mean_zscored_q_dist_to_ref > 2
        )
        self.nhood_stats["median_zscore>2"] = (
            self.nhood_stats.median_zscored_q_dist_to_ref > 2
        )
        self.nhood_stats[f"perc{q}_zscore>2"] = (
            self.nhood_stats[f"perc{q}_zscored_q_dist_to_ref"] > 2
        )
        if verbose:
            perc_of_nhoods_percq_zscore_over_2 = round(
                self.nhood_stats[f"perc{q}_zscore>2"].sum()
                / self.nhood_stats.shape[0]
                * 100
            )
            print(
                f"For {perc_of_nhoods_percq_zscore_over_2}% of filtered neighborhoods, >{100-q}% of samples are more than two standard deviations above the mean distance from the reference."
            )
        # check sample-specific statistics:
        samples_with_no_results = [
            sample
            for sample in self.query_samples_all
            if sample not in self.sample_median_dist_to_ref_across_nhoods.index
        ]
        if len(samples_with_no_results) > 0:
            print(
                f"Warning: no results were calculated for samples {samples_with_no_results}. This could be because these samples have too few cells."
            )
        n_samples_with_results = len(self.query_samples_all) - len(
            samples_with_no_results
        )
        n_samples_not_mixing_with_ref = (
            self.sample_median_dist_to_ref_across_nhoods.values > 2
        ).sum()
        perc_of_query_samples_not_mixing = round(
            n_samples_not_mixing_with_ref / n_samples_with_results * 100
        )
        print(
            f"{perc_of_query_samples_not_mixing}% of query samples ({n_samples_not_mixing_with_ref} sample(s)) consistently do not mix well with the reference."
        )
        if perc_of_query_samples_not_mixing > 0:
            print(
                "If these are control samples, consider removing them from the query and re-mapping."
            )
            print(
                "For more information, check .sample_median_dist_to_ref_across_nhoods\nor run plotting.plot_sample_distances_to_reference."
            )
        # remember that this function has been successfully run:
        self._nhood_stats_calculated = True

    def calculate_cell_statistics_from_nhoods_using_nhood_stats(self):
        """
        Calculate for each query cell sampled in one or more neighborhoods
        the mean overall query statistic across its neighborhoods. Statistics
        to calculate:
        - mean zscored query distance to reference in nhood(s)
        - median zscored query distance to reference in nhood(s)
        - qth percentile of nhood's query sample distances to reference
        (mean qth percentile across nhoods)
        """
        if not self._nhood_stats_calculated:
            raise ValueError(
                "Neighborhood statistics need to be calculated first. Run .calculate_nhood_statistics."
            )
        cell_results = pd.DataFrame(index=self.adata.obs.index)
        for stat in [
            "mean_zscored_q_dist_to_ref",
            "median_zscored_q_dist_to_ref",
            f"perc{self.q}_zscored_q_dist_to_ref",
        ]:
            stat_res_per_nhood = np.empty(
                shape=(1, self.nhood_stats.shape[0])
            )  # self.center_cells.shape[0]))
            stat_res_per_nhood[:] = np.nan
            stat_res_per_nhood[
                :
            ] = self.nhood_stats[  # self.nhood_stats.nhood_number] = self.nhood_stats[
                stat
            ]
            with np.errstate(invalid="ignore"):
                stat_per_cell = (
                    stat_res_per_nhood
                    @ self._neighbor_matrix
                    / self._neighbor_matrix.sum(axis=0)
                )
            # add to cell_results dataframe
            cell_results[f"mapqc_per_cell_{stat}_nhood_based"] = stat_per_cell[0, :]
            # specify whether cells pass threshold of Zscore >2 for each statistic
            cell_results[f"mapqc_per_cell_{stat}>2_nhood_based"] = (
                cell_results[f"mapqc_per_cell_{stat}_nhood_based"] > 2
            )
            # map to categoricals for plotting
            cell_results[f"mapqc_per_cell_{stat}>2_nhood_based"] = cell_results[
                f"mapqc_per_cell_{stat}>2_nhood_based"
            ].map({True: f">2", False: f"<=2"})
            # set cells without results (i.e. that were not in any neighborhood that passed
            # filtering) to nan:
            cell_results.loc[
                np.isnan(cell_results[f"mapqc_per_cell_{stat}_nhood_based"]),
                f"mapqc_per_cell_{stat}>2_nhood_based",
            ] = np.nan
        # keep only cells from query:
        cell_results = cell_results.loc[
            self.adata.obs[self.ref_q_key] == self.query_cat, :
        ].copy()
        # print results:
        total_n_cells_with_results = np.sum(~np.isnan(cell_results.iloc[:, 0]))
        perc_cells_percq_zscored_q_dist_to_ref_over_2 = round(
            np.sum(
                cell_results[
                    f"mapqc_per_cell_perc{self.q}_zscored_q_dist_to_ref>2_nhood_based"
                ]
                == ">2"
            )
            / total_n_cells_with_results
            * 100,
            2,
        )
        print(
            f"Percentage of query cells in low-mixing neighborhoods: {perc_cells_percq_zscored_q_dist_to_ref_over_2} (cutoff percentile: {self.q})."
        )
        # store
        self.cell_results_nhood_based = cell_results
        # remember that these stats were calculated:
        self._cell_level_stats_nhood_only_based_calculated = True

    def calculate_cell_statistics_using_sample_level_results_per_nhood(self):
        """
        Calculate the per-cell zscored distance to the reference, based on the 
        cell's sample's distance to the reference in all the neighborhoods the 
        cell occurred in, and taking the mean across those neighborhoods.
        """
        if not self._nhood_stats_calculated:
            raise ValueError(
                "Neighborhood statistics need to be calculated first. Run .calculate_nhood_statistics."
            )
        query_cells = self.adata.obs.loc[
            self.adata.obs[self.ref_q_key] == self.query_cat, :
        ].index
        cell_results = pd.DataFrame(
            index=query_cells, columns=["mean_zscored_dist_to_ref"]
        )
        # collect indices of cells that did not pass filtering in any 
        # neighborhood:
        cells_from_filtered_out_nhoods = list(set([
            idx
            for sublist in self._cells_from_filtered_neighborhoods.values()
            for idx in sublist
        ]))
        cells_without_results_due_to_filtering = list()
        for cell_idx, cell in enumerate(self.adata.obs.index):
            if cell in query_cells:
                cell_nhoods_binary = self._neighbor_matrix[:, cell_idx]
                # check if cell was in any nhoods:
                if cell_nhoods_binary.sum() > 0:
                    # if so, store neighborhoods
                    query_cell_nhoods = self.nhood_stats.iloc[
                        cell_nhoods_binary.astype(bool), :
                    ].index.tolist()
                    # check which sample the query came from
                    query_cell_sample = self.adata.obs.loc[cell, self.sample_key]
                    # get results for that sample, in the neighborhood(s) in which this
                    # cell was included
                    rows_with_nhood_results_for_query_sample = (
                        self.mean_s2ref_dist_per_nhood_zscored_df.nhood.isin(
                            query_cell_nhoods
                        )
                    ) & (
                        self.mean_s2ref_dist_per_nhood_zscored_df["sample"].values
                        == query_cell_sample
                    )
                    # check if any rows are left. Not sure if it can happen that there are none,
                    # trying to check that with the conditional below:
                    if len(rows_with_nhood_results_for_query_sample) == 0:
                        raise ValueError("There are no results for this query sample, but you are trying to calculate the mean results. Fix this in the code...")
                    cell_results.loc[
                        cell, "mean_zscored_dist_to_ref"
                    ] = self.mean_s2ref_dist_per_nhood_zscored_df.mean_zscored_dist_to_ref[
                        rows_with_nhood_results_for_query_sample
                    ].mean()
                else:
                    # TO DO: check if query cell was excluded due to filtering, or
                    # just not sampled....
                    if cell_idx in cells_from_filtered_out_nhoods:
                        cells_without_results_due_to_filtering.append(cell)
        cell_results["mean_zscored_dist_to_ref>2"] = (
            cell_results.mean_zscored_dist_to_ref > 2
        )
        # keep nans, they are set to False with the above code
        cell_results.loc[
            cell_results.mean_zscored_dist_to_ref.isna(), "mean_zscored_dist_to_ref>2"
        ] = np.nan
        cell_results["mean_zscored_dist_to_ref>2"] = cell_results[
            "mean_zscored_dist_to_ref>2"
        ].map({True: f">2", False: f"<=2"})
        # add filtering info:
        cell_results.loc[cells_without_results_due_to_filtering,'mean_zscored_dist_to_ref>2'] = "filtered out"
        
        # print results:
        total_n_cells_with_results = np.sum(
            ~cell_results.mean_zscored_dist_to_ref.isna()
        )
        perc_cells_percq_zscored_q_dist_to_ref_over_2 = round(
            np.sum(cell_results["mean_zscored_dist_to_ref>2"] == ">2")
            / total_n_cells_with_results
            * 100,
            2,
        )
        print(
            f"Percentage of query cells showing low sample-mixing: {perc_cells_percq_zscored_q_dist_to_ref_over_2}."
        )
        # store percentage as result:
        self.results['perc_non_mixing_cells'] = perc_cells_percq_zscored_q_dist_to_ref_over_2
        self.cell_results_nhood_sample_based = cell_results
        self._cell_level_stats_sample_based_calculated = True

    def save(self, path):
        """
        Save mapQC results as pickle. Provide a path ending with .pkl.
        """
        if not path.endswith(".pkl"):
            path = path + ".pkl"
        print("saving results at", path)
        with open(path, 'wb') as outp:
            pickle.dump(self, outp, pickle.HIGHEST_PROTOCOL)

    def _calculate_intersample_distances(self):
        """
        For each pair of samples (one from the reference, one from either reference
        or query) calculate the mean pairwise Euclidean distance in the integrated
        embedding space between cells from the one sample and cells from the other.

        Returns
        ------
        Adds array self.intersample_distances to class instance. The array is of shape
        (n_reference_samples x n_total_samples (with ref_samples first) x n_neighborhoods)
        and contains mean intersample distances for each sample pair in each neighborhood.
        Samples that were not observed in a given neighborhood are set to np.nan
        across that neighborhood.
        """
        # generate empty results dataframe
        self.ref_samples_all = np.array(
            sorted(
                self.adata.obs.loc[
                    self.adata.obs[self.ref_q_key] == self.ref_cat, self.sample_key
                ].unique()
            )
        )
        self.query_samples_all = np.array(
            sorted(
                self.adata.obs.loc[
                    self.adata.obs[self.ref_q_key] == self.query_cat,
                    self.sample_key,
                ].unique()
            )
        )
        self.ref_q_samples_all = np.concatenate(
            (
                self.ref_samples_all,
                self.query_samples_all,
            )
        )
        mean_s2s_dist_per_nhood = np.empty(
            (
                len(self.ref_samples_all),
                len(self.ref_q_samples_all),
                len(self.center_cells),
            )
        )
        mean_s2s_dist_per_nhood[:] = np.nan
        # generate array with nans in which to store each nhoods cells (i.e.
        # the k nearest neighbors of each index cell, including itself)
        self.nhood_cells = np.zeros((len(self.center_cells), self.max_k))
        self.nhood_cells[:] = np.nan
        # generate empty dict with cells from neighborhoods that have 
        # been filtered out:
        self._cells_from_filtered_neighborhoods = dict()
        # now loop through neighborhoods and calculate mean inter-sample
        # distances per nhood
        for nhood_count, center_cell in tqdm(enumerate(self.center_cells)):
            nhood = nhoods.Neighborhood(
                adata=self.adata,
                index_cell=center_cell,
                k=self.k,
                min_n_cells_per_sample=self.min_n_cells_per_sample,
                min_n_samples_ref=self.min_n_samples_ref,
                min_n_samples_q=self.min_n_samples_q,
                emb_key=self.emb_key,
                sample_key=self.sample_key,
                ref_q_key=self.ref_q_key,
                ref_cat=self.ref_cat,
                query_cat=self.query_cat,
                batch_key=self.batch_key,
                exclude_same_batch_for_ref_samples=self.exclude_same_batch_for_ref_samples,
                distance_type=self.distance_type,
                use_adaptive_k=self.use_adaptive_k,
                max_k=self.max_k,
                adaptive_k_margin=self.adaptive_k_margin,
            )
            nhood_results = nhood.get_pairwise_sample_dists()
            # store whether or not neighborhood passed sample number filtering
            self._nhood_filtered.loc[center_cell] = nhood._filter_info
            # store final k:
            self._nhood_k.loc[center_cell] = nhood.final_k
            # add results to overall tensor
            if not isinstance(nhood_results, type(np.nan)):
                self.nhood_cells[nhood_count, : nhood.final_k] = nhood.knn_idc
                row_idc_for_nhood_ref_samples = np.where(
                    np.isin(self.ref_samples_all, nhood_results.index)
                )[0]
                col_idc_for_nhood_all_samples = np.where(
                    np.isin(self.ref_q_samples_all, nhood_results.columns)
                )[0]
                mean_s2s_dist_per_nhood[
                    row_idc_for_nhood_ref_samples[:, np.newaxis, np.newaxis],
                    col_idc_for_nhood_all_samples[np.newaxis, :, np.newaxis],
                    np.array([nhood_count])[np.newaxis, np.newaxis, :],
                ] = nhood_results.values[:, :, np.newaxis]
            else:
                # store neighbors as cells that have been filtered out at some 
                # point. This will be used when calculating and visualizing
                # the cell-level results
                self._cells_from_filtered_neighborhoods[nhood.index_cell] = nhood.knn_idc
        # print filtering information
        self.perc_of_nhoods_passed_filter = round(
            (self._nhood_filtered == "pass").sum() / len(self.center_cells) * 100
        )
        print(
            f"{self.perc_of_nhoods_passed_filter}% of neighborhoods passed filtering. (For more details, see [mapqc_object]._nhood_filtered)"
        )
        # store intersample distances
        self.intersample_distances = mean_s2s_dist_per_nhood

    def _calculate_median_sample_dist_to_ref_across_nhoods(self):
        """
        Calculates the median distance of each query sample to the reference,
        taking the median across neighborhoods. These statistics are useful to
        check if specific samples performed poorly during the mapping.

        Returns
        -------
        Adds a pd.DataFrame sample_median_dist_to_ref_across_nhoods to the
        mapQC_results object, containing the median distance to the reference
        (calculated across neighborhoods) for each query sample.
        """
        per_sample_meandists_query = self.mean_s2ref_dist_per_nhood_zscored_df.loc[
            self.mean_s2ref_dist_per_nhood_zscored_df.sample_r_or_q == self.query_cat,
            :,
        ]
        self.sample_median_dist_to_ref_across_nhoods = (
            per_sample_meandists_query.groupby("sample")
            .agg({"mean_zscored_dist_to_ref": "median"})
            .sort_values(by="mean_zscored_dist_to_ref")
        ).rename(
            columns={"mean_zscored_dist_to_ref": "median_dist_to_ref_across_nhoods"}
        )

    def _calculate_nhood_ks_statistics(self):
        """
        Calculate Kolmogorov-Smirnov statistic and p-value for all neighborhoods,
        testing whether query sample distances to reference samples are larger
        than reference sample distances to reference samples.

        Returns
        -------
        Adds KS test statistic and p-value for each neighborhood to
        self.nhood_stats dataframe.
        """
        for nhood_number, nhood in enumerate(self.center_cells):
            # slice distances to ref samples for the current neighborhood, and
            # for the ref samples
            r2r_dists = self.mean_sample_dists_to_ref[
                : len(self.ref_samples_all), nhood_number
            ]
            # do the same for the distances to ref samples for the query samples
            r2q_dists = self.mean_sample_dists_to_ref[
                len(self.ref_samples_all) :, nhood_number
            ]
            # if no results for the given neighborhood (because it did not
            # pass filtering during distance calculation), either for ref
            # or for query, skip it
            if np.isnan(r2r_dists).all() or np.isnan(r2q_dists).all():
                continue
            # remove nans, i.e. drop samples that did not pass earlier filtering
            r2r_dists = r2r_dists[~np.isnan(r2r_dists)]
            r2q_dists = r2q_dists[~np.isnan(r2q_dists)]
            # perform KS test
            self.nhood_stats.loc[nhood, ["ks_stat", "pval"]] = kstest(
                rvs=r2r_dists, cdf=r2q_dists, alternative="greater"
            )[:2]

def load(path):
    """
    Load mapQC results from pickle file. Provide path to file ending with .pkl.
    """
    if not os.path.isfile(path):
        raise ValueError(f"the given path {path} does not lead to a file.")
    if not path.endswith(".pkl"):
        raise ValueError("The given path is not a pickle file.")
    with open(path, 'rb') as inp:
        return pickle.load(inp)
