import nhoods
import numpy as np
import pandas as pd
from scipy.stats import kstest
import pval_corr
from anndata import AnnData


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
        TO DO
    batch_key
        TO DO (only needed if exclude_same_batch_for_ref_samples set to True)


    TO DO: add filter settings as parameters with defaults here (now only
    in nhood class I think)
    """

    def __init__(
        self,
        adata: AnnData,
        center_cells: np.ndarray,
        k: int,
        emb_key: str = "X",
        ref_q_key: str = "reference_or_query",
        ref_cat: str = "reference",
        query_cat: str = "query",
        sample_key: str = "sample",
        batch_key: str = None,
        exclude_same_batch_for_ref_samples: bool = True,
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
        if exclude_same_batch_for_ref_samples:
            if isinstance(batch_key, type(None)):
                raise ValueError(
                    "No argument provided for batch_key, even though exclude_same_batch_for_ref_samples is set to true."
                )
            elif batch_key not in adata.obs.columns:
                raise ValueError("batch_key not found in adata.obs columns.")
            self.batch_key = batch_key
        # set parameters specifying if specific functions were run or not
        # (TO DO: complete this further)
        self._adata_prepped_for_plotting = False
        self._results_prepped_for_boxplot = False
        self._nhood_stats_calculated = False
        self._sample_dists_to_ref_calculated = False

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
        self.ref_q_samples_all = np.concatenate(
            (
                self.ref_samples_all,
                np.array(
                    sorted(
                        self.adata.obs.loc[
                            self.adata.obs[self.ref_q_key] == self.query_cat, "sample"
                        ].unique()
                    )
                ),
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
        # generate empty array in which to store each nhoods cells (i.e.
        # the k nearest neighbors of each index cell, including itself)
        self.nhood_cells = np.zeros((len(self.center_cells), self.k), dtype=int)
        # now loop through neighborhoods and calculate mean inter-sample
        # distances per nhood
        for nhood_count, center_cell in enumerate(self.center_cells):
            nhood = nhoods.Neighborhood(
                adata=self.adata,
                index_cell=center_cell,
                k=self.k,
                emb_key=self.emb_key,
                sample_key=self.sample_key,
                ref_q_key=self.ref_q_key,
                ref_cat=self.ref_cat,
                query_cat=self.query_cat,
                batch_key=self.batch_key,
                exclude_same_batch_for_ref_samples=self.exclude_same_batch_for_ref_samples,
            )
            nhood_results = nhood.get_pairwise_sample_dists()
            self.nhood_cells[nhood_count] = nhood.cell_idc
            # add results to overall tensor
            if not isinstance(nhood_results, type(np.nan)):
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
        self.intersample_distances = mean_s2s_dist_per_nhood

    def calculate_sample_dists_to_ref_samples(self):
        """
        Calculate the mean distance to all reference samples for each
        sample in each neighborhood.
        Distances for a given pair of samples are calculated as the mean
        distance between the cells from each sample. Distances are
        Euclidean and calculaten in the integrated embedding space.

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
        # add z-scored distances to reference, setting the mean of the reference
        # sample distances to reference as 0, and their standard deviation to 1.
        # first, calculate mean dist to ref among ref samples (1 per nhood)
        nhood_ref_mean_dists = np.nanmean(
            self.mean_sample_dists_to_ref[: len(self.ref_samples_all),],
            axis=0,
        )
        # calculate standard deviations of ref dist to ref samples (1 per nhood)
        nhood_ref_stdev = np.nanstd(
            self.mean_sample_dists_to_ref[: len(self.ref_samples_all),],
            axis=0,
        )
        # now zscore all distances and store
        self.mean_sample_dists_to_ref_zscored = (
            self.mean_sample_dists_to_ref - nhood_ref_mean_dists
        ) / nhood_ref_stdev
        # store that this function has been run:
        self._sample_dists_to_ref_calculated = True

    def calculate_nhood_statistics(self, alpha=0.05, verbose=True):
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
        self.nhood_stats = pd.DataFrame(
            index=self.center_cells, columns=["ks_stat", "pval"]
        )
        self._calculate_nhood_ks_statistics()
        # keep only neighborhoods that passed filtering
        self.nhood_stats["nhood_number"] = range(0, len(self.center_cells))
        self.nhood_stats.dropna(axis=0, how="any", inplace=True)
        # store perc. of nhoods that passed filtering
        self.perc_of_nhoods_passed_filter = round(
            self.nhood_stats.shape[0] / len(self.center_cells) * 100
        )
        if verbose:
            print(
                f"{self.perc_of_nhoods_passed_filter}% of neighborhoods passed filtering."
            )
        # calculate weight per neighborhood based on its overlap with other
        # nhoods
        self.nhood_stats["nhood_weight"] = pval_corr.calculate_neighborhood_weights(
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
        # calculate the mean zscored distance of query samples to reference
        # for each neighborhood:
        mean_zscored_q_dist_to_ref = np.nanmean(
            self.mean_sample_dists_to_ref_zscored[len(self.ref_samples_all) :, :],
            axis=0,
        )
        # take only the nhoods that passed filtering:
        self.nhood_stats["mean_zscored_q_dist_to_ref"] = mean_zscored_q_dist_to_ref[
            self.nhood_stats.nhood_number
        ]
        # remember that this function has been successfully run:
        self._nhood_stats_calculated = True

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
            self.nhood_stats.loc[nhood] = kstest(
                rvs=r2r_dists, cdf=r2q_dists, alternative="greater"
            )[:2]
