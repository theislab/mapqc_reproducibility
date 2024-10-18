from scipy.spatial.distance import cdist
from sklearn.metrics import pairwise_distances
import pandas as pd
import numpy as np
from anndata import AnnData
from typing import Literal

# import jax
# from ott.geometry import pointcloud
# from ott.solvers.linear import sinkhorn


class Neighborhood:
    """
    Class to calculate a neighborhood for a given center cell, and calculate
    statistics on that neighborhood to be used for mapQC.

    Parameters
    ----------
    adata
        AnnData object that contains both query and reference cells
    index_cell
        Barcode of the center cell for the neighborhood to be calculated
        (should be a row name in adata.obs)
    k
        Number of neighbors to be included in the neighborhood (including
        index cell itself)
    emb_key
        Location of the integrated embedding into which the query was mapped.
        If set to "X", adata.X will be used. Otherwise, set to a key in
        adata.obsm
    sample_key
        adata.obs column name in which sample names are specified for each cell
    ref_q_key
        adata.obs column name in which is specified whether a cell is part of
        the reference or the query
    ref_cat
        Category name of reference cells in adata.obs[ref_q_key]
    query_cat
        Category name of query cells in adata.obs[ref_q_key]
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
    exclude_same_batch_for_ref_samples
        TO DO
    dataset_key
        TO DO (only needed if exclude_same_batch_for_ref_samples set to True)
    distance_type
        Whether to use main pairwise (cell-cell) euclidean distance to quantify
        distance between a pair of samples, or sinkhorn divergence (approximation
        of Wasserstein distance).
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


    Returns
    -------
    By default, determines k nearest neighbors and stores them under self.knn_idc
    (anndata row numbers of nearest neighbors) and self.knn_barcodes (anndata row names
    of nearest neighbors). Index cell itself is included in neighbors.
    Further stats are calculated using class functions.
    """

    def __init__(
        self,
        adata: AnnData,
        index_cell: str,
        k: int,
        min_n_cells_per_sample: int,
        min_n_samples_ref: int,
        min_n_samples_q: int,
        distance_type: Literal["pairwise_euclidean", "sinkhorn"],
        emb_key: str,
        sample_key: str = "sample",
        ref_q_key: str = "reference_or_query",
        ref_cat: str = "reference",
        query_cat: str = "query",
        dataset_key: str = None,
        exclude_same_batch_for_ref_samples: bool = True,
        use_adaptive_k: bool = True,
        max_k: int = None,
        adaptive_k_margin: float = 1.1,
    ):
        # store embedding as separate attribute
        if emb_key == "X":
            self.adata_emb = adata.X
        elif emb_key in adata.obsm.keys():
            self.adata_emb = adata.obsm[emb_key]
        else:
            raise ValueError(
                "emb_key should be set to either 'X' or a key in adata.obsm"
            )
        if distance_type not in ["pairwise_euclidean", "sinkhorn"]:
            raise ValueError(
                "Argument 'distance_type' should be set to either 'pairwise_euclidean' or 'sinkhorn'."
            )
        self.distance_type = distance_type
        # store parameters as attributes (add sanity checks here too?)
        self.min_n_cells_per_sample = min_n_cells_per_sample
        self.min_n_samples_ref = min_n_samples_ref
        self.min_n_samples_q = min_n_samples_q
        self.min_k = k
        self.exclude_same_batch_for_ref_samples = exclude_same_batch_for_ref_samples
        self.sample_key = sample_key
        self.dataset_key = dataset_key
        self.ref_q_key = ref_q_key
        self.ref_cat = ref_cat
        self.use_adaptive_k = use_adaptive_k
        self.max_k = max_k
        self.adaptive_k_margin = adaptive_k_margin
        # store adata as attribute (might not be necessary, not sure if an issue)
        self.adata = adata
        # store whether cell came from reference or query, and map to standard
        # terminology (i.e. use terms 'reference' and 'query')
        if ref_q_key not in adata.obs.columns:
            raise ValueError("Your ref_q_key is not present in adata.obs.columns.")
        if (
            ref_cat not in adata.obs[ref_q_key].values
            or query_cat not in adata.obs[ref_q_key].values
        ):
            raise ValueError(
                "Your ref_cat and query_cat should be values in your adata.obs ref_q_key column."
            )
        ref_q_harmonizer = {ref_cat: "reference", query_cat: "query"}
        self.adata.obs["mapqc_ref_or_q"] = (
            adata.obs[ref_q_key].map(ref_q_harmonizer).values
        )
        # check if index cell in adata
        if index_cell not in adata.obs.index:
            raise ValueError("Your index cell is not found in adata.obs.index.")
        # get row number in adata that has the index cell
        self.index_cell = index_cell
        self.index_cell_idx = np.where(adata.obs.index == index_cell)[0][0]
        # calculate distances of all cells to the index cell:
        self._dists_all_cells_to_center_cell = cdist(
            self.adata_emb[self.index_cell_idx, :].reshape(
                (1, self.adata_emb.shape[1])
            ),
            self.adata_emb,
        )[0]
        # get indices of columns (cell indices) sorted from small to large distance
        # to index cell (cell itself should therefore be first)
        self._cell_indices_sorted_by_dist = np.argsort(
            self._dists_all_cells_to_center_cell
        )
        # set the k
        if not use_adaptive_k:
            self.max_k = k
        self._ref_filter_info, self.final_k = self._calculate_adaptive_nhood_k()
        # if a k was found, store information about k nearest neighbors of index cell
        if not np.isnan(self.final_k):
            self._store_neighbor_info(self.final_k)
        # otherwise, store only the idc of the cell's nearest neighbors for later
        # analysis of filtering at cell level
        else:
            # store embedding of cells in nhood with default k:
            self.knn_idc = self._get_nearest_neighbors(k)

    def get_pairwise_sample_dists(self):
        """
        Calculate mean distances between all sample pairs in the neighborhood.
        Include filtering of samples with too few cells. Distances are
        Euclidean and calculated in the integrated embedding space in which
        the query was mapped. For each sample pair, the mean distance between
        the cells from the first sample and those from the second sample are
        calculated and considered the inter-sample distance for that pair.

        Returns
        -------
        pandas.DataFrame with ref samples as rows and all samples (reference
        and query) as columns, with the intersample distance specified for
        each pair. Distance to the same sample are set to np.nan. For
        neighborhoods that do not have enough reference or query samples
        (see filtering parameters in class), np.nan is returned.

        """
        # check if nhood passes min_n_samples_ref filter
        # precalculated when calculating the adaptive k:
        if not self._ref_filter_info == "pass":
            self._filter_info = self._ref_filter_info
            return np.nan
        # check which samples have sufficient cells in the neighborhood
        ref_samples = self._filter_samples("reference")
        q_samples = self._filter_samples("query")
        # check if enough samples from query are present in nhood:
        if len(q_samples) < self.min_n_samples_q:
            self._filter_info = "too few query samples"
            return np.nan
        self._filter_info = "pass"
        # create empty dataframe in which to store pairwise distances
        sample_dists = pd.DataFrame(index=ref_samples, columns=ref_samples + q_samples)
        # store sample to batch mapping if batch needs to be taken into account
        if self.exclude_same_batch_for_ref_samples:
            sample_to_batch = self.adata.obs.groupby(self.sample_key).agg(
                {self.dataset_key: "first"}
            )
        # loop through all ref samples (and count in which loop we are)
        for ref_sample_count, sample_1 in enumerate(ref_samples):
            # loop through all reference samples that that we haven't looped
            # through in the first loop yet, plus all query samples:
            for sample_2 in (ref_samples + q_samples)[ref_sample_count + 1 :]:
                if self.exclude_same_batch_for_ref_samples:
                    if (
                        sample_to_batch.loc[sample_1, self.dataset_key]
                        == sample_to_batch.loc[sample_2, self.dataset_key]
                    ):
                        # do not include distances to samples from same batch for
                        # reference samples
                        continue
                intersample_dist = self._calculate_sample_pair_distance(
                    sample_1, sample_2
                )
                sample_dists.loc[sample_1, sample_2] = intersample_dist
                if sample_2 in ref_samples:
                    sample_dists.loc[sample_2, sample_1] = intersample_dist
        return sample_dists

    def _calculate_adaptive_nhood_k(self):
        """
        Calculate the k needed to fulfill "diversity/number requirements", i.e.
        given the neighborhood's minimum number of reference samples and the minimum
        number of cells per sample, check how large k needs to be to reach these
        minimums.

        Parameters
        ----------
        nhood:
            instance of selfs.Neighborhood class

        Returns
        -------
        A tuple (filtering_info, adapted_k) specifying whether or not the self passed
        filtering given the max_k, and what the final adapted k is (np.nan if filtering
        not passed). Filtering info is either ..........The lowest value (integer) of k for which all filtering criteria are passed.
        If no value below max_k passes filtering, np.nan is returned.
        """
        # store some sample information (which batch it came from, and whether or not it
        # is from the reference) that we will need further down:
        sample_info = self.adata.obs.groupby(self.sample_key).agg(
            {self.dataset_key: "first", self.ref_q_key: "first"}
        )
        # check if sufficient reference samples in the neighborhod of default size k.
        # code here should be much faster than for adaptive k, hence we use this
        # whenever possible
        knn_samples = (
            self.adata.obs[self.sample_key]
            .iloc[self._cell_indices_sorted_by_dist[: self.min_k]]
            .value_counts()
        )
        knn_sample_set_suff_cells = knn_samples.loc[
            knn_samples >= self.min_n_cells_per_sample
        ].index
        (
            filter_info,
            suff_ref_samples_in_k_neighbors,
        ) = self._check_if_sufficient_ref_samples(
            knn_sample_set_suff_cells, sample_info
        )
        if (not self.use_adaptive_k) or suff_ref_samples_in_k_neighbors:
            return (filter_info, self.min_k)
        else:
            # for each sample, calculate at which neighbor (ordered by distance from the index
            # cell) its nth cell shows up, with n being the min_n_cells per sample. This will
            # help us calculate how many samples are "sufficiently present" (i.e. have at
            # least min_n_cells_per_sample cells) at a given k.
            def _get_nth_idx(list, value, n):
                inst_idc = [idx for idx, x in enumerate(list) if x == value]
                if len(inst_idc) < n:
                    return (
                        len(list) - 1
                    )  # as if only the last cell is the nth instance of the sample
                else:
                    nth_inst_idx = inst_idc[
                        n - 1
                    ]  # -1 because of python indexing: 1st cell is at index 0
                    return nth_inst_idx

            # first, get sample names for all cells in adata, ordered by ascending distance from
            # index cell.
            samples_ordered = (
                self.adata.obs[self.sample_key]
                .iloc[self._cell_indices_sorted_by_dist]
                .values
            )
            # now, for each sample calculate at which neighbor number it sees its nth instance, and
            # order the output by closest to furthest sample
            sample_nth_inst_idc = pd.Series(
                {
                    sample: _get_nth_idx(
                        samples_ordered, sample, self.min_n_cells_per_sample
                    )
                    for sample in set(samples_ordered)
                }
            ).sort_values()
            # loop through sample sets of increasing size (given nearest neighbors' samples)
            # and check if they pass our min_n_sample criteria. Stop once we have found
            # a set of samples that fulfills filtering criteria, or once we have passed our
            # maximum k
            suff_ref_samples_with_increased_k = False
            sample_to_check_idx = len(knn_sample_set_suff_cells) + 1
            while (not suff_ref_samples_with_increased_k) and round(
                sample_nth_inst_idc.iloc[sample_to_check_idx] * self.adaptive_k_margin
            ) <= self.max_k:
                (
                    filter_info,
                    suff_ref_samples_with_increased_k,
                ) = self._check_if_sufficient_ref_samples(
                    sample_nth_inst_idc.index[:sample_to_check_idx],
                    sample_info,
                )
                if not suff_ref_samples_with_increased_k:
                    sample_to_check_idx += 1
            if suff_ref_samples_with_increased_k:
                # return filtering info (passed filtering), and the neighbor index at which
                # this sample has min_n_cells_per_sample, i.e. the suggested minimum k
                return (
                    "pass",
                    int(
                        round(
                            sample_nth_inst_idc.iloc[sample_to_check_idx]
                            * self.adaptive_k_margin
                        )
                    ),
                )
            else:
                # if not sufficient samples were found under max_k, return np.nan
                return (filter_info, np.nan)

    def _check_if_sufficient_ref_samples(
        self,
        sample_set: list,
        sample_info_df: pd.DataFrame,
    ):
        """
        Checks if a given set of samples present in a neighborhood fulfills the minimum
        requirements for the number of reference samples in a neighborhood.

        Parameters
        ----------
        sample_set
            List of samples present in a neighborhood, already filtered based on minimum
            number of cells per sample (i.e. samples with fewer cells in the nhood are not
            included in this list
        sample_info_df
            DataFrame with samples as index, specifying whether a sample came from the
            reference or query, and (if self.exclude_same_batch_for_ref_samples is True)
            which batch the sample came from.

        Returns
        -------
        A tuple with a string with filtering information, and a boolean specifying whether
        or not the neighborhood (sample set) passed filtering.
        """
        # subset to reference samples:
        sample_set_ref_only = [
            sample
            for sample in sample_set
            if sample_info_df.loc[sample, self.ref_q_key] == self.ref_cat
        ]
        if len(sample_set_ref_only) < self.min_n_samples_ref:
            return ("too few reference samples", False)
        ref_sample_batches = sample_info_df.loc[
            sample_set_ref_only, self.dataset_key
        ].values
        if self.exclude_same_batch_for_ref_samples:
            # check per ref sample if enough ref samples from other batches are present
            # to calculate ref to ref distances
            n_passed_filter = 0
            sufficient_ref_samples = False
            for s, s_batch in zip(sample_set_ref_only, ref_sample_batches):
                if n_passed_filter < self.min_n_samples_ref:
                    n_samples_outside_batch = (ref_sample_batches != s_batch).sum()
                    # could make the below condition less strict, e.g. consider
                    # only a single sample outside of batch sufficient per ref
                    # sample
                    if n_samples_outside_batch > (self.min_n_samples_ref - 1):
                        n_passed_filter += 1
                else:
                    return ("pass", True)
            return ("too few reference samples from different batches", False)
        else:
            return ("pass", True)

    def _store_neighbor_info(self, k: int):
        """
        Store information about the k nearest neighbors of the neighborhoods
        index cell (including index cell itself).
        """
        # get nearest neighbors of index cell, i.e. the cells in this nhood:
        self.knn_idc = self._get_nearest_neighbors(k)
        self.knn_barcodes = self.adata.obs.iloc[self.knn_idc, :].index
        # get sample names of cells in neighborhoods
        self.knn_samples = self.adata.obs.loc[self.knn_barcodes, self.sample_key].values
        self.knn_ref_or_q = self.adata.obs.loc[
            self.knn_barcodes, "mapqc_ref_or_q"
        ].values
        # store embedding of cells in nhood:
        self.knn_emb = self.adata_emb[self.knn_idc, :]
        if self.distance_type == "pairwise_euclidean":
            # calculate pairwise distances between cells in nhood:
            self.dists = pairwise_distances(self.knn_emb)

    def _get_nearest_neighbors(self, k: int):
        """
        Get k nearest neighbors of a given index cell, including index cell itself.

        Returns:
        -------
        np.array of the indices (row numbers) in adata of the index cell's k nearest
        neighbors, ordered from closest to furtheset away (index cell itself should i
        therefore be first).

        """
        # get column numbers for the columns with the k lowest values, i.e.
        # the k nearest neighbors (including the index cell itself)
        return self._cell_indices_sorted_by_dist[:k]

    def _filter_samples(self, ref_or_q: Literal["reference", "query"]):
        """
        Filters samples based on the number of cells of the sample that are present
        in the neighborhood. Does this for either the reference or the query samples.

        Parameters
        ----------
        ref_or_q
            Whether to filter reference samples or query samples

        Returns
        -------
        List of sample names that passed filtering

        """
        if ref_or_q not in ["reference", "query"]:
            raise ValueError("ref_or_q should be set to either 'reference' or 'query'.")
        sample_cell_counts = self.knn_samples[
            self.knn_ref_or_q == ref_or_q
        ].value_counts()
        sample_set_filtered = sample_cell_counts.index[
            sample_cell_counts >= self.min_n_cells_per_sample
        ].tolist()
        return sample_set_filtered

    def _calculate_sample_pair_distance(self, sample_1: str, sample_2: str):
        """
        Calculate mean distance between all cells of one sample and all
        cells of another sample, based on previously calculated cell-cell
        distances for all cells in the neighborhood.

        Parameters
        ----------
        sample_1
            Name of first sample in sample pair for which to calculate mean distance
        sample_2
            Name of second sample in sample pair

        Returns
        -------
        Mean distance (float) between cells of sample_1 and cells of sample_2

        """
        # check which cells in the neigbhorhood belong to each sample
        sample_1_cells = np.where(self.knn_samples == sample_1)[0]
        sample_2_cells = np.where(self.knn_samples == sample_2)[0]
        if self.distance_type == "pairwise_euclidean":
            # subset nhood's distance matrix to cells from the two samples
            pair_specific_dists = self.dists[sample_1_cells][:, sample_2_cells]
            return np.mean(pair_specific_dists)
        # TO DO: USE MOSCOT IMPLEMENTATION INSTEAD OF TUTORIAL IMPLEMENTATION!
        # (their output.divergence is different from our output.kl_reg_cost)
        # also: this code is waaaayyyy too slow; TO DO: PARALLELIZE CALCULATION
        # ACROSS SAMPLE PAIRS!
        # elif self.distance_type == "sinkhorn":
        #     sample_1_cell_coords = self.knn_emb[sample_1_cells, :]
        #     sample_2_cell_coords = self.knn_emb[sample_2_cells, :]
        #     geom = pointcloud.PointCloud(
        #         sample_1_cell_coords, sample_2_cell_coords, epsilon=10
        #     )
        #     sinkhorn_solve_fn = jax.jit(sinkhorn.solve)
        #     sinkhorn_out = sinkhorn_solve_fn(geom)
        #     # check if converged:
        #     if not sinkhorn_out.converged:
        #         print(
        #             f"Warning: sinkhorn solver did not converge for {self.index_cell}, sample {sample_1} and sample {sample_2}."
        #         )
        #     return sinkhorn_out.kl_reg_cost
