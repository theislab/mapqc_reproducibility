import random
from typing import Literal

import numpy as np
import pandas as pd
import pval_corr
import scanpy as sc
from anndata import AnnData
from scipy.sparse import csr_matrix, load_npz, save_npz
from scipy.spatial.distance import cdist
from scipy.stats import kstest


def get_highest_column_numbers(matrix: np.array, n: int):
    """
    Computes the indices of the highest values in a distance matrix,
    row-wise.

    Parameters
    ----------
    matrix
        Array of size (n_index, k) with euclidean distances
        between embeddings.
    n
        The highest 1 <= n <= k distances to extract the indices from.

    Returns
    -------
    Array of size (n_index, n) of indices for each index cell.
    """
    sorted_indices = np.argsort(matrix, axis=1)[:, ::-1]
    highest_columns = sorted_indices[:, :n]
    return highest_columns


def mean_per_row_group(matrix: np.array, grouping: np.array):
    """
    Calculates the mean distance of each cell to each sample.

    Parameters
    ----------
    matrix
        Array of size (n_index, k) with euclidean distances
        between embeddings.
    grouping
        Array of size (n_index, k) with sample id corresponding to each neighbour.

    Returns
    -------
    unique_groups
        1D array of unique sample ids.
    means_per_group
        2D array of size (n_index, len(unique_groups)) with average distances of
        index cells from each of the unique samples.
    """
    unique_groups = np.unique(grouping)
    means_per_group = np.zeros((matrix.shape[0], unique_groups.shape[0]))

    for group_number, group in enumerate(unique_groups):
        mask = grouping == group
        group_sum = np.nansum(matrix * mask, axis=1)
        group_count = np.nansum(mask, axis=1)
        mean_per_row = group_sum / group_count
        means_per_group[:, group_number] = mean_per_row
    return unique_groups, means_per_group


def keep_k_smallest_distances(
    distances: np.array, n_neighbors: int, exclude_null_dist: bool, sparse: bool = False
):
    """
    Selects the closest `n_neighbors` neighbors and sets all the other
    distances to 0.

    Parameters
    ----------
    distances
        Array of size (n_index, k) with euclidean distances
        between embeddings.
    n_neighbors
        the closest 1 <= n_neighbors <= k distances to keep.
    exclude_null_dist
        whether to exclude null distances from the subsetting operation.
    sparse
        whether to return a sparse matrix in `csr` format.

    Returns
    -------
    2D Array of size (n_index, k) where distances greater than the `n_neighbors`
    closest distances are set to 0.
    """
    distances_copy = distances.copy()

    if exclude_null_dist:
        max_distance = np.max(distances_copy) + 1
        distances_copy[distances_copy == 0] = max_distance

    sorted_distances = np.sort(distances_copy, axis=1)
    mask = distances < sorted_distances[:, n_neighbors - 1].reshape(-1, 1)
    distances *= mask
    if sparse:
        distances = csr_matrix(distances)
    return distances


def get_pvalues(
    adata_ext_h: AnnData,
    k: int,
    prop_cells: float,
    condition: Literal["Healthy", "Disease"],
    filter_samples: bool = False,
    filter_nhoods: bool = False,
):
    """
    Parameters
    ----------
    adata_ext_h
        AnnData object including "reference" and "query" observations.
    k
        Size of the neighborhood.
    prop_cells
        Frequency of sampling from the query (i.e. 1 cell sampled every `prop_cells`).
    condition
        Lung condition (or its absence for "Healthy") in the query.
    filter_samples
        Whether to exclude samples on the cutoffs specified below.
    filter_nhoods
        Whether to exclude neighborhoods on the cutoffs specified below.

    Returns
    -------
    res_healthy
        A pd.DataFrame summarizing features of the neighborhoods.
    neighborhood_stats
        A pd.DataFrame including KS test statistics, FDR-corrected pvalues
        and neighborhood weights.

    """
    # subsample `adata_ext_h` to condition
    tmp_adata = adata_ext_h[
        (adata_ext_h.obs.core_or_extension == "core")
        | (adata_ext_h.obs.lung_condition == condition)
    ].copy()

    if len(tmp_adata) < k:
        raise ValueError(
            f"`k` ({k}) cannot be greater than the number of observations ({len(tmp_adata)})."
        )

    # Subsample center cells from query
    n_cells_query = tmp_adata[
        tmp_adata.obs.core_or_extension
        == "extension"  # TODO: expand to include different labels
    ].shape[0]
    print(f"Number of cells:", n_cells_query)
    if n_cells_query < prop_cells:
        raise ValueError(
            f"Argument `prop_cells` ({prop_cells}) cannot be greater than the size of the query ({n_cells_query})"
        )
    n_cells_to_sample = round(n_cells_query / prop_cells)
    print(f"Sample {n_cells_to_sample} cells")

    cell_subset_idc = random.sample(range(n_cells_query), k=n_cells_to_sample)
    cell_subset_barcodes = (
        tmp_adata.obs[
            tmp_adata.obs.core_or_extension
            == "extension"  # TODO: expand to include different labels
        ]
        .iloc[cell_subset_idc, :]
        .index
    )  # contains only query now

    # Euclidean distances w scipy

    tmp_adata.obs["cell_sampled"] = "no"
    tmp_adata.obs.loc[cell_subset_barcodes, "cell_sampled"] = "yes"

    # get distances
    # dist_ext_subset = tmp_adata.obsp["distances"][cell_subset_idc, :]
    dist_ext_subset = cdist(tmp_adata[cell_subset_barcodes].X, tmp_adata.X)
    dist_ext_subset = keep_k_smallest_distances(
        distances=dist_ext_subset,
        n_neighbors=k,
        exclude_null_dist=True,  # TODO: consider removal in a subsequent versions
        sparse=False,
    )

    neighbors_idc = get_highest_column_numbers(dist_ext_subset, k - 1)
    neighbors_distances = dist_ext_subset[
        np.arange(n_cells_to_sample)[:, np.newaxis],
        neighbors_idc,
    ]
    neighbors_distances = neighbors_distances  # @Mariam that seems a redundancy to me

    # get name of sample origin
    neighbors_samples = np.array(tmp_adata.obs["sample"].values)[neighbors_idc]
    approved_samples = {}
    for center, neighborhood in zip(cell_subset_barcodes, neighbors_samples):
        unique_samples, counts = np.unique(neighborhood, return_counts=True)
        idx = counts > 4
        approved_samples[center] = unique_samples[idx]

    samples_ext, means_per_sample_ext = mean_per_row_group(
        neighbors_distances, neighbors_samples
    )

    # store mean distances per sample
    # `res_healthy` is not strictly accurate here; see argument `condition`
    res_healthy = pd.DataFrame(
        data=means_per_sample_ext.T, index=samples_ext, columns=cell_subset_barcodes
    )

    res_healthy_flat = res_healthy.unstack()
    res_healthy_flat = pd.DataFrame(res_healthy_flat)
    res_healthy_flat.reset_index(level=1, inplace=True)
    res_healthy_flat.rename(
        columns={"level_1": "sample", 0: "mean_distance_to_index_cell"}, inplace=True
    )
    res_healthy_flat["neighborhood"] = res_healthy_flat.index

    # I leave this field here for debugging/re-adapting purposes
    res_healthy_flat["index_cell_origin"] = tmp_adata.obs.loc[
        res_healthy_flat.neighborhood, "core_or_extension"
    ]

    res_healthy_flat["neighborhood, cell type, ref or q"] = [
        f"{nh}, {tmp_adata.obs.loc[nh,'ann_finest_level']} ({tmp_adata.obs.loc[nh,'core_or_extension']})"
        for nh in res_healthy_flat.neighborhood
    ]
    res_healthy_flat["ref_or_q"] = "ref"

    q_samples = tmp_adata.obs.loc[
        tmp_adata.obs.core_or_extension == "extension", "sample"
    ].unique()

    res_healthy_flat.loc[res_healthy_flat["sample"].isin(q_samples), "ref_or_q"] = "q"

    # for conditions
    res_healthy_flat[
        "cond"
    ] = "Healthy"  # FIXME: maybe we want to change this to `condition`

    if filter_samples:
        res_healthy_flat = res_healthy_flat[
            [
                s in approved_samples[n]
                for s, n in zip(
                    res_healthy_flat["sample"], res_healthy_flat.neighborhood
                )
            ]
        ].copy()

    """conditions= tmp_adata.obs.lung_condition.cat.categories
    for condition in conditions:
            cond_samples = tmp_adata.obs.loc[tmp_adata.obs.lung_condition == condition, "sample"].unique()
            res_healthy_flat.loc[res_healthy_flat["sample"].isin(cond_samples), "cond"] = condition"""

    # stats

    neighborhood_stats = pd.DataFrame(
        index=cell_subset_barcodes, columns=["KS_stat", "pvalue"]
    )

    # calculate stats
    # loop through neighborhoods
    for index_cell in cell_subset_barcodes:
        # index_source = tmp_adata.obs.loc[index_cell, "core_or_extension"]
        # extract mean distances from query samples to index cell.
        # drop nas, i.e. ignore samples not observed in the neighborhood
        query_sample_results = res_healthy_flat.loc[
            [
                nh == index_cell and ref_or_q == "q"
                for nh, ref_or_q in zip(
                    res_healthy_flat.neighborhood, res_healthy_flat.ref_or_q
                )
            ],
            "mean_distance_to_index_cell",
        ].dropna()
        # get same results for reference samples
        ref_sample_results = res_healthy_flat.loc[
            [
                nh == index_cell and ref_or_q == "ref"
                for nh, ref_or_q in zip(
                    res_healthy_flat.neighborhood, res_healthy_flat.ref_or_q
                )
            ],
            "mean_distance_to_index_cell",
        ].dropna()

        threshold = 1

        if filter_nhoods:
            threshold = 3
        if (
            len(ref_sample_results.values) < threshold
            or len(query_sample_results.values) < threshold
        ):  # filtering for sample number
            continue
        """
        # test one-sided, query greater than reference for neighborhoods with ref
        # index cells. Store results in dataframe
        if index_source == "core":
            # test if query distances are larger than reference:
            neighborhood_stats.loc[index_cell, :] = kstest(
                rvs=ref_sample_results.values,
                cdf=query_sample_results.values,
                alternative="greater",
            )
        """
        # test one-sided, query smaller than reference for neighborhoods
        # with query index cells. Store results in dataframe
        # elif index_source == "extension":
        # test if query distances are larger than reference:
        neighborhood_stats.loc[index_cell, :] = kstest(
            rvs=ref_sample_results.values,
            cdf=query_sample_results.values,
            alternative="less",
        )
    # perform weighted p-value correction
    # determine weights based on overlap between neighborhoods (the more
    # overlap, the lower the weight)
    nhood_weights = pval_corr.calculate_neighborhood_weights(
        neighbors_idc, tmp_adata.n_obs
    )
    neighborhood_stats["nhood_weight"] = nhood_weights
    # calculate weighted fdr-corrected pvalues
    neighborhood_stats["pvalue_corr"] = pval_corr.weighted_fdr_corr_pvals(
        neighborhood_stats.pvalue.values, weights=nhood_weights
    )

    return res_healthy, neighborhood_stats
