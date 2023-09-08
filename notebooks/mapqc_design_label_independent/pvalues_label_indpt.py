import scanpy as sc
import numpy as np
import pandas as pd
import random
from scipy.stats import kstest
from scipy.sparse import csr_matrix, save_npz, load_npz
from scipy.spatial.distance import cdist
import pval_corr

def get_highest_column_numbers(matrix, n):
    sorted_indices = np.argsort(matrix, axis=1)[:, ::-1]
    highest_columns = sorted_indices[:, :n]
    return highest_columns

def mean_per_row_group(matrix, grouping):
    unique_groups = np.unique(grouping)
    means_per_group = np.zeros((matrix.shape[0], unique_groups.shape[0]))

    for group_number, group in enumerate(unique_groups):
        mask = grouping == group
        group_sum = np.nansum(matrix * mask, axis=1)
        group_count = np.nansum(mask, axis=1)
        mean_per_row = group_sum / group_count
        means_per_group[:, group_number] = mean_per_row
    return unique_groups, means_per_group


def keep_k_smallest_distances(distances, n_neighbors, exclude_null_dist, sparse=False):
    """
    Selects the closest `n_neighbors` neighbors and sets all the other
    distances to 0.
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

def get_pvalues(adata_ext_h,k,prop_cells,condition="Healthy",filter_samples=False,filter_nhoods=False):
    #subsample tmp_adata to condition
    tmp_adata = adata_ext_h[(adata_ext_h.obs.core_or_extension == "core")|(adata_ext_h.obs.lung_condition==condition)].copy()
    #subsampling center cells
    n_cells = tmp_adata.shape[0]
    print(f"Number of cells:", n_cells)
    n_cells_to_sample = round(n_cells / prop_cells)
    print(f"Sample {n_cells_to_sample} cells")

    cell_subset_idc = random.sample(range(n_cells), k=n_cells_to_sample)
    cell_subset_barcodes = tmp_adata.obs.iloc[cell_subset_idc, :].index

    # Euclidean distances w scipy

    tmp_adata.obs["cell_sampled"] = "no"
    tmp_adata.obs.loc[cell_subset_barcodes, "cell_sampled"] = "yes"

    #get distances
    #dist_ext_subset = tmp_adata.obsp["distances"][cell_subset_idc, :]
    dist_ext_subset= cdist(tmp_adata[cell_subset_idc, :].X, tmp_adata.X)
    dist_ext_subset = keep_k_smallest_distances(
        distances=dist_ext_subset,
        n_neighbors=k,
        exclude_null_dist=True,
        sparse=False,
    )


    neighbors_idc = get_highest_column_numbers(dist_ext_subset, k - 1)
    neighbors_distances = dist_ext_subset[
    np.arange(n_cells_to_sample)[:, np.newaxis],neighbors_idc,]
    neighbors_distances=neighbors_distances

    #get name of sample origin
    neighbors_samples = np.array(tmp_adata.obs["sample"].values)[neighbors_idc]
    approved_samples={}
    for center,neighborhood in zip(cell_subset_barcodes,neighbors_samples):
        unique_samples,counts = np.unique(neighborhood, return_counts=True)
        idx = counts>4
        approved_samples[center] = unique_samples[idx]
    
    samples_ext, means_per_sample_ext = mean_per_row_group(neighbors_distances, neighbors_samples)
    
    #store mean distances per sample
    res_healthy = pd.DataFrame(data=means_per_sample_ext.T, index=samples_ext, columns=cell_subset_barcodes)

    res_healthy_flat = res_healthy.unstack()
    res_healthy_flat = pd.DataFrame(res_healthy_flat)
    res_healthy_flat.reset_index(level=1, inplace=True)
    res_healthy_flat.rename(
        columns={"level_1": "sample", 0: "mean_distance_to_index_cell"}, inplace=True
    )
    res_healthy_flat["neighborhood"] = res_healthy_flat.index
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

    #for conditions
    res_healthy_flat["cond"] = "Healthy"


    if filter_samples:
        res_healthy_flat= res_healthy_flat[
            [s in approved_samples[n] 
            for s,n in zip(res_healthy_flat["sample"],res_healthy_flat.neighborhood)]
            ].copy()

    '''conditions= tmp_adata.obs.lung_condition.cat.categories
    for condition in conditions:
            cond_samples = tmp_adata.obs.loc[tmp_adata.obs.lung_condition == condition, "sample"].unique()
            res_healthy_flat.loc[res_healthy_flat["sample"].isin(cond_samples), "cond"] = condition'''


    
    
    #stats

    neighborhood_stats = pd.DataFrame(index=cell_subset_barcodes, columns=["KS_stat", "pvalue"])

    #calculate stats
    # loop through neighborhoods
    for index_cell in cell_subset_barcodes:
        # check if index cell is from query or ref
        index_source = tmp_adata.obs.loc[index_cell, "core_or_extension"]
        #index_condition = tmp_adata.obs.loc[index_cell, "lung_condition"]
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
        # test one-sided, query greater than reference for neighborhoods with ref
        # index cells. Store results in dataframe
        threshold = 1
        if filter_nhoods:
            threshold=3
        if len(ref_sample_results.values) < threshold or len(query_sample_results.values) < threshold: #filtering for sample number
            # %print(index_cell)
            continue
        if index_source == "core":
            # test if query distances are larger than reference:
            neighborhood_stats.loc[index_cell, :] = kstest(
                rvs=ref_sample_results.values,
                cdf=query_sample_results.values,
                alternative="greater",
            )
        # test one-sided, but now query smaller than reference for neighborhoods
        # with query index cells. Store results in dataframe
        elif index_source == "extension":
            # test if query distances are larger than reference:
            neighborhood_stats.loc[index_cell, :] = kstest(
                rvs=ref_sample_results.values,
                cdf=query_sample_results.values,
                alternative="less",
            )
    # perform weighted p-value correction
    # determine weights based on overlap between neighborhoods (the more 
    # overlap, the lower the weight)
    nhood_weights = pval_corr.calculate_neighborhood_weights(neighbors_idc, tmp_adata.n_obs) 
    neighborhood_stats["nhood_weight"] = nhood_weights
    # calculate weighted fdr-corrected pvalues
    neighborhood_stats["pvalue_corr"] = pval_corr.weighted_fdr_corr_pvals(
            neighborhood_stats.pvalue.values, weights=nhood_weights
        )

    return res_healthy,neighborhood_stats
