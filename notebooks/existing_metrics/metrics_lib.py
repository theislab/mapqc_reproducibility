import anndata as ad
import numpy as np
import scanpy as sc
import pandas as pd

# We implement a lot of things based on reimplementation of scib by YosefLab
# https://github.com/YosefLab/scib-metrics/tree/main
from scib_metrics import lisi_knn, kbet
from scib_metrics.utils import silhouette_samples
from scib_metrics.nearest_neighbors import pynndescent

from time_logger import log_time


def calculate_neighbors(adata, obsm_emb_key, k=100, n_jobs=1, neighbor_computer=None):
    # Calculates `k` nearest neighbors in the space defined by `obsm_emb_key` of `adata`
    # One can pass `neighbor_computer` as in scib_metrics. Default is good enough.
    if neighbor_computer is not None:
        neigh_output = neighbor_computer(adata.obsm[obsm_emb_key], k)
    neigh_output = pynndescent(adata.obsm[obsm_emb_key], n_neighbors=k, random_state=0, n_jobs=n_jobs)
    indices, distances = neigh_output.indices, neigh_output.distances
    return indices, distances


def get_results(per_cell_scores, adata, group_keys=None, sep_key=None, reduce='mean', after_function=None):
    # Gets per cell scores and prduces group-wise scores.
    # This function is applied after aggregation
    if after_function is None:
        after_function = lambda x: x

    res_pd = adata.obs.copy()
    res_pd['score'] = per_cell_scores
    results = {'per_cell': res_pd['score'].apply(after_function)}
    if group_keys is not None:
        for title, obs_key in group_keys.items():
            if sep_key is None:
                if reduce == 'mean':
                    results[title] = res_pd.groupby(obs_key).mean(['score'])['score'].apply(after_function)
                elif reduce == 'median':
                    results[title] = res_pd.groupby(obs_key).median(['score'])['score'].apply(after_function)
                else:
                    raise NotImplementedError()
            else:
                if reduce == 'mean':
                    results[title] = (
                        res_pd.groupby([obs_key, sep_key]).mean(['score']).reset_index()
                        .groupby(obs_key).mean(['score'])['score'].apply(after_function)
                    )
                elif reduce == 'median':
                    results[title] = (
                        res_pd.groupby([obs_key, sep_key]).mean(['score']).reset_index()
                        .groupby(obs_key).median(['score'])['score'].apply(after_function)
                    )
                else:
                    raise NotImplementedError()
    return results


def calculate_asw_batch(adata, obsm_emb_key, batch_key, group_keys=None,
                        rescale=True, chunk_size=256):
    # Calculates ASW within one population (between query and reference in our case)
    # This metric is meaningless in our context and should be revised (e.g. consider celltypes and use the silhouette_batch function).
    # Based on silhouette_samples @ https://github.com/YosefLab/scib-metrics/blob/5681f7ef6b6d4cfd04fda90d1d33fab1dbe1bb50/src/scib_metrics/_silhouette.py#L7C4-L7C4
    latent = adata.obsm[obsm_emb_key].copy()
    batch = adata.obs[batch_key].copy()
    sw_per_cell = silhouette_samples(latent, batch, chunk_size=chunk_size)
    sw_per_cell = np.abs(sw_per_cell)

    if rescale:
        sw_per_cell = 1 - sw_per_cell

    return get_results(sw_per_cell, adata, group_keys)


def calculate_lisi_batch(adata, ng_indices, ng_distances, batch_key, group_keys=None,
                         k=90, perplexity=None, rescale=True):
    # Based on https://github.com/YosefLab/scib-metrics/blob/5681f7ef6b6d4cfd04fda90d1d33fab1dbe1bb50/src/scib_metrics/_lisi.py#L40
    # Calculates lisi between query and reference (in our case batch_key should be core_or_extension)
    # Note: if perplexity is None it is initialized as np.floor(knn_idx.shape[1] / 3) in scib-metrics (check this: https://github.com/YosefLab/scib-metrics/blob/5681f7ef6b6d4cfd04fda90d1d33fab1dbe1bb50/src/scib_metrics/_lisi.py#L19).
    sp_distances, sp_conns = sc.neighbors._compute_connectivities_umap(
        ng_indices[:, :k], ng_distances[:, :k], adata.n_obs, n_neighbors=k
    )
    batch = adata.obs[batch_key].copy()
    lisi = lisi_knn(sp_distances, batch, perplexity=perplexity)

    if rescale:
        nbatches = len(np.unique(batch))
        after_function = lambda x: (x - 1) / (nbatches - 1)
    else:
        after_function = None

    # We use reduce='median' as in https://github.com/YosefLab/scib-metrics/blob/5681f7ef6b6d4cfd04fda90d1d33fab1dbe1bb50/src/scib_metrics/_lisi.py#L66
    return get_results(lisi, adata, group_keys, reduce='median', after_function=after_function)


def calculate_kbet_batch(adata, ng_indices, ng_distances, batch_key, group_keys=None,
                         k=50, alpha=0.05):
    # Uses: https://github.com/YosefLab/scib-metrics/blob/5681f7ef6b6d4cfd04fda90d1d33fab1dbe1bb50/src/scib_metrics/_kbet.py#L43
    # Calculates KBET between query and reference (in our case batch_key should be core_or_extension)
    # In original implementation they mean over cells of each label. then they mean over all labels.
    # YosefLab scib implementation with labels: https://github.com/YosefLab/scib-metrics/blob/5681f7ef6b6d4cfd04fda90d1d33fab1dbe1bb50/src/scib_metrics/_kbet.py#L98
    # Here we use kbet only as a single cell-type. But it should ligically be fine.
    sp_distances, sp_conns = sc.neighbors._compute_connectivities_umap(
        ng_indices[:, :k], ng_distances[:, :k], adata.n_obs, n_neighbors=k
    )
    batch = adata.obs[batch_key].copy()
    # We use p_values as we are interested in fine grained metrics
    acceptance_rate, test_statistics, p_values = kbet(sp_distances, batch, alpha=alpha)
    scores = (p_values >= alpha).astype(int)
    # Mean is the correct logic
    return get_results(scores, adata, group_keys, reduce='mean')



def spearmanr_respect_sorted(x):
    # Note wikipedia for the source of this formula:
    # https://en.wikipedia.org/wiki/Spearman%27s_rank_correlation_coefficient#Definition_and_calculation
    # it is assumed taht x values are not equal which is a gair assumption here.
    n = x.shape[0]
    p = x.argsort()
    return 1 - (6 * ((p - np.arange(n)) ** 2).sum()) / (n ** 3 - n)
# To test above method simply run:
# import numpy as np
# impoty scipy
# a, b = np.arange(100), np.random.uniform(size=100)
# print(scipy.stats.spearmanr(a, b).statistic)
# print(spearmanr_respect_sorted(b))


def calculate_wiq_knn_correlations(adata_query, obsm_emb_key, original_emb_key=None, group_keys=None,
                                   n_pc=50, k=500):
    # Note: adata_query.X should be log normalized.
    # Note: This implementation is new. The original implementation in R: https://rdrr.io/github/immunogenomics/symphony/src/R/knncorr.R
    # This is more efficient compared to the R code. They iterate over each anchor point!

    # 3 lines: Original space KNN calculation
    original_ad_x = ad.AnnData(adata_query.X if original_emb_key is None else adata_query.obsm[original_emb_key],
                               obs=adata_query.obs.copy())
    sc.tl.pca(original_ad_x, n_comps=n_pc, zero_center=True)
    q_ng_indices, q_ng_distances = calculate_neighbors(original_ad_x, 'X_pca', k=k)  # N x K

    # Then we will find distances of the same KNN in the new embedded space.
    # For this we need NxK pairs of points. Each pair has (anchor, neighbor)
    # Creake (nxK, ) anchor array.
    anchor_indices = np.tile(np.arange(original_ad_x.n_obs).reshape(-1, 1), (1, k)).flatten()  # N*k (one dimentional)
    # Creake (nxK, ) neighbor array.
    neighbor_indices = q_ng_indices.flatten()  # N*k (one dimentional)
    # Calculate pairwise distances between anchors and neighbors
    embeded_distances = np.linalg.norm(adata_query.obsm[obsm_emb_key][anchor_indices] - adata_query.obsm[obsm_emb_key][neighbor_indices], axis=1)   # N*k (one dimentional)
    # reshape distances in new space so that it matches the shape of q_ng_distances
    embeded_distances = embeded_distances.reshape(-1, k)  # N x K
    # I have already checked that the below code is same as q_ng_distances. So above code is perfect.
    # original_distances = np.linalg.norm(original_ad_x.obsm['X_pca'][anchor_indices] - original_ad_x.obsm['X_pca'][neighbor_indices], axis=1).reshape(-1, n)

    # Calculate spearmanr for pairwise orders. Since KNN in original space is sorted, we calculate spearman respect sorted array.
    scores = np.apply_along_axis(spearmanr_respect_sorted, 1, embeded_distances)

    # return results and sumamrise by mean.
    return get_results(scores, adata_query, group_keys, reduce='mean')


def calculate_cluster_preservation_score(adata_query, obsm_emb_key, original_emb_key=None, group_keys=None,
                                         ds_amount=5000, n_pc=50, k=20, cluster_res=0.6, eps=1e-8):
    # Original Implementation in R
    # https://rdrr.io/github/satijalab/azimuth/src/R/azimuth.R#sym-ClusterPreservationScore
    
    # Reformat adata_query so that .X is original space and .obs[obsm_emb_key] contains embedded space.
    adata_good_format = ad.AnnData(adata_query.X if original_emb_key is None else adata_query.obsm[original_emb_key],
                                   obs=adata_query.obs.copy())
    adata_good_format.obsm[obsm_emb_key] = adata_query.obsm[obsm_emb_key]
    adata_query = adata_good_format

    # 4 lines: Data preparation + clustering in original space
    # data is normalized and log-transformed but not sc.pp.scaled.
    sc.pp.scale(adata_query, max_value=10)
    sc.tl.pca(adata_query, n_comps=n_pc, zero_center=True)
    sc.pp.neighbors(adata_query, n_pcs=n_pc, use_rep='X_pca')
    # To be same as original version. we can also change to leiden. should we?
    sc.tl.louvain(adata_query, resolution=cluster_res, key_added='cluster')

    # Function to calculate entropy of clsters in original space in a new space defined by obsm_key
    def calculate_entropy(adata, obsm_key, cluster_codes, k):
        ng_indices, ng_distances = calculate_neighbors(adata, obsm_key, k=k)  # get KNN for each cell
        ng_shape = ng_indices.shape
        ng_labels = cluster_codes[ng_indices.flatten()].reshape(*ng_shape)    # get labels for KNNs of each cell
        px = np.apply_along_axis(np.bincount, axis=1, arr=ng_labels, minlength=np.max(ng_labels) + 1)  # Counts occurrences of each label
        return -np.nansum(px * np.log(px + eps), axis=1)  # calculate label entropy for each cell

    cluster_codes = adata_query.obs['cluster'].cat.codes.to_numpy()
    int_ent = calculate_entropy(adata_query, obsm_emb_key, cluster_codes, k=k)   # Entropy in new space
    orig_ent = calculate_entropy(adata_query, 'X_pca', cluster_codes, k=k)       # Entropy in original space

    # Py equivalent to original implementation. But we need fine grained info.
    # def align_stat(stat):
    #     corrected_stat = np.where(
    #         stat <= 0,
    #         5.0,
    #         np.clip(-1 * np.log2(stat + eps), 0.0, 5.0)
    #     )
    #     if isinstance(stat, pd.Series):
    #         return pd.Series(corrected_stat, index=stat.index)
    #     return corrected_stat
    # stat = np.median(
    #     pd.DataFrame({'entropy': orig_ent, 'cluster': cluster_codes}).groupby('cluster').agg('mean')['entropy'] -
    #     pd.DataFrame({'entropy': int_ent, 'cluster': cluster_codes}).groupby('cluster').agg('mean')['entropy']
    # )
    # stat = float(align_stat(stat))
    
    diff_ent = orig_ent - int_ent
    align_stat = lambda x: 5.0 if x <= 0 else np.clip(-1 * np.log2(x + eps), 0.0, 5.0)
    return get_results(diff_ent, adata_query, group_keys, sep_key='cluster', reduce='median', after_function=align_stat)


def calculate_scib_metrics(adata, obsm_emb_key, batch_key, dataset_key, sample_key, dry_run=False):
    K = {'lisi': 90, 'kbet': 50}
    with log_time("scib neighbors calculation", log=not dry_run):
        ng_indices, ng_distances = calculate_neighbors(adata, obsm_emb_key, k=max(list({'lisi': 90, 'kbet': 50}.values()))) if not dry_run else (None, None)
    group_keys = {'per_dataset': dataset_key, 'per_sample': sample_key}
    results = {}
    with log_time("asw_batch", log=not dry_run):
        results['asw_batch'] = calculate_asw_batch(adata, obsm_emb_key, batch_key,
                                                   group_keys=group_keys) if not dry_run else None
    with log_time("lisi_batch", log=not dry_run):
        results['lisi_batch'] = calculate_lisi_batch(adata, ng_indices, ng_distances, batch_key,
                                                     group_keys=group_keys, k=K['lisi']) if not dry_run else None
    with log_time("kbet_batch", log=not dry_run):
        results['kbet_batch'] = calculate_kbet_batch(adata, ng_indices, ng_distances, batch_key,
                                                     group_keys=group_keys, k=K['kbet']) if not dry_run else None
    return results


def calculate_symphony_metrics(adata, adata_query, obsm_emb_key, donor_key, dataset_key, sample_key, dry_run=False):
    group_keys = {'per_dataset': dataset_key, 'per_sample': sample_key}
    results = {}
    # TODO: @lisa: what is k for this? (currently k=90 as in ilisi (lisi_batch))
    K_lisi = 90
    with log_time("symphony metrics neighbors calculation", log=not dry_run):
        q_ng_indices, q_ng_distances = calculate_neighbors(adata_query, obsm_emb_key, k=K_lisi) if not dry_run else (None, None)
    with log_time("lisi_query_donor", log=not dry_run):
        results['lisi_query_donor'] = calculate_lisi_batch(adata_query, q_ng_indices, q_ng_distances, donor_key,
                                                           group_keys=group_keys, k=K_lisi) if not dry_run else None
    with log_time("wiq_knn_corr", log=not dry_run):
        results['wiq_knn_corr'] = calculate_wiq_knn_correlations(adata_query, obsm_emb_key,
                                                                 group_keys=group_keys) if not dry_run else None
    return results


def calculate_other_metrics(adata, adata_query, obsm_emb_key, dataset_key, sample_key, dry_run=False):
    group_keys = {'per_dataset': dataset_key, 'per_sample': sample_key}
    results = {}
    with log_time("cluster_preservation_score", log=not dry_run):
        results['cluster_preservation_score'] = calculate_cluster_preservation_score(adata_query, obsm_emb_key,
                                                                                     group_keys=group_keys) if not dry_run else None
    return results
