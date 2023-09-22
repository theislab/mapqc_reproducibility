from scipy.spatial.distance import cdist
from sklearn.metrics import pairwise_distances
import pandas as pd
import numpy as np


class Neighborhood:  
    def __init__(
        self,
        adata,
        index_cell,
        k,
        emb_key="X",
        sample_key="sample",
        ref_q_key="reference_or_query",
        ref_cat="reference",
        query_cat="query",
        min_n_cells_per_sample=5,
        min_n_samples_ref_or_q=3,
    ):
        # store embedding as separate attribute
        if emb_key == "X":
            self.adata_emb = adata.X
        elif emb_key in adata.obsm.keys():
            self.adata_emb = adata.obsm["emb_key"]
        # store parameters as attributes (add sanity checks here too?)
        self.min_n_cells_per_sample = min_n_cells_per_sample
        self.min_n_samples_ref_or_q = min_n_samples_ref_or_q
        self.k = k
        # check if index cell in adata
        if index_cell not in adata.obs.index:
            raise ValueError("Your index cell is not found in adata.obs.index.")
        # get row number in adata that has the index cell
        self.index_cell_idx = np.where(adata.obs.index == index_cell)[0][0]
        # get nearest neighbors of index cell, i.e. the cells in this nhood:
        self.cell_idc = self._get_nearest_neighbors()
        self.barcodes = adata.obs.iloc[self.cell_idc, :].index
        # get sample names of cells in neighborhoods
        if sample_key not in adata.obs.columns:
            raise ValueError("Your sample key is not present in adata.obs.columns.")
        self.samples = adata.obs.loc[self.barcodes, sample_key].values
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
        self.ref_or_q = (
            adata.obs.loc[self.barcodes, ref_q_key].map(ref_q_harmonizer).values
        )
        # store embedding of cells in nhood:
        self.emb = self.adata_emb[self.cell_idc, :]
        # calculate pairwise distances between cells in nhood:
        self.dists = pairwise_distances(self.emb)

    def _get_nearest_neighbors(self):
        """get k nearest neighbors of a given index cell"""
        dists_to_center_cell = cdist(
            self.adata_emb[self.index_cell_idx, :].reshape(
                (1, self.adata_emb.shape[1])
            ),
            self.adata_emb,
        )[0]
        # get indices of columns (cell indices) sorted from small to large
        sorted_indices = np.argsort(dists_to_center_cell)  # , axis=1)
        # get column numbers for the columns with the k lowest values, i.e.
        # the k nearest neighbors (including the index cell itself)
        return sorted_indices[:self.k]

    def _filter_samples(self, ref_or_q):
        """Filter samples based on the number of cells they have in the current
        neighborhood. Do this for either the reference or the query samples."""
        if ref_or_q not in ["reference", "query"]:
            raise ValueError("ref_or_q should be set to either 'reference' or 'query'.")
        sample_cell_counts = self.samples[self.ref_or_q == ref_or_q].value_counts()
        sample_set_filtered = sample_cell_counts.index[
            sample_cell_counts >= self.min_n_cells_per_sample
        ].tolist()
        return sample_set_filtered

    def _calculate_sample_pair_mean_distances(self, sample_1, sample_2):
        # check which cells in the neigbhorhood belong to each sample
        sample_1_cells = np.where(self.samples == sample_1)[0]
        sample_2_cells = np.where(self.samples == sample_2)[0]
        # subset nhood's distance matrix to cells from the two samples
        pair_specific_dists = self.dists[sample_1_cells][:, sample_2_cells]
        return np.mean(pair_specific_dists)

    def get_pairwise_sample_dists(self):
        # check which samples have sufficient cells in the neighborhood
        ref_samples = self._filter_samples("reference")
        q_samples = self._filter_samples("query")
        # check if enough samples from ref and query are present in nhood:
        if (
            len(ref_samples) < self.min_n_samples_ref_or_q
            or len(q_samples) < self.min_n_samples_ref_or_q
        ):
            return np.nan
        # create empty dataframe in which to store pairwise distances
        sample_dists = pd.DataFrame(index=ref_samples, columns=ref_samples + q_samples)
        # loop through all ref samples (and count in which loop we are)
        for ref_sample_count, sample_1 in enumerate(ref_samples):
            # loop through all reference samples that that we haven't looped
            # through in the first loop yet, plus all query samples:
            for sample_2 in (ref_samples + q_samples)[ref_sample_count + 1 :]:
                mean_intersample_dist = self._calculate_sample_pair_mean_distances(
                    sample_1, sample_2
                )
                sample_dists.loc[sample_1, sample_2] = mean_intersample_dist
                if sample_2 in ref_samples:
                    sample_dists.loc[sample_2, sample_1] = mean_intersample_dist
        return sample_dists
