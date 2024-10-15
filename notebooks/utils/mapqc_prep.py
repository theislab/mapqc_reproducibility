import scanpy as sc
import numpy as np

def add_mapping_info_to_counts_adata(adata_counts_combined, adata_emb_combined):
    # order both in the same order, to be sure:
    adata_emb_combined = adata_emb_combined[adata_counts_combined.obs.index, :]
    # copy .obs columns
    for col in adata_emb_combined.obs.columns:
        adata_counts_combined.obs[col] = adata_emb_combined.obs[col]
    # copy umap and embedding, and knn graph (in obsp)
    adata_counts_combined.obsm["X_umap"] = adata_emb_combined.obsm["X_umap"]
    adata_counts_combined.obsm["X_scanvi"] = np.float32(adata_emb_combined.X)
    adata_counts_combined.obsp = adata_emb_combined.obsp
    return adata_counts_combined
