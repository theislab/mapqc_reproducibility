import scanpy as sc
import numpy as np


def import_and_prepare_reference(
    path_to_adata: str, testrun: bool = False, verbose: bool = True
):
    adata_full = sc.read_h5ad(path_to_adata)
    if verbose:
        print("Removing disease data from reference.")
    adata = adata_full[adata_full.obs.Endometrial_pathology == "C", :].copy()
    if testrun:
        if verbose:
            print("Subsetting to 50k cells for testrun...")
        adata = adata[:50000, :].copy()
    # remove dataset-specific info from adata.var
    var_cols_to_remove = list()
    for col in adata.var.columns:
        if "-" in col:
            var_cols_to_remove.append(col)
    adata.var.drop(columns=var_cols_to_remove, inplace=True)
    # remove uns and obsp + obsm
    del adata.uns
    del adata.obsp
    del adata.obsm
    # add query versus reference info
    adata.obs["r_or_q"] = "r"
    # set X to raw counts instead of normalized
    adata.X = adata.layers["counts"]
    # remove genes not expressed in the reference
    sc.pp.filter_genes(adata, min_cells=1)
    # now normalize to 10k counts per cell, then log-transform
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    # remove existing umap if it exists:
    if "X_umap" in adata.obsm.keys():
        del adata.obsm["X_umap"]
    # and return adata
    return adata


def import_and_prepare_query(
    path_to_adata: str, testrun: bool = False, verbose: bool = True
):
    adata = sc.read_h5ad(path_to_adata)
    if testrun:
        if verbose:
            print("Subsetting to 10k cells for testrun...")
        adata = adata[:10000, :].copy()
    # clean up obs columns:
    obs_col_renaming = {
        "nCount_RNA": "n_counts",
        "mitochondria_ratio": "percent_mito",
        "celltype": "celltype_query",
        "principal_cell_types": "principal_celltype_query",
    }
    adata.obs.rename(columns=obs_col_renaming, inplace=True)
    # remove unexpressed genes:
    sc.pp.filter_genes(adata, min_cells=1)
    # store raw counts in layer counts:
    adata.layers["counts"] = adata.X.copy()
    # normalize to 10k counts per cell, then log-transform
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    # remove existing umap if it exists:
    if "X_umap" in adata.obsm.keys():
        del adata.obsm["X_umap"]
    # add ref/q info:
    adata.obs["r_or_q"] = "q"
    # and return adata
    return adata


def add_mapping_info_to_counts_adata(
    adata_counts_combined, adata_emb_combined, celltype_col: str = None
):
    # order both in the same order, to be sure:
    adata_emb_combined = adata_emb_combined[adata_counts_combined.obs.index, :]
    # copy .obs columns
    for col in adata_emb_combined.obs.columns:
        adata_counts_combined.obs[col] = adata_emb_combined.obs[col]
    # copy umap and embedding, and knn graph (in obsp)
    adata_counts_combined.obsm["X_umap"] = adata_emb_combined.obsm["X_umap"]
    adata_counts_combined.obsm["X_scanvi_emb"] = np.float32(adata_emb_combined.X)
    adata_counts_combined.obs["batch_variable_for_mapping"] = adata_emb_combined.obs[
        "batch_variable_for_mapping"
    ]
    if "leiden" in adata_emb_combined.obs.columns:
        adata_counts_combined.obs["leiden"] = adata_emb_combined.obs["leiden"]
    adata_counts_combined.obsp = adata_emb_combined.obsp
    if celltype_col is not None:
        adata_counts_combined.obs[f"{celltype_col}_merged"] = adata_counts_combined.obs[
            celltype_col
        ].tolist()
        adata_counts_combined.obs.loc[
            adata_counts_combined.obs["r_or_q"] == "q", f"{celltype_col}_merged"
        ] = adata_emb_combined.obs.loc[
            adata_counts_combined.obs["r_or_q"] == "q", f"{celltype_col}_pred"
        ].copy()
    return adata_counts_combined
