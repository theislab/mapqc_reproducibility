import scanpy as sc
import numpy as np
import pandas as pd


def prepare_reference_from_cellxgene_h5ad(path_to_adata: str, testrun: bool = False):
    # Load the data in backed mode:
    adata_full = sc.read_h5ad(path_to_adata, backed="r")
    # Subset to the core, which is the reference to which query cells can be mapped
    if testrun:
        core_mask = adata_full.obs["core_or_extension"] == "core"
        core_indices = np.where(core_mask)[0][:50000]
        adata_reference = adata_full[core_indices, :].to_memory()
    else:
        adata_reference = adata_full[
            adata_full.obs["core_or_extension"] == "core",
        ].to_memory()
    # Remove soupX layer:
    del adata_reference.layers["soupX"]
    # store raw counts in layer counts:
    adata_reference.layers["counts"] = adata_reference.raw.X
    # delete raw to save memory:
    del adata_reference.raw
    # Filter out genes not expressed in the reference
    sc.pp.filter_genes(adata_reference, min_cells=1)
    # Remove all adata.obs columns starting with 'transf' (these are for transfered
    # labels, which only apply to query cells):
    obs_columns_to_remove = [
        col for col in adata_reference.obs.columns if col.startswith("transf")
    ]
    adata_reference.obs = adata_reference.obs.drop(columns=obs_columns_to_remove)
    # remove useless var columns
    var_columns_to_remove = ["feature_is_filtered", "feature_biotype", "feature_length"]
    adata_reference.var = adata_reference.var.drop(columns=var_columns_to_remove)
    # Specify that this is the reference
    adata_reference.obs["r_or_q"] = "r"
    # Specify that all cells are from controls
    adata_reference.obs["status"] = "control"
    # for cells that have no annotation at a level (because the maximum granularity
    # has been reached in a lower level already), forward pass the annotation from
    # the previous level:
    for lev in range(2, 6):
        adata_reference.obs[f"ann_level_{lev}"] = adata_reference.obs[
            f"ann_level_{lev}"
        ].tolist()
        cells_with_no_ann = adata_reference.obs.loc[
            adata_reference.obs[f"ann_level_{lev}"] == "None", :
        ].index
        adata_reference.obs.loc[
            cells_with_no_ann, f"ann_level_{lev}"
        ] = adata_reference.obs.loc[cells_with_no_ann, f"ann_level_{lev -1}"]
        adata_reference.obs[f"ann_level_{lev}"] = pd.Categorical(
            adata_reference.obs[f"ann_level_{lev}"]
        )
    return adata_reference


def prepare_query_from_cellxgene_h5ad(
    path_to_adata: str, study_name: str, testrun: bool = False
):
    # Load the data in backed mode:
    adata_full = sc.read_h5ad(path_to_adata, backed="r")
    # Subset to the query study of interest
    if testrun:
        study_mask = adata_full.obs["study"] == study_name
        study_indices = np.where(study_mask)[0][:1000]
        adata_query = adata_full[study_indices, :].to_memory()
    else:
        adata_query = adata_full[adata_full.obs["study"] == study_name, :].to_memory()
    # Remove soupX layer:
    del adata_query.layers["soupX"]
    # store raw counts in layer counts:
    adata_query.layers["counts"] = adata_query.raw.X
    # delete raw to save memory:
    del adata_query.raw
    # Filter out genes not expressed in the query dataset
    sc.pp.filter_genes(adata_query, min_cells=1)
    # Specify that this is the query
    adata_query.obs["r_or_q"] = "q"
    # Remove all adata.obs columns starting with 'transf' (these are for transferred
    # labels, which only apply to query cells):
    obs_columns_to_remove = [
        col for col in adata_query.obs.columns if col.startswith("transf")
    ]
    adata_query.obs = adata_query.obs.drop(columns=obs_columns_to_remove)
    # Remove annotation columns that start with 'ann', as these are also
    # label-transfer based in the case of HLCA mapped data:
    obs_columns_to_remove = [
        col for col in adata_query.obs.columns if col.startswith("ann")
    ]
    adata_query.obs = adata_query.obs.drop(columns=obs_columns_to_remove)
    # remove useless var columns
    var_columns_to_remove = ["feature_is_filtered", "feature_biotype", "feature_length"]
    adata_query.var = adata_query.var.drop(columns=var_columns_to_remove)
    return adata_query

def replace_cellxgene_counts_with_Sheppard_2020_from_raw(
        adata_prepped_from_cellxgene: sc.AnnData,
        path_to_raw_subsetted_counts: str
):
    # Note that the cellxgene upload caused a number of genes to disappear,
    # most likely due to filtering in the upload process.
    # However, it seems like the mapping is of worse quality when using the 
    # filtered counts. Therefore, with this function we offer the option
    # to replace the counts in the cellxgene upload with the raw counts
    # as used in the HLCA paper.
    # Load the raw counts:
    adata_raw = sc.read_h5ad(path_to_raw_subsetted_counts)
    # reformat
    adata_raw.var['gene_symbol'] = adata_raw.var.index
    adata_raw.var.index = adata_raw.var.ensembl
    adata_raw.obs.index = [f"{idx}_tsukui" for idx in adata_raw.obs.index]
    # reorder adata_raw, such that cells and genes are in the same order 
    # as in adata_prepped_from_cellxgene
    adata_raw = adata_raw[adata_prepped_from_cellxgene.obs.index,adata_prepped_from_cellxgene.var.index].copy()
    # replace counts
    adata_prepped_from_cellxgene.layers['counts'] = adata_raw.X
    adata_prepped_from_cellxgene.X = adata_raw.X
    # note that these counts are the same for most genes, only not for 87 genes
    # (88 are missing in the cellxgene object, most likely 87 of those are 
    # the ones that are different in raw versus filtered, number 88 most likely 
    # had 0 counts and was therefore padded with 0s, corresponding to the original
    # counts)
    # remove adata_raw
    del adata_raw
    return adata_prepped_from_cellxgene

    


def clean_Sheppard_2020_data(adata: sc.AnnData):
    # Remove cells from SCD samples:
    adata = adata[
        adata.obs.lung_condition != "Systemic sclerosis-associated ILD", :
    ].copy()
    # Specify which  cells are from cases
    adata.obs.loc[adata.obs.lung_condition == "Healthy", "status"] = "control"
    adata.obs.loc[adata.obs.lung_condition == "IPF", "status"] = "IPF"
    # filter genes
    sc.pp.filter_genes(adata, min_cells=1)
    return adata


def clean_Thiepont_2018_data(adata: sc.AnnData):
    # TO DO
    return adata
