import scanpy as sc
from scipy import sparse
import pandas as pd
import scarches as sca
import numpy as np
import os
import anndata as ad

import sys

sys.path.append("../../utils/")
import label_transfer


def prepare_query_data_for_mapping(
    adata: sc.AnnData,
    batch_key: str,
):
    # Remove obsm, varm, and obsp attributes if they exist
    if hasattr(adata, "obsm"):
        del adata.obsm
    if hasattr(adata, "varm"):
        del adata.varm
    if hasattr(adata, "obsp"):
        del adata.obsp

    # Ensure batch_key is in adata.obs
    if batch_key not in adata.obs.columns:
        raise ValueError(f"batch_key '{batch_key}' not found in adata.obs")

    # Set the batch variable. Note that for this specific reference model,
    # the batch key was defined as "dataset", which can get a bit confusing.
    # Even if our batch variable is not dataset but rather e.g. sample, we
    # have to set our dataset variable to sample. We will store dataset as
    # a separate variable to retain it, and we will re-set this variable
    # in the create_and_save_mapping_output_adata function:
    adata.obs["dataset_original"] = adata.obs["dataset"].copy()
    adata.obs["dataset"] = adata.obs[batch_key].copy()

    # Set cell type label variable, as used in this specific reference model,
    # to the unlabelled category used in this specific model:
    adata.obs["scanvi_label"] = "unlabeled"

    return adata


def subset_query_data_for_mapping(
    adata: sc.AnnData,
    path_to_ref_model_dir: str,
    check_integer_values_on_full_data: bool = False,
):
    # copy adata to avoid modifying the original object
    adata_subset = adata.copy()
    # Set X to counts layer:
    adata_subset.X = adata_subset.layers["counts"]

    # Convert X to sparse CSR matrix if it's not already
    if not sparse.issparse(adata_subset.X):
        adata_subset.X = sparse.csr_matrix(adata_subset.X)

    # Check that values in adata.X are integers:
    if not check_integer_values_on_full_data and adata_subset.X.shape[0] > 50:
        if not np.all(
            adata_subset.X[:50].toarray() == np.floor(adata_subset.X[:50].toarray())
        ):
            raise ValueError(
                "adata.X contains non-integer values for the first 50 cells"
            )
        if not np.all(
            adata_subset.X[-50:].toarray() == np.floor(adata_subset.X[-50:].toarray())
        ):
            raise ValueError(
                "adata.X contains non-integer values for the last 50 cells"
            )
    else:
        if not np.all(adata_subset.X.toarray() == np.floor(adata_subset.X.toarray())):
            raise ValueError("adata.X contains non-integer values")
    # Prepare query data for scANVI. This includes data subsetting.
    adata_subset = sca.models.SCANVI.prepare_query_anndata(
        adata=adata_subset, reference_model=path_to_ref_model_dir, inplace=False
    )
    return adata_subset


def map_data_to_HLCA_reference(
    prepped_query_adata: sc.AnnData,
    path_ref_model_dir: str,
    path_surgery_model_dir: str,
    path_output_query_emb: str,
    max_kl_weight: float,
    testrun: bool = False,
):
    """
    Maps query data to reference, and stores resulting surgery model
    and output query embedding in path_surgery_model_dir directory.
    """
    # Check that directories exist:
    if not os.path.exists(path_ref_model_dir):
        raise ValueError(f"Directory {path_ref_model_dir} does not exist")
    if not os.path.exists(path_surgery_model_dir):
        print(f"Creating directory {path_surgery_model_dir}")
        os.makedirs(path_surgery_model_dir)
    # Load the query data into the surgery model
    surgery_model = sca.models.SCANVI.load_query_data(
        prepped_query_adata,
        path_ref_model_dir,
        freeze_dropout=True,
    )
    # Set parameters for training. The only parameters we have
    # changed in this project were max_kl_weight
    if testrun:
        print("Running testrun with only 10 epochs...")
        surgery_epochs = 10
    else:
        surgery_epochs = 500
    early_stopping_kwargs_surgery = {
        "early_stopping_monitor": "elbo_train",
        "early_stopping_patience": 10,
        "early_stopping_min_delta": 0.001,
        "plan_kwargs": {"weight_decay": 0.0, "max_kl_weight": max_kl_weight},
    }
    # Train the model
    surgery_model.train(max_epochs=surgery_epochs, **early_stopping_kwargs_surgery)
    # Save the model
    surgery_model.save(path_surgery_model_dir, overwrite=True)
    # and save the embedding
    print(f"Saving mapped query embedding to {path_output_query_emb}...")
    sc.AnnData(
        surgery_model.get_latent_representation(prepped_query_adata),
        obs=prepped_query_adata.obs.iloc[:, :0],
    ).write(path_output_query_emb)


def create_and_save_mapping_output_adata(
    adata_reference: sc.AnnData,
    adata_query: sc.AnnData,
    path_to_mapped_query_emb: str,
    output_path_joint_emb: str,
    add_umap: bool = True,
    add_leiden: bool = True,
    perform_label_transfer: bool = True,
    k: int = 30,
    verbose: bool = True,
):
    # obtain latent representation of query data by passing it through the surgery model:
    adata_query_latent = sc.read_h5ad(path_to_mapped_query_emb)
    adata_query.obsm[
        "X_scanvi_emb"
    ] = adata_query_latent.X  # do we really need this line?
    adata_query_latent.obs["r_or_q"] = "q"
    # create anndata object of reference latent representation
    adata_ref_latent = sc.AnnData(
        adata_reference.obsm["X_scanvi_emb"],
        obs=adata_reference.obs.loc[
            :,
            [col for col in adata_reference.obs.columns if col.startswith("ann_level_")]
            + ["dataset"],
        ],
    )
    adata_ref_latent.obs["r_or_q"] = "r"
    # store batch variable for both ref and query. Note that this is based on the HLCA model,
    # which uses the "dataset" column as the batch variable
    adata_ref_latent.obs["batch_variable_for_mapping"] = adata_reference.obs["dataset"]
    adata_query_latent.obs["batch_variable_for_mapping"] = adata_query.obs["dataset"]
    # perform label transfer:
    if perform_label_transfer:
        if verbose:
            print(
                "Performing label transfer using the reference and query latent representations..."
            )

        adata_query_latent = label_transfer.perform_label_transfer(
            r_adata=adata_ref_latent,
            q_adata=adata_query_latent,
            reference_obs_targets=[f"ann_level_{lev}" for lev in [1, 2, 3, 4, 5]],
            reference_obsm_features="X",
            query_obsm_features="X",
            k=k,
        )
    # join the two latent representations:
    adata_latent_combined = ad.concat(
        (adata_ref_latent, adata_query_latent), index_unique=None, join="outer"
    )

    # calculate knn graph if UMAP and/or clustering is desired:
    if add_umap or add_leiden:
        if verbose:
            print("Calculating KNN graph...")
        sc.pp.neighbors(adata_latent_combined, n_neighbors=k)

    # add UMAP:
    if add_umap:
        if verbose:
            print("Calculating UMAP (this might take a while!)...")
        sc.tl.umap(adata_latent_combined)
    # add Leiden clustering:
    if add_leiden:
        if verbose:
            print("Calculating Leiden clustering...")
        sc.tl.leiden(adata_latent_combined, key_added="leiden")
    # save the output:
    if verbose:
        print(f"Saving joint embedding to {output_path_joint_emb}...")
    adata_latent_combined.write(output_path_joint_emb)
    return adata_latent_combined


def create_joint_ref_q_counts_object(
    adata_reference: sc.AnnData, adata_query: sc.AnnData, verbose: bool = True
):
    # concatenate reference and query data:
    if verbose:
        print("Concatenating reference and query data...")
    # remove re-set dataset column (this was added for the mapping, as the HLCA
    # reference model uses the dataset column as the batch variable. Therefore,
    # if we wanted to map per sample we needed to set sample as "dataset".
    # for downstream use this needs to be corrected back again)
    del adata_query.obs["dataset"]
    adata_query.obs["dataset"] = adata_query.obs["dataset_original"].copy()
    del adata_query.obs["dataset_original"]
    # delete existing umap and scanvi embedding if they exist (keep scanvi emb for reference)
    if "X_scanvi_emb" in adata_query.obsm.keys():
        del adata_query.obsm["X_scanvi_emb"]
    if "X_umap" in adata_query.obsm.keys():
        del adata_query.obsm["X_umap"]
    if "X_umap" in adata_reference.obsm.keys():
        del adata_reference.obsm["X_umap"]
    # now concatenate ref and query data:
    adata_combined = ad.concat(
        (adata_reference, adata_query), join="outer", index_unique=None, merge="first"
    )
    # remove irrelevant var columns:
    adata_combined.var.drop(columns=["feature_reference", "n_cells"], inplace=True)
    # prevent "TypeError: Can't implicitly convert non-string objects to strings" error
    for col in adata_combined.obs.columns:
        if (
            not pd.api.types.is_float_dtype(adata_combined.obs[col])
            and not pd.api.types.is_integer_dtype(adata_combined.obs[col])
            and not pd.api.types.is_categorical_dtype(adata_combined.obs[col])
        ):
            adata_combined.obs[col] = adata_combined.obs[col].astype(str)
    return adata_combined
