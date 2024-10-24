import scanpy as sc
from scipy import sparse
import pandas as pd
import scarches as sca
import numpy as np
import os

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
    # a separate variable to retain it:
    adata.obs["dataset_original"] = adata.obs["dataset"]
    adata.obs["dataset"] = adata.obs[batch_key]

    # Set cell type label variable, as used in this specific reference model,
    # to the unlabelled category used in this specific model:
    adata.obs["scanvi_label"] = "unlabeled"

    return adata


def subset_query_data_for_mapping(
    adata: sc.AnnData,
    path_to_ref_model_dir: str,
    check_integer_values_on_full_data: bool = False,
):
    # Set X to counts layer:
    adata.X = adata.layers["counts"]

    # Convert X to sparse CSR matrix if it's not already
    if not sparse.issparse(adata.X):
        adata.X = sparse.csr_matrix(adata.X)

    # Check that values in adata.X are integers:
    if not check_integer_values_on_full_data and adata.X.shape[0] > 50:
        if not np.all(adata.X[:50].toarray() == np.floor(adata.X[:50].toarray())):
            raise ValueError(
                "adata.X contains non-integer values for the first 50 cells"
            )
        if not np.all(adata.X[-50:].toarray() == np.floor(adata.X[-50:].toarray())):
            raise ValueError(
                "adata.X contains non-integer values for the last 50 cells"
            )
    else:
        if not np.all(adata.X.toarray() == np.floor(adata.X.toarray())):
            raise ValueError("adata.X contains non-integer values")
    # Prepare query data for scANVI. This includes data subsetting.
    adata = sca.models.SCANVI.prepare_query_anndata(
        adata=adata, reference_model=path_to_ref_model_dir, inplace=False
    )
    return adata


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
            [
                col
                for col in adata_reference.obs.columns
                if col.startswith("ann_level_")
            ],
        ],
    )
    adata_ref_latent.obs["r_or_q"] = "r"

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
    adata_latent_combined = adata_ref_latent.concatenate(
        adata_query_latent, index_unique=None
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
    adata_combined = adata_reference.concatenate(
        adata_query, join="outer", index_unique=None
    )
    # prevent "TypeError: Can't implicitly convert non-string objects to strings" error
    for col in adata_combined.obs.columns:
        if (
            not pd.api.types.is_float_dtype(adata_combined.obs[col])
            and not pd.api.types.is_integer_dtype(adata_combined.obs[col])
            and not pd.api.types.is_categorical_dtype(adata_combined.obs[col])
        ):
            adata_combined.obs[col] = adata_combined.obs[col].astype(str)
    return adata_combined
