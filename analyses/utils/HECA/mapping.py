import scanpy as sc
import scarches as sca
import numpy as np
import pandas as pd
import anndata as ad
import scipy.sparse as sparse
import sys
import os

sys.path.append("../utils/")
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
    # the batch key was defined as "genotype", which can get a bit confusing.
    adata.obs["genotype"] = adata.obs[batch_key].copy()

    return adata


def subset_query_data_for_mapping_and_get_ref_latent(
    adata_ref: sc.AnnData,
    adata_q: sc.AnnData,
    path_to_ref_model_dir: str,
    check_integer_values_on_full_data: bool = False,
):
    """
    returns adata_ref_latent, adata_q_subset
    """
    # Start with reference, so that we can load the reference model. Subset ref to hvgs:
    adata_ref_subset = adata_ref[:, adata_ref.var["highly_variable"]].copy()
    adata_ref_subset.X = adata_ref_subset.layers["counts"]
    adata_ref_subset = adata_ref_subset[:, adata_ref_subset.var.highly_variable].copy()
    ref_model = sca.models.SCANVI.load(path_to_ref_model_dir, adata_ref_subset)
    adata_ref_latent = sc.AnnData(
        ref_model.get_latent_representation(adata_ref_subset),
        obs=adata_ref_subset.obs.iloc[:, :0],
    )
    adata_ref_latent.obs["celltype"] = adata_ref.obs["celltype"]
    # copy adata to avoid modifying the original object
    adata_q_subset = adata_q.copy()
    # Set X to counts layer:
    adata_q_subset.X = adata_q_subset.layers["counts"]

    # Convert X to sparse CSR matrix if it's not already
    if not sparse.issparse(adata_q_subset.X):
        adata_q_subset.X = sparse.csr_matrix(adata_q_subset.X)

    # Set celltype covariate to unlabeled:
    adata_q_subset.obs["celltype"] = ref_model.unlabeled_category_
    # Check that values in adata.X are integers:
    if not check_integer_values_on_full_data and adata_q_subset.X.shape[0] > 50:
        if not np.all(
            adata_q_subset.X[:50].toarray() == np.floor(adata_q_subset.X[:50].toarray())
        ):
            raise ValueError(
                "adata_q.X contains non-integer values for the first 50 cells"
            )
        if not np.all(
            adata_q_subset.X[-50:].toarray()
            == np.floor(adata_q_subset.X[-50:].toarray())
        ):
            raise ValueError(
                "adata_q.X contains non-integer values for the last 50 cells"
            )
    else:
        if not np.all(
            adata_q_subset.X.toarray() == np.floor(adata_q_subset.X.toarray())
        ):
            raise ValueError("adata_q.X contains non-integer values")
    # Prepare query data for scANVI. This includes data subsetting.
    adata_q_subset = sca.models.SCANVI.prepare_query_anndata(
        adata=adata_q_subset, reference_model=path_to_ref_model_dir, inplace=False
    )
    return adata_ref_latent, adata_q_subset


def map_data_to_HECA_reference(
    prepped_query_adata: sc.AnnData,
    path_ref_model_dir: str,
    path_surgery_model_dir: str,
    path_output_query_emb: str,
    testrun: bool = False,
    verbose: bool = True,
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
        prepped_query_adata, path_ref_model_dir, freeze_dropout=True, use_gpu=True
    )
    # Set parameters for training. The only parameters we have
    # changed in this project were max_kl_weight
    if testrun:
        print("Running testrun with only 10 epochs...")
        surgery_epochs = 10
    else:
        surgery_epochs = 500
    # Train the model (according to HECA tutorial, except we use higher max_epochs.)
    surgery_model.train(
        max_epochs=surgery_epochs,
        plan_kwargs=dict(weight_decay=0.0),
        check_val_every_n_epoch=10,  # I think this is not used, as early stopping is not set to True.
        use_gpu=True,
    )
    # Save the model
    if verbose:
        print(f"Saving surgery model to {path_surgery_model_dir}...")
    surgery_model.save(path_surgery_model_dir, overwrite=True)
    # and save the embedding
    if verbose:
        print(f"Saving mapped query embedding to {path_output_query_emb}...")
    sc.AnnData(
        surgery_model.get_latent_representation(prepped_query_adata),
        obs=prepped_query_adata.obs.iloc[:, :0],
    ).write(path_output_query_emb)


def create_and_save_mapping_output_adata(
    adata_reference: sc.AnnData,
    adata_reference_latent: sc.AnnData,
    adata_query: sc.AnnData,
    path_to_mapped_query_emb: str,
    output_path_joint_emb: str,
    add_umap: bool = True,
    add_leiden: bool = True,
    perform_label_transfer: bool = True,
    k: int = 30,
    verbose: bool = True,
):
    # load stored adata_query_latent
    adata_query_latent = sc.read_h5ad(path_to_mapped_query_emb)
    adata_query.obsm[
        "X_scanvi_emb"
    ] = adata_query_latent.X  # do we really need this line?
    adata_query_latent.obs["r_or_q"] = "q"
    # create anndata object of reference latent representation
    adata_reference_latent.obs["r_or_q"] = "r"
    # store batch variable for both ref and query. Note that this is based on the HLCA model,
    # which uses the "dataset" column as the batch variable
    adata_reference_latent.obs["batch_variable_for_mapping"] = adata_reference.obs[
        "genotype"
    ]
    adata_query_latent.obs["batch_variable_for_mapping"] = adata_query.obs["genotype"]
    # perform label transfer:
    if perform_label_transfer:
        if verbose:
            print(
                "Performing label transfer using the reference and query latent representations..."
            )

        adata_query_latent = label_transfer.perform_label_transfer(
            r_adata=adata_reference_latent,
            q_adata=adata_query_latent,
            reference_obs_targets=["celltype"],
            reference_obsm_features="X",
            query_obsm_features="X",
            k=k,
        )
    # join the two latent representations:
    adata_latent_combined = ad.concat(
        (adata_reference_latent, adata_query_latent), index_unique=None, join="outer"
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
    adata_combined.var.drop(columns=["n_cells"], inplace=True)
    # prevent "TypeError: Can't implicitly convert non-string objects to strings" error
    for col in adata_combined.obs.columns:
        if (
            not pd.api.types.is_float_dtype(adata_combined.obs[col])
            and not pd.api.types.is_integer_dtype(adata_combined.obs[col])
            and not pd.api.types.is_categorical_dtype(adata_combined.obs[col])
        ):
            adata_combined.obs[col] = adata_combined.obs[col].astype(str)
    # and for var columns:
    for col in adata_combined.var.columns:
        if (
            not pd.api.types.is_float_dtype(adata_combined.var[col])
            and not pd.api.types.is_integer_dtype(adata_combined.var[col])
            and not pd.api.types.is_categorical_dtype(adata_combined.var[col])
        ):
            adata_combined.var[col] = adata_combined.var[col].astype(str)
    return adata_combined
