import sys
import os

sys.path.append("../../utils/")

import data_cleanup_utils
import mapping_utils

par = {
    "study": "Sheppard_2020",
    "mapping_max_kl_weight": 1.0,
    "batch_covariate": "sample",
    "testrun": False,
    "verbose": True,
    "use_raw_instead_of_cellxgene_counts": True
}

if par["verbose"]:
    print(f"Parameters: {par}")
path_to_HLCA = "/lustre/groups/ml01/workspace/lisa.sikkema/atlas_extension/data/HLCA/HLCA_extended_from_cellxgene.h5ad"
path_to_model_features_csv = "../HLCA_reference_model/var_names.csv"
path_to_ref_model_dir = "../HLCA_reference_model"

if par['study'] != "Sheppard_2020":
    file_prefix = f"{par['study']}_"
else:
    if par['use_raw_instead_of_cellxgene_counts']:
        file_prefix = f"{par['study']}_old_counts_"
        # This is specific for Sheppard_2020 data when using raw counts:
        path_to_raw_subsetted_counts = "/lustre/groups/ml01/workspace/hlca_lisa.sikkema_malte.luecken/HLCA_reproducibility/data/HLCA_extended/extension_datasets/ready/subsetted/tsukui_sub.h5ad"
    else:
        file_prefix = f"{par['study']}_"
path_to_surgery_model_dir = f"../surgery_models_and_output_embs/{file_prefix}batch_{par['batch_covariate']}_kl{par['mapping_max_kl_weight']}"

if par["testrun"]:
    path_to_mapped_query_embedding = os.path.join(
        path_to_surgery_model_dir, "query_emb_test.h5ad"
    )
    path_to_joint_ref_q_counts_object = f"/lustre/groups/ml01/workspace/lisa.sikkema/atlas_extension/data/HLCA/HLCA_and_{file_prefix}counts_test.h5ad"
    path_to_joint_ref_q_latent_object = f"/lustre/groups/ml01/workspace/lisa.sikkema/atlas_extension/data/HLCA/HLCA_and_{file_prefix}batch_{par['batch_covariate']}_kl{par['mapping_max_kl_weight']}_latent_test.h5ad"
else:
    path_to_mapped_query_embedding = os.path.join(
        path_to_surgery_model_dir, "query_emb.h5ad"
    )
    path_to_joint_ref_q_counts_object = f"/lustre/groups/ml01/workspace/lisa.sikkema/atlas_extension/data/HLCA/HLCA_and_{file_prefix}counts.h5ad"
    path_to_joint_ref_q_latent_object = f"/lustre/groups/ml01/workspace/lisa.sikkema/atlas_extension/data/HLCA/HLCA_and_{file_prefix}batch_{par['batch_covariate']}_kl{par['mapping_max_kl_weight']}_latent.h5ad"

if par["verbose"]:
    print("Preparing query data...")
adata_query = data_cleanup_utils.prepare_query_from_cellxgene_h5ad(
    path_to_adata=path_to_HLCA, study_name=par["study"], testrun=par["testrun"]
)

if par["verbose"]:
    print("Cleaning query data further...")
adata_query = data_cleanup_utils.clean_Sheppard_2020_data(adata_query)

if par["verbose"]:
    print("Preparing reference data...")
adata_reference = data_cleanup_utils.prepare_reference_from_cellxgene_h5ad(
    path_to_HLCA, testrun=par["testrun"]
)

if par["verbose"]:
    print("Preparing query data for mapping...")
adata_query = mapping_utils.prepare_query_data_for_mapping(
    adata_query, batch_key=par["batch_covariate"]
)

if par["verbose"]:
    print("Subsetting query data for mapping...")
adata_query_for_mapping = mapping_utils.subset_query_data_for_mapping(
    adata_query, path_to_ref_model_dir=path_to_ref_model_dir
)

if par["use_raw_instead_of_cellxgene_counts"]:
    if par["verbose"]:
        print("Replacing cellxgene counts with raw counts...")
        n_genes_with_counts_before = (adata_query_for_mapping.X.sum(axis=0)>0).sum()
    adata_query_for_mapping = data_cleanup_utils.replace_cellxgene_counts_with_Sheppard_2020_from_raw(
        adata_query_for_mapping,
        path_to_raw_subsetted_counts=path_to_raw_subsetted_counts
    )
    if par["verbose"]:
        n_genes_with_counts_after = (adata_query_for_mapping.X.sum(axis=0)>0).sum()
        n_genes_diff = n_genes_with_counts_after - n_genes_with_counts_before
        print(f"Number of genes with counts added (or removed if negative): {n_genes_diff}")
        print(f"Number of genes with counts after replacement: {n_genes_with_counts_after}")
        print(f"Number of genes with counts before replacement: {n_genes_with_counts_before}")
    

if par['verbose']:
    print("Mapping data to HLCA reference...")
    mapping_utils.map_data_to_HLCA_reference(
        adata_query_for_mapping,
        path_ref_model_dir=path_to_ref_model_dir,
        path_surgery_model_dir=path_to_surgery_model_dir,
        max_kl_weight=par['mapping_max_kl_weight']
    )

if par["verbose"]:
    print("Creating mapping output embedding adata...")
# TO DO: add preparation parameters to adata_latent_combined.uns
adata_latent_combined = mapping_utils.create_and_save_mapping_output_adata(
    adata_reference,
    adata_query,
    path_to_mapped_query_emb=path_to_mapped_query_embedding,
    output_path_joint_emb=path_to_joint_ref_q_latent_object,
    k=50,
)

if not os.path.isfile(path_to_joint_ref_q_counts_object):
    if par["verbose"]:
        print(
            f"Creating joint counts object and writing to {path_to_joint_ref_q_counts_object}..."
        )
    adata_combined = mapping_utils.create_joint_ref_q_counts_object(
        adata_reference, adata_query
    )
    adata_combined.write(path_to_joint_ref_q_counts_object)
else:
    if par["verbose"]:
        print("Not creating joint counts object, as it has already been created in a previous run.")
