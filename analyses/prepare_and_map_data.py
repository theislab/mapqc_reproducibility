import sys
import os

# Get the directory of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, "utils"))
import data_cleanup
import mapping

# check if true or false was correctly inputted:
for i in [4, 5, 6]:
    if sys.argv[i].lower() not in ["true", "false"]:
        raise ValueError(
            f"Argument {i} is not a boolean. Please use 'True' or 'False'."
        )

par = {
    "study": sys.argv[1],  # e.g. "Sheppard_2020"
    "mapping_max_kl_weight": float(sys.argv[2]),  # e.g. "1.0"
    "batch_covariate": sys.argv[3],  # e.g. "sample"
    "testrun": sys.argv[4].lower() == "true",  # e.g.  False
    "verbose": sys.argv[5].lower() == "true",  # e.g. True
    "use_original_instead_of_cellxgene_counts": sys.argv[6].lower()
    == "true",  # e.g. True
}
kwargs_query_reading_function = dict()

if par["verbose"]:
    print(f"Parameters: {par}")
path_to_HLCA = "/ictstr01/groups/ml01/projects/2024_hlca_v2_raphael.kfurirubens/hlca_v1/hlca_v1_full.h5ad"
path_to_ref_model_dir = "/lustre/groups/ml01/code/lisa.sikkema/atlas_extension/analyses/HLCA/HLCA_reference_model"
path_to_model_features_csv = os.path.join(path_to_ref_model_dir, "var_names.csv")

# study-specific paths
if par["study"] == "Thienpont_2018":
    kwargs_query_reading_function[
        "path_to_sample_info_Thienpont_2018"
    ] = "./lambrechts_health_status_info.csv"

if par["use_original_instead_of_cellxgene_counts"]:
    file_prefix = f"original_counts_"
    # This is specific for Sheppard_2020 data when using raw counts:
    old_study_names = {"Sheppard_2020": "tsukui", "Thienpont_2018": "lambrechts"}
    old_name = old_study_names[par["study"]]
    path_to_raw_subsetted_counts = f"/lustre/groups/ml01/workspace/hlca_lisa.sikkema_malte.luecken/HLCA_reproducibility/data/HLCA_extended/extension_datasets/ready/subsetted/{old_name}_sub.h5ad"
else:
    file_prefix = f"cellxgene_counts_"
path_to_surgery_model_dir = f"/lustre/groups/ml01/code/lisa.sikkema/atlas_extension/results/HLCA/{par['study']}/{file_prefix}batch_{par['batch_covariate']}_kl{par['mapping_max_kl_weight']}/surgery_model_and_output_emb"

# check if parent directory exists:
grandparent_directory = os.path.dirname(os.path.dirname(path_to_surgery_model_dir))
if not os.path.isdir(grandparent_directory):
    raise ValueError(
        f"Grandparent directory of the specified surgery model directory ({grandparent_directory}) does not exist."
    )
else:
    if not os.path.isdir(path_to_surgery_model_dir):
        # create directory:
        print(f"Creating directory {path_to_surgery_model_dir}...")
        os.makedirs(path_to_surgery_model_dir)

if par["testrun"]:
    suffix = "_test"
else:
    suffix = ""
path_to_mapped_query_embedding = os.path.join(
    path_to_surgery_model_dir, f"query_emb{suffix}.h5ad"
)
path_to_joint_ref_q_counts_object = f"/lustre/groups/ml01/workspace/lisa.sikkema/atlas_extension/data/HLCA/HLCA_and_{par['study']}_{file_prefix}counts{suffix}.h5ad"
path_to_joint_ref_q_latent_object = f"/lustre/groups/ml01/workspace/lisa.sikkema/atlas_extension/data/HLCA/HLCA_and_{par['study']}_{file_prefix}batch_{par['batch_covariate']}_kl{par['mapping_max_kl_weight']}_latent{suffix}.h5ad"

if par["verbose"]:
    print("Preparing query data...")
adata_query = data_cleanup.prepare_query_from_cellxgene_h5ad(
    path_to_adata=path_to_HLCA,
    study_name=par["study"],
    testrun=par["testrun"],
    **kwargs_query_reading_function,
)

if par["verbose"]:
    print("Preparing reference data...")
adata_reference = data_cleanup.prepare_reference_from_cellxgene_h5ad(
    path_to_HLCA, testrun=par["testrun"]
)

if par["verbose"]:
    print("Preparing query data for mapping...")
adata_query = mapping.prepare_query_data_for_mapping(
    adata_query, batch_key=par["batch_covariate"]
)

if par["verbose"]:
    print("Subsetting query data for mapping...")
adata_query_for_mapping = mapping.subset_query_data_for_mapping(
    adata_query, path_to_ref_model_dir=path_to_ref_model_dir
)

if par["use_original_instead_of_cellxgene_counts"]:
    if par["verbose"]:
        print("Replacing cellxgene counts with original counts...")
        n_genes_with_counts_before = (adata_query_for_mapping.X.sum(axis=0) > 0).sum()
    adata_query_for_mapping = data_cleanup.replace_cellxgene_counts_with_original(
        adata_query_for_mapping,
        study_name=par["study"],
        path_to_raw_subsetted_counts=path_to_raw_subsetted_counts,
    )
    if par["verbose"]:
        n_genes_with_counts_after = (adata_query_for_mapping.X.sum(axis=0) > 0).sum()
        n_genes_diff = n_genes_with_counts_after - n_genes_with_counts_before
        print(
            f"Number of genes with counts added (or removed if negative): {n_genes_diff}"
        )
        print(
            f"Number of genes with counts after replacement: {n_genes_with_counts_after}"
        )
        print(
            f"Number of genes with counts before replacement: {n_genes_with_counts_before}"
        )


if par["verbose"]:
    print("Mapping data to HLCA reference...")
    mapping.map_data_to_HLCA_reference(
        adata_query_for_mapping,
        path_ref_model_dir=path_to_ref_model_dir,
        path_surgery_model_dir=path_to_surgery_model_dir,
        path_output_query_emb=path_to_mapped_query_embedding,
        max_kl_weight=par["mapping_max_kl_weight"],
        testrun=par["testrun"],
    )

if par["verbose"]:
    print("Creating mapping output embedding adata...")
# TO DO: add preparation parameters to adata_latent_combined.uns
adata_latent_combined = mapping.create_and_save_mapping_output_adata(
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
    adata_combined = mapping.create_joint_ref_q_counts_object(
        adata_reference, adata_query, verbose=False  # otherwise double print statements
    )
    adata_combined.write(path_to_joint_ref_q_counts_object)
else:
    if par["verbose"]:
        print(
            "Not creating joint counts object, as it has already been created in a previous run."
        )
