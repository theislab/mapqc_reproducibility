import sys
import os

sys.path.append("../../utils/")

from data_cleanup_utils import *
from mapping_utils import *

study = "Sheppard_2020"
mapping_max_kl_weight = 1.0
batch_covariate = "sample"
testrun = True

path_to_HLCA = "/lustre/groups/ml01/workspace/lisa.sikkema/atlas_extension/data/HLCA/HLCA_extended_from_cellxgene.h5ad"
path_to_model_features_csv = "../HLCA_reference_model/var_names.csv"
path_to_ref_model_dir = "../HLCA_reference_model"
path_to_surgery_model_dir = f"../surgery_models_and_output_embs/{study}_batch_{batch_covariate}_kl{mapping_max_kl_weight}"
path_to_mapped_query_embedding = os.path.join(
    path_to_surgery_model_dir, "query_emb_test.h5ad"
)
path_to_joint_ref_q_counts_object = f"/lustre/groups/ml01/workspace/lisa.sikkema/atlas_extension/data/HLCA/HLCA_and_{study}.h5ad"

adata_query = prepare_query_from_cellxgene_h5ad(
    path_to_adata=path_to_HLCA, study_name=study, testrun=testrun
)

adata_query = clean_Sheppard_2020_data(adata_query)

adata_reference = prepare_reference_from_cellxgene_h5ad(path_to_HLCA, testrun=testrun)

adata_query = prepare_query_data_for_mapping(adata_query, batch_key=batch_covariate)

adata_query_for_mapping = subset_query_data_for_mapping(
    adata_query, path_to_ref_model_dir=path_to_ref_model_dir
)

# map_data_to_HLCA_reference(
#     adata_query_for_mapping,
#     path_ref_model_dir=path_to_ref_model_dir,
#     path_surgery_model_dir=path_to_surgery_model_dir,
#     max_kl_weight=mapping_max_kl_weight
# )

adata_latent_combined = create_mapping_output_adata(
    adata_reference,
    adata_query,
    path_to_mapped_emb=path_to_mapped_query_embedding,
    k=50,
)

if not os.path.isfile(path_to_joint_ref_q_counts_object):
    print("Creating joint counts object and writing to file...")
    adata_combined = create_joint_ref_q_counts_object(adata_reference, adata_query)
    adata_combined.write(path_to_joint_ref_q_counts_object)
