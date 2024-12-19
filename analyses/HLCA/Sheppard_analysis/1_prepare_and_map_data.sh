#! /bin/bash
study="Sheppard_2020"
mapping_max_kl_weight=2.0
batch_covariate="dataset"
testrun=False
verbose=True
use_original_instead_of_cellxgene_counts=False

# parameter order: study, mapping_max_kl_weight, batch_covariate, testrun, verbose, use_original_instead_of_cellxgene_counts
python ../../prepare_and_map_data.py \
$study \
$mapping_max_kl_weight \
$batch_covariate \
$testrun \
$verbose \
$use_original_instead_of_cellxgene_counts > 1_prepare_and_map_data_cellxgene_counts_kl2.out 2>&1
