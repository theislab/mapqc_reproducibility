# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.15.0
#   kernelspec:
#     display_name: q2r_metric
#     language: python
#     name: q2r_metric
# ---

# %load_ext autoreload
# %autoreload 2



# +
import os
import pickle
from pathlib import Path

import anndata as ad
import numpy as np
import scipy
import scanpy as sc
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scib_metrics

sc.settings.set_figure_params(dpi=200, frameon=False)
sc.set_figure_params(dpi=500)
plt.rcParams['figure.figsize'] = (5, 5)
# -

from metrics_lib import *





# +
DATASET_KEY = 'dataset'
SAMPLE_KEY = 'sample'
EMB_KEY = 'X_scanvi_emb'
REF_SAMPLE_FRAC = 1.
# REF_SAMPLE_FRAC = 0.1

REF_DATASET_SUBSET = ['Meyer_2019', 'Teichmann_Meyer_2019']
# REF_DATASET_SUBSET = []
QUERY_DATASET_SUBSET = ['Meyer_2021_3prime', 'Regev_2021_Nuclei']
# QUERY_DATASET_SUBSET = []

HLCA_DATA_PATH = os.path.expanduser('~/data/HLCA/hlca_all_cellxgene.h5ad')
METRIC_OUT_PATH = os.path.expanduser('~/io/q2r_metric/')
OUTPUT_FORMAT = f"metric_results_ref_frac_{REF_SAMPLE_FRAC}" + "_query_{query_dataset}_metric_{metric}.pkl"
# -



adata = sc.read(HLCA_DATA_PATH, backed='r')
adata

# +
reference_obs = adata.obs[adata.obs['core_or_extension'] == 'core']
if len(REF_DATASET_SUBSET) > 0:
    reference_obs = reference_obs[reference_obs[DATASET_KEY].isin(REF_DATASET_SUBSET)]

reference_obs[DATASET_KEY].cat.remove_unused_categories().value_counts()
# -



query_all_obs = adata.obs[adata.obs['core_or_extension'] == 'extension']
if len(QUERY_DATASET_SUBSET) > 0:
    query_all_obs = query_all_obs[query_all_obs[DATASET_KEY].isin(QUERY_DATASET_SUBSET)]
query_all_obs[DATASET_KEY].cat.remove_unused_categories().value_counts()

query_keys = query_all_obs[DATASET_KEY].cat.remove_unused_categories().value_counts().index[::-1]
query_keys


# +
def is_already_calculated(qkey, output_schema):
    already_calculated = True
    for key in output_schema:
        filename = os.path.join(METRIC_OUT_PATH, OUTPUT_FORMAT.format(query_dataset=qkey, metric=key))
        if os.path.exists(filename):
            pass
            # print(f"File already exists: {filename}")
        else:
            already_calculated = False
    return already_calculated

def save_output(qkey, output):
    for key in output:
        filename = os.path.join(METRIC_OUT_PATH, OUTPUT_FORMAT.format(query_dataset=qkey, metric=key))
        if os.path.exists(filename):
            print(f"WARNING: File already exists: {filename}")
        with open(filename, 'wb') as f:
            pickle.dump(output[key], f)

if REF_SAMPLE_FRAC < 1.:
    np.random.seed(0)
    current_reference_obs = reference_obs.sample(frac=REF_SAMPLE_FRAC)
else:
    current_reference_obs = reference_obs
# -

tasks = {
    'scib': lambda adata_all, adata_query, dry_run: calculate_scib_metrics(
        adata_all, EMB_KEY, 'core_or_extension', 
        dataset_key=DATASET_KEY, sample_key=SAMPLE_KEY, dry_run=dry_run),
    # Note: adata_query.X should be log normalized.
    'symphony': lambda adata_all, adata_query, dry_run: calculate_symphony_metrics(
        adata_all, adata_query, EMB_KEY, 'donor_id',
        dataset_key=DATASET_KEY, sample_key=SAMPLE_KEY, dry_run=dry_run),
    'other': lambda adata_all, adata_query, dry_run: calculate_other_metrics(
        adata_all, adata_query, EMB_KEY,
        dataset_key=DATASET_KEY, sample_key=SAMPLE_KEY, dry_run=dry_run),
}
taks_keys = list(tasks.keys())


def run(qkey, task, dry_run=False):
    if not dry_run:
        current_query_obs = query_all_obs[query_all_obs[DATASET_KEY] == qkey].copy()
        metric_obs = pd.concat([current_reference_obs, current_query_obs])
        adata_all = adata[metric_obs.index]
        adata_query = adata[metric_obs[metric_obs['core_or_extension'] == 'extension'].index]
    else:
        adata_all, adata_query = None, None

    output_schema = task(adata_all, adata_query, True)
    already_calculated = is_already_calculated(qkey, output_schema)
    if dry_run:
        return already_calculated
    if not already_calculated:
        try:
            output = task(adata_all, adata_query, False)
            save_output(qkey, output)
        except Exception as e:
            print(e)
            return False
    return True


# +
remaining_runs = np.ones((len(query_keys), len(taks_keys)))
for i, qkey in enumerate(query_keys):
    for j, tkey in enumerate(taks_keys):
        task = tasks[tkey]
        if run(qkey, task, dry_run=True):
            remaining_runs[i, j] = 0

print(remaining_runs.shape)
print(f"{remaining_runs.sum()} out of {np.ones_like(remaining_runs).sum()} remaining.")

# Random task picking strategy
while remaining_runs.sum() > 0:
    print(f"{remaining_runs.sum()} out of {np.ones_like(remaining_runs).sum()} remaining.")
    i = np.random.choice(range(len(query_keys)))
    j = np.random.choice(range(len(taks_keys)))
    qkey = query_keys[i]
    tkey = taks_keys[j]
    task = tasks[tkey]
    if run(qkey, task, dry_run=True):
        remaining_runs[i, j] = 0
        print(f"{tkey} for {qkey} already calculated. Skipping.")
        continue
    else:
        print(f"Running {tkey} for {qkey}.")
        if run(qkey, task, dry_run=False):
            remaining_runs[i, j] = 0
# -


query_items = {
    'per_dataset': query_all_obs['dataset'].cat.remove_unused_categories().unique(),
    'per_sample': query_all_obs['sample'].cat.remove_unused_categories().unique(),
    'per_cell': query_all_obs.index,
}
query_items


# +
dataframes_to_save = {}
query_items = {
    'per_dataset': query_all_obs['dataset'].cat.remove_unused_categories().unique(),
    'per_sample': query_all_obs['sample'].cat.remove_unused_categories().unique(),
    'per_cell': query_all_obs.index,
}


for i, qkey in enumerate(query_keys):
    for j, tkey in enumerate(taks_keys):
        task = tasks[tkey]
        output_schema = task(None, None, True)
        for key in output_schema:
            filename = os.path.join(METRIC_OUT_PATH, OUTPUT_FORMAT.format(query_dataset=qkey, metric=key))
            if not os.path.exists(filename):
                print(f"Error: File Not Found!: {filename}")
                continue
            with open(filename, 'rb') as f:
                print(f"Reading {filename}")
                output = pickle.load(f)
                # print(output)
            for grouping in output:
                df = output[grouping]
                if isinstance(df, pd.Series):
                    df = pd.DataFrame({'value': df, 'metric': key, 'grouping': grouping}, index=df.index)
                if not isinstance(df, pd.DataFrame):
                    print(f"{grouping} is of type {type(df)} not pd.DataFrame.")
                    continue
                
                df = df[df.index.isin(query_items[grouping])].copy()
                if grouping == 'per_dataset':
                    df = df[df.index.isin([qkey])].copy()
                save_key = f"{key}__{grouping}"
                if save_key not in dataframes_to_save:
                    dataframes_to_save[save_key] = df
                else:
                    dataframes_to_save[save_key] = pd.concat([dataframes_to_save[save_key], df])
# -
dataframes_to_save


for save_key in dataframes_to_save:
    dataframes_to_save[save_key].to_csv(f"./results/{save_key}.csv", index=True)













# +
# # Full running strategy
# for qkey in query_keys:
#     print(qkey)
#     current_query_obs = query_all_obs[query_all_obs[DATASET_KEY] == qkey].copy()
#     metric_obs = pd.concat([current_reference_obs, current_query_obs])
#     adata_all = adata[metric_obs.index]
#     adata_query = adata[metric_obs[metric_obs['core_or_extension'] == 'extension'].index]

#     for x in :
#         output_schema = x(True)
#         already_calculated = is_already_calculated(qkey, output_schema)
#         if not already_calculated:
#             output = x(False)
#             save_output(qkey, output_schema)
# -








# + active=""
#

# + active=""
#


# + active=""
#

# + active=""
# # Small sample data to test if the metric runs with no error


# + active=""
# metric_obs = pd.concat([reference_obs, query_all_obs]).sample(1000)
# len(metric_obs)

# + active=""
# c = calculate_other_metrics(adata[metric_obs.index], 
#                             adata[metric_obs[metric_obs['core_or_extension'] == 'extension'].index],
#                             EMB_KEY,
#                             dataset_key=DATASET_KEY, sample_key=SAMPLE_KEY)
# c

# + active=""
# b=calculate_symphony_metrics(adata[metric_obs.index], 
#                              adata[metric_obs[metric_obs['core_or_extension'] == 'extension'].index],
#                              EMB_KEY, 'donor_id',
#                              dataset_key=DATASET_KEY, sample_key=SAMPLE_KEY)
# b

# + active=""
# a=calculate_scib_metrics(adata[metric_obs.index], EMB_KEY, 'core_or_extension',
#                          dataset_key=DATASET_KEY, sample_key=SAMPLE_KEY)
# a

# + active=""
#
# -


