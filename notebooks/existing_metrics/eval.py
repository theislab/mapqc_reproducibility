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

# ## Import required libraries

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

# Set visualization settings
sc.settings.set_figure_params(dpi=200, frameon=False)
sc.set_figure_params(dpi=500)
plt.rcParams['figure.figsize'] = (5, 5)
# -

# Import functions from metrics_lib module
from metrics_lib import *



# ## Define constants and paths

# +
DATASET_KEY = 'dataset'       # The key to access the dataset information in observation metadata
SAMPLE_KEY = 'sample'         # The key to access the sample information in observation metadata
EMB_KEY = 'X_scanvi_emb'      # The key to access the embedding data in the obsm of Anndata object
REF_SAMPLE_FRAC = 1.          # Fraction of reference samples to use (set to 1 for all samples)
REF_DATASET_SUBSET = []       # List of dataset subsets for filtering reference observations (empty means no filtering)
QUERY_DATASET_SUBSET = []     # List of dataset subsets for filtering query observations (empty means no filtering)

# For testing:
# REF_DATASET_SUBSET = ['Meyer_2019', 'Teichmann_Meyer_2019']
# QUERY_DATASET_SUBSET = ['Meyer_2021_3prime', 'Regev_2021_Nuclei']
# REF_SAMPLE_FRAC = 0.1

# Path to the main data file
HLCA_DATA_PATH = os.path.expanduser('~/data/HLCA/hlca_all_cellxgene.h5ad')

# Path to the directory where metric output files will be saved
METRIC_OUT_PATH = os.path.expanduser('~/io/q2r_metric/')

# Format for generating output file names, including placeholders for variables
OUTPUT_FORMAT = f"metric_results_ref_frac_{REF_SAMPLE_FRAC}" + "_query_{query_dataset}_metric_{metric}.pkl"
# -


# ## Load data


adata = sc.read(HLCA_DATA_PATH, backed='r')
adata

# +
# Filter reference observations based on dataset subset
reference_obs = adata.obs[adata.obs['core_or_extension'] == 'core']
if len(REF_DATASET_SUBSET) > 0:
    reference_obs = reference_obs[reference_obs[DATASET_KEY].isin(REF_DATASET_SUBSET)]

reference_obs[DATASET_KEY].cat.remove_unused_categories().value_counts()
# -
# Sample reference observations based on REF_SAMPLE_FRAC
if REF_SAMPLE_FRAC < 1.:
    np.random.seed(0)
    current_reference_obs = reference_obs.sample(frac=REF_SAMPLE_FRAC)
else:
    current_reference_obs = reference_obs




# Filter query observations based on dataset subset
query_all_obs = adata.obs[adata.obs['core_or_extension'] == 'extension']
if len(QUERY_DATASET_SUBSET) > 0:
    query_all_obs = query_all_obs[query_all_obs[DATASET_KEY].isin(QUERY_DATASET_SUBSET)]
query_all_obs[DATASET_KEY].cat.remove_unused_categories().value_counts()

# A list of all relevant query keys
query_keys = query_all_obs[DATASET_KEY].cat.remove_unused_categories().value_counts().index[::-1]
query_keys



# ## Helper functions

# +
# Function to check if a metric is already calculated
def is_already_calculated(qkey, output_schema):
    # Check if files exist for the given query key and its output schema
    already_calculated = True
    for key in output_schema:
        filename = os.path.join(METRIC_OUT_PATH, OUTPUT_FORMAT.format(query_dataset=qkey, metric=key))
        if os.path.exists(filename):
            pass
            # print(f"File already exists: {filename}")
        else:
            already_calculated = False
    return already_calculated

# Function to save output
def save_output(qkey, output):
    # Saves output data to pickle files
    for key in output:
        filename = os.path.join(METRIC_OUT_PATH, OUTPUT_FORMAT.format(query_dataset=qkey, metric=key))
        if os.path.exists(filename):
            print(f"WARNING: File already exists: {filename}")
        with open(filename, 'wb') as f:
            pickle.dump(output[key], f)

# Function to run tasks and save output
def run(qkey, task, dry_run=False):
    # Make relevant data for the task
    if not dry_run:
        current_query_obs = query_all_obs[query_all_obs[DATASET_KEY] == qkey].copy()
        metric_obs = pd.concat([current_reference_obs, current_query_obs])
        adata_all = adata[metric_obs.index]
        adata_query = adata[metric_obs[metric_obs['core_or_extension'] == 'extension'].index]
    else:
        # If dry_run, no actual data is needed
        adata_all, adata_query = None, None

    # Check if output is already calculated
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
# -



# ## Metric calculation Tasks

# Define tasks and their corresponding functions
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



# ## Running

# Either use sequential or random task picking

# ### Sequential running strategy
# run everything one by one in a sequential manner

# +
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


# ### Random task picking strategy for better parallelization
# Same result as full running strtegy

# +
# Check remaining runs
remaining_runs = np.ones((len(query_keys), len(taks_keys)))
for i, qkey in enumerate(query_keys):
    for j, tkey in enumerate(taks_keys):
        task = tasks[tkey]
        if run(qkey, task, dry_run=True):
            remaining_runs[i, j] = 0

print(remaining_runs.shape)
print(f"{remaining_runs.sum()} out of {np.ones_like(remaining_runs).sum()} remaining.")

# Random task picking strategy for remaining runs
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




# ## Result Aggregation

# Define valid groups that metrics can be defined for and possible keys for them
query_items = {
    'per_dataset': query_all_obs['dataset'].cat.remove_unused_categories().unique(),
    'per_sample': query_all_obs['sample'].cat.remove_unused_categories().unique(),
    'per_cell': query_all_obs.index,
}
query_items


# +
# Aggregate and save results
dataframes_to_save = {}
query_items = {
    'per_dataset': query_all_obs['dataset'].cat.remove_unused_categories().unique(),
    'per_sample': query_all_obs['sample'].cat.remove_unused_categories().unique(),
    'per_cell': query_all_obs.index,
}


# Iterate over each query key and task key to aggregate results
for i, qkey in enumerate(query_keys):
    for j, tkey in enumerate(taks_keys):
        task = tasks[tkey]  # Get the task function based on the task key
        output_schema = task(None, None, True)   # Get the output schema of the task
        # Output for each key in schema is saved separately
        for key in output_schema:
            filename = os.path.join(METRIC_OUT_PATH, OUTPUT_FORMAT.format(query_dataset=qkey, metric=key))
            if not os.path.exists(filename):
                print(f"Error: File Not Found!: {filename}")
                continue
            # Load the saved metric output from the pickle file
            with open(filename, 'rb') as f:
                print(f"Reading {filename}")
                output = pickle.load(f)
            # Iterate over different groupings in the metric output (e.g., 'per_dataset', 'per_sample', etc.)
            for grouping in output:
                df = output[grouping]
                # If the output is a series, convert it to a dataframe with appropriate columns
                if isinstance(df, pd.Series):
                    df = pd.DataFrame({'value': df, 'metric': key, 'grouping': grouping}, index=df.index)
                # Check if the output is a valid dataframe
                if not isinstance(df, pd.DataFrame):
                    print(f"WARNING: {grouping} is of type {type(df)} not pd.DataFrame.")
                    continue
                
                # Filter the dataframe to include only relevant "query" items and not reference things
                df = df[df.index.isin(query_items[grouping])].copy()
                # For 'per_dataset' grouping, further filter the dataframe to be a query dataset
                if grouping == 'per_dataset':
                    df = df[df.index.isin([qkey])].copy()
                save_key = f"{key}__{grouping}"
                # Add or concatenate the dataframe to the dictionary of dataframes to save
                if save_key not in dataframes_to_save:
                    dataframes_to_save[save_key] = df
                else:
                    dataframes_to_save[save_key] = pd.concat([dataframes_to_save[save_key], df])
# -
# ### Saving

dataframes_to_save


# Save aggregated results to CSV files
for save_key in dataframes_to_save:
    dataframes_to_save[save_key].to_csv(f"./results/{save_key}.csv", index=True)



















# + active=""
#

# + active=""
#
# -


# ## This part help debugging and testing individual functions

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


