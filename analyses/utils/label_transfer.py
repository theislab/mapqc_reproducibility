# This code is adapted from
# https://github.com/openpipelines-bio/openpipeline/blob/main/src/labels_transfer/utils/helper.py
# and
# https://github.com/openpipelines-bio/openpipeline/blob/main/src/labels_transfer/pynndescent_knn/script.py

import scanpy as sc
import numpy as np
from pynndescent import PyNNDescentTransformer
from sklearn.neighbors import KNeighborsClassifier


def distances_to_affinities(distances):
    # Apply Gaussian kernel to distances
    stds = np.std(distances, axis=1)
    stds = (2.0 / stds) ** 2
    stds = stds.reshape(-1, 1)
    distances_tilda = np.exp(-np.true_divide(distances, stds))

    # normalize the distances_tilda
    # if the sum of a row of the distances tilda equals 0,
    # set normalized distances for that row to 1
    # else divide the row values by the value of the sum of the row
    distances_tilda_normalized = np.where(
        np.sum(distances_tilda, axis=1, keepdims=True) == 0,
        1,
        distances_tilda / np.sum(distances_tilda, axis=1, keepdims=True),
    )
    return distances_tilda_normalized


def perform_label_transfer(
    r_adata: sc.AnnData,
    q_adata: sc.AnnData,
    reference_obs_targets: list,
    reference_obsm_features: str = "X",
    query_obsm_features: str = "X",
    k: int = 30,
):
    """
    Performs label transfer from reference to query data using pynndescent knn.
    Arguments:
        r_adata: sc.AnnData
            Reference data.
        q_adata: sc.AnnData
            Query data.
        reference_obs_targets: list
            List of r_adata.obs column names containing label to transfer
            to query data.
        reference_obsm_features: str = None
            Features to use from reference data. Set either to "X" to use
            r_adata.X, or to a key in r_adata.obsm.
        query_obsm_features: str = None
            Features to use from query data. Set either to "X" to use
            q_adata.X, or to a key in q_adata.obsm.
        k: int = 30
            Number of neighbors to use for label transfer.
    Returns:
        q_adata: sc.AnnData
            Query data with transferred labels under [original_label]_pred and
            label transfer probabilities under [original_label]_prob.
    """
    # check input parameters:
    if reference_obsm_features == "X":
        print("Using .X of reference data")
        train_X = r_adata.X
    else:
        print(f"Using .obsm[{reference_obsm_features}] of reference data")
        train_X = r_adata.obsm[reference_obsm_features]
    if query_obsm_features == "X":
        print("Using .X of query data")
        inference_X = q_adata.X
    else:
        print(f"Using .obsm[{query_obsm_features}] of query data")
        inference_X = q_adata.obsm[query_obsm_features]

    neighbors_transformer = PyNNDescentTransformer(
        n_neighbors=k,
        parallel_batch_queries=True,
    )
    neighbors_transformer.fit(train_X)

    # Square sparse matrix with distances to n neighbors in reference data
    reference_neighbors = neighbors_transformer.transform(inference_X)
    query_neighbors = neighbors_transformer.transform(train_X)

    # For each target, train a classifier and predict labels
    for obs_tar in reference_obs_targets:
        obs_pred = obs_tar + "_pred"
        obs_proba = obs_tar + "_prob"
        print(f"Predicting labels for {obs_tar}")

        weights_dict = {
            "uniform": "uniform",
            "distance": "distance",
            "gaussian": distances_to_affinities,
        }

        train_y = r_adata.obs[obs_tar].to_numpy()
        classifier = KNeighborsClassifier(
            n_neighbors=k, metric="precomputed", weights="distance"
        )
        classifier.fit(X=query_neighbors, y=train_y)
        predicted_labels = classifier.predict(reference_neighbors)
        probabilities = classifier.predict_proba(reference_neighbors).max(axis=1)

        # save_results
        q_adata.obs[obs_pred] = predicted_labels
        q_adata.obs[obs_proba] = probabilities
    # return result
    return q_adata
