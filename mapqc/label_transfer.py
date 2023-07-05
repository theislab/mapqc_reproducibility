#!/bin/env python
import sys
from typing import Literal, Optional

import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData
from sklearn.neighbors import KNeighborsTransformer

# from paths import DATA_DIR, FIG_DIR


def weighted_knn_trainer(
    train_adata: AnnData,
    train_adata_emb: str,
    label_key: Optional[str],
    n_neighbors: int = 50,
):
    """
    Trains a weighted KNN classifier on `train_adata`.

    Parameters
    ----------
    train_adata: :class:`~anndata.AnnData`
        Annotated dataset to be used to train KNN classifier with
        ``label_key`` as the target variable.
    train_adata_emb
        Name of the obsm layer to be used for calculation of neighbors.
        If set to "X", anndata.X will be used.
    label_key
        Name of the column to be used as target variable (e.g. cell_type)
        in ``train_adata`` and ``query_adata``.
    n_neighbors
        Number of nearest neighbors in KNN classifier.

    Returns
    -------
    Fitted k-nearest neighbor transformer.
    """
    print(
        f"Weighted KNN with n_neighbors = {n_neighbors} ... ",
        end="",
    )
    k_neighbors_transformer = KNeighborsTransformer(
        n_neighbors=n_neighbors,
        mode="distance",
        algorithm="brute",
        metric="euclidean",
        n_jobs=-1,
    )
    if train_adata_emb == "X":
        train_emb = train_adata.X
    elif train_adata_emb in train_adata.obsm.keys():
        train_emb = train_adata.obsm[train_adata_emb]
    else:
        raise ValueError(
            "train_adata_emb should be set to either 'X'"
            + "or the name of the obsm layer to be used!"
        )
    k_neighbors_transformer.fit(train_emb)
    return k_neighbors_transformer


def weighted_knn_transfer(
    query_adata: AnnData,
    query_adata_emb: str,
    ref_adata_obs: pd.DataFrame,
    label_keys: Optional[str],
    knn_model: KNeighborsTransformer,
    threshold: float = 1.0,
    pred_unknown: bool = False,
    mode: Literal["paper", "package"] = "package",
    numpy_pickle: bool = False,
):
    """Annotates ``query_adata`` cells with an input trained weighted KNN classifier.

    Parameters
    ----------
    query_adata
        Annotated dataset to be used to queryate KNN classifier. Embedding to be used.
    query_adata_emb
        Name of the obsm layer to be used for label transfer. If set to "X",
        query_adata.X will be used.
    ref_adata_obs
        observations of reference Anndata.
    label_keys
        Names of the columns to be used as target variables (e.g. cell_type) in ``query_adata``.
    knn_model
        knn model trained on reference adata with weighted_knn_trainer function.
    threshold: float
        Threshold of uncertainty used to annotating cells as "Unknown". cells with
        uncertainties higher than this value will be annotated as "Unknown".
        Set to 1.0 to keep all predictions. This enables one to later on play
        with thresholds.
    pred_unknown
        ``False`` by default. Whether to annotate any cell as "unknown" or not.
        If `False`, ``threshold`` will not be used and each cell will be annotated
        with the label which is the most common in its ``n_neighbors`` nearest cells.
    mode
        Has to be one of "paper" or "package". If mode is set to "package",
        uncertainties will be 1 - P(pred_label), otherwise it will be 1 - P(true_label).
    numpy_pickle
        Whether to source from saved k neighbors after transformation.

    Returns
    -------
    pred_labels
        pd.DataFrame of predicted cell type labels per query observation
    uncertainties
        pd.DataFrame of predicted cell type label uncertainties per query observation
    """
    if not isinstance(knn_model, KNeighborsTransformer):
        raise ValueError(
            "knn_model should be of type sklearn.neighbors._graph.KNeighborsTransformer!"
        )

    if query_adata_emb == "X":
        query_emb = query_adata.X
    elif query_adata_emb in query_adata.obsm.keys():
        query_emb = query_adata.obsm[query_adata_emb]
    else:
        raise ValueError(
            "query_adata_emb should be set to either 'X' or the name of the obsm layer to be used!"
        )

    if not numpy_pickle:
        top_k_distances, top_k_indices = knn_model.kneighbors(X=query_emb)
        np.savez("top", top_k_distances, top_k_indices)
    else:
        print("pickle!")
        npzfile = np.load("top.npz")
        top_k_distances = npzfile["arr_0"]
        top_k_indices = npzfile["arr_1"]

    stds = np.std(top_k_distances, axis=1)
    stds = (2.0 / stds) ** 2
    stds = stds.reshape(-1, 1)

    top_k_distances_tilda = np.exp(-np.true_divide(top_k_distances, stds))

    weights = top_k_distances_tilda / np.sum(
        top_k_distances_tilda, axis=1, keepdims=True
    )

    uncertainties = pd.DataFrame(columns=["uncert"], index=query_adata.obs_names)
    pred_labels = pd.DataFrame(columns=[label_keys], index=query_adata.obs_names)
    y_train_labels = ref_adata_obs[label_keys].values

    for i, weight in enumerate(weights):
        unique_labels = np.unique(y_train_labels[top_k_indices[i]])
        best_label, best_prob = None, 0.0
        for candidate_label in unique_labels:
            candidate_prob = weight[
                y_train_labels[top_k_indices[i]] == candidate_label
            ].sum()
            if best_prob < candidate_prob:
                best_prob = candidate_prob
                best_label = candidate_label

        pred_label = (
            best_label if not pred_unknown or best_prob >= threshold else "Unknown"
        )

        if mode == "package":
            uncertainties.iloc[i]["uncert"] = max(1 - best_prob, 0)
        else:
            raise Exception("Inquery Mode!")

        pred_labels.iloc[i][label_keys] = pred_label

    print("finished!")

    return pred_labels, uncertainties


if __name__ == "__main__":
    query_dataset = sys.argv[1]
    outpath = sys.argv[2]
    full_emb = sc.read_h5ad(
        "../data/HLCA_extended/HLCA_extended/HLCA_extended_scarches_emb.h5ad"
    )  # FIXME: change to DATA_DIR
    ref = full_emb[full_emb.obs.core_or_extension == "core", :].copy()
    cts_ordered = pd.read_csv(
        "../supporting_files/celltype_structure_and_colors/manual_anns_and_leveled_anns_ordered.csv",
        index_col=0,
    )  # FIXME: change to DATA_DIR
    query = full_emb[full_emb.obs.dataset == query_dataset, :].copy()

    # Fit k-NN transformer
    k_neighbors_transformer = weighted_knn_trainer(
        train_adata=ref,
        train_adata_emb="X",
        label_key="ann_level",  # this argument is not used in this function
        n_neighbors=50,
    )

    # Transform query data
    labels, uncert = weighted_knn_transfer(
        query_adata=query,
        query_adata_emb="X",
        label_keys="Level",
        knn_model=k_neighbors_transformer,
        ref_adata_obs=ref.obs.join(cts_ordered, on="manual_ann"),
    )
    labels.to_csv(outpath + "/" + query_dataset + "_labels.csv")
    uncert.to_csv(outpath + "/" + query_dataset + "_uncert.csv")
