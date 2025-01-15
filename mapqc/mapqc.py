import random
import numpy as np
import pandas as pd
import scanpy as sc
import os
import pickle
from typing import Union, List, Literal
from .mapqc_dists import DistResults
from .sampler import sample_query_cells
from .result_processing import add_nhood_stats_to_adata, add_cell_level_stats_to_adata


class MapqcResults(DistResults):
    def __init__(
        self,
        adata: sc.AnnData,
        ref_q_key: str,
        ref_cat: str,
        query_cat: str,
        condition_key: str,
        control_cats: Union[str, List[str]],
        case_cats: Union[str, List[str]],
        emb_key: str,
        dataset_key: str,
        sample_key: str,
        control_or_case: str,
        distance_type: Literal[
            "pairwise_euclidean", "centroid_euclidean", "sinkhorn", "edistance"
        ] = "pairwise_euclidean",
        sampling_group_cat: str = None,  # either
        percentile_cutoff: int = 66,
        center_cells: np.ndarray = None,
        k: int = None,
        max_k: int = None,
        n_nhoods: int = None,
        min_n_samples_q: int = 1,
        seed: int = None,
        n_cores: int = 1,
        verbose: bool = True,
    ):
        """
        ... class docstring

        Arguments
        ---------
        control_or_case: str
            Should be set to either "case" or "control"
        """
        self.percentile_cutoff = percentile_cutoff
        self.ref_q_key = ref_q_key
        self.ref_cat = ref_cat
        self.query_cat = query_cat
        self.result_stats = dict()
        # TO DO: update this to either a single condition (control or case), or
        # restructure such that it runs mapQC for multiple conditions.
        # run a couple of preparory functions, then call the parent init
        # to initialize the rest of the mapqc_results object
        # Check that control_cat and case_cat are both categories in adata.obs[condition_key]:
        if isinstance(control_cats, str):
            control_cats = [control_cats]
        if isinstance(case_cats, str):
            case_cats = [case_cats]
        for control_cat in control_cats:
            if control_cat not in adata.obs[condition_key].values:
                raise ValueError(
                    f"'{control_cat}' is not a category in adata.obs['{condition_key}']"
                )
        for case_cat in case_cats:
            if case_cat not in adata.obs[condition_key].values:
                raise ValueError(
                    f"'{case_cat}' is not a category in adata.obs['{condition_key}']"
                )
        if not pd.isnull(sampling_group_cat):
            if sampling_group_cat not in adata.obs.columns:
                raise ValueError(f"'{sampling_group_cat}' is not a column in adata.obs")
        self.control_cats = control_cats
        self.case_cats = case_cats

        if control_or_case == "control":
            subset_cats = control_cats
        elif control_or_case == "case":
            subset_cats = case_cats
        else:
            raise ValueError(
                f"control_or_case should be set to either 'control' or 'case', not '{control_or_case}'"
            )
        # randomize order of adata cells, this is relevant for plotting down the line
        if not pd.isnull(seed):
            np.random.seed(seed)
        adata = adata[np.random.permutation(adata.n_obs)].copy()
        if emb_key != "X":
            if emb_key not in adata.obsm.keys():
                raise ValueError(f"Embedding key '{emb_key}' not found in adata.obsm")
            else:
                print(f"Using adata.obsm['{emb_key}'] for mapQC analysis")
                self.adata = sc.AnnData(adata.obsm[emb_key], obs=adata.obs)
                self.adata.obsm["X_umap"] = adata.obsm["X_umap"]
        else:
            self.adata = adata
        if verbose:
            print(f"Subsetting query to {control_or_case} condition(s) '{subset_cats}'")
        self.adata = self.adata[
            (self.adata.obs[ref_q_key] == ref_cat)
            | (self.adata.obs[condition_key].isin(subset_cats))
        ].copy()
        query_cell_idc = np.where(self.adata.obs[ref_q_key] == query_cat)[0].tolist()
        n_cells = len(query_cell_idc)
        n_samples = len(self.adata.obs.iloc[query_cell_idc, :][sample_key].unique())
        n_cells_ref = (self.adata.obs[ref_q_key] == ref_cat).sum()
        n_cells_ref_and_q = n_cells_ref + n_cells
        if verbose:
            print(
                f"Number of {control_or_case} query samples: {n_samples}, number of query cells: {n_cells}"
            )
        if center_cells is not None:
            n_nhoods = len(center_cells)

        if k is None:
            # set the k, this will be the size of the neigborhood in which we will calculate the distances
            # note that this should roughly correspond to a cell subtype/substate, or a sizeable proportion of that.
            # To estimate the amount of cells in a cell subtype, we divide the number of reference cells by
            # 2500 (a rather arbitrary estimate of the number of cells in a cell subtype, note that the k
            # can also be manually set by the user).
            n_substates = n_cells_ref / 2500
            # Then we divide the number of reference + query cells by the estimated number of cell subtypes
            # to get the k value, i.e. the size of the neighborhoods we will use.
            k = int(n_cells_ref_and_q / n_substates)
        if max_k is None:
            max_k = int(k * 2)
        if n_nhoods is None:
            # set the number of neighborhoods such that each cell is expected to be covered in at least 2 neighborhoods
            # on average. Note that this is in the scenario where query mixes perfectly with reference, and is
            # spread out evenly across the reference. In practice, the query is rarely or never perfectly spread out
            # across the reference (i.e. does not include all population variation, all cell types, all anatomical
            # locations etc.), and each cell will be covered in more than 2 neighborhoods on average.
            n_nhoods = int(n_cells_ref_and_q / k * 2)
        if verbose:
            print(f"n nhoods: {n_nhoods}, k: {k}, max k: {max_k}")

        self.result_stats["n_nhoods"] = n_nhoods
        self.result_stats["k"] = k
        self.result_stats["max_k"] = max_k

        if not pd.isnull(seed):
            if verbose:
                print(f"Setting random seed to {seed}")
            random.seed(seed)
        else:
            if verbose:
                print("No random seed provided.")
        if center_cells is None:
            if sampling_group_cat is None:
                center_cell_idc = random.sample(query_cell_idc, k=n_nhoods)
                center_cells = np.array(self.adata.obs.iloc[center_cell_idc].index)
            else:
                if verbose:
                    print(
                        f"Sampling neighborhood center cells proportionally to {sampling_group_cat} groups and reference/query cell counts."
                    )
                _, center_cells = sample_query_cells(
                    mapqc_results=self,
                    n_cells=n_nhoods,
                    grouping_key=sampling_group_cat,
                    seed=seed,
                )

        super().__init__(
            adata=self.adata,
            center_cells=center_cells,
            k=k,
            emb_key="X",  # as we set it to X above, even if originally in obsm
            ref_q_key=ref_q_key,
            ref_cat=ref_cat,
            query_cat=query_cat,
            sample_key=sample_key,
            dataset_key=dataset_key,
            exclude_same_batch_for_ref_samples=True,
            max_k=max_k,
            min_n_samples_q=min_n_samples_q,
            distance_type=distance_type,
            verbose=verbose,
            n_cores=n_cores,
        )

    def run_mapqc(self):
        self.calculate_sample_dists_to_ref_samples()
        self.calculate_nhood_statistics(q=self.percentile_cutoff)
        self.calculate_cell_statistics_using_sample_level_results_per_nhood()
        add_nhood_stats_to_adata(self)
        add_cell_level_stats_to_adata(self)


# def load(path):
#     """
#     Load mapQC results from pickle file. Provide path to file ending with .pkl.
#     """
#     if not os.path.isfile(path):
#         raise ValueError(f"the given path {path} does not lead to a file.")
#     if not path.endswith(".pkl"):
#         raise ValueError("The given path is not a pickle file.")
#     with open(path, "rb") as inp:
#         return pickle.load(inp)


def load(path, adata=None):
    """
    Load mapQC results from pickle file. Replaced the old, simpler loading function
    so that it now handles some strange error with loading the AnnData object that
    is part of the MapqcResults object, code generated by Claude (chatGPT like),
    I don't understand it but it works.
    (Removed the option to add a separate AnnData object).

    Parameters
    ----------
    path : str
        Path to the pickle file
    # adata : AnnData, optional
    #     AnnData object to restore with the results. Required for new-style pickles
    #     where AnnData was saved separately.

    Returns
    -------
    DistResults
        Loaded mapQC results
    """
    if not path.endswith(".pkl"):
        raise ValueError("The given path is not a pickle file.")

    try:
        # First try normal loading (for old files)
        with open(path, "rb") as inp:
            results = pickle.load(inp)

    except KeyError as e:
        if "_adata_ref" in str(e):
            # If we got the specific AnnData reference error, try loading with a workaround
            import anndata._core.file_backing

            # Temporarily monkey patch the AnnDataFileManager
            original_setstate = (
                anndata._core.file_backing.AnnDataFileManager.__setstate__
            )

            def new_setstate(self, state):
                self.__dict__ = state.copy()
                if "_adata_ref" in state:
                    del self.__dict__["_adata_ref"]

            anndata._core.file_backing.AnnDataFileManager.__setstate__ = new_setstate

            try:
                # Try loading again
                with open(path, "rb") as inp:
                    results = pickle.load(inp)
            finally:
                # Restore original method
                anndata._core.file_backing.AnnDataFileManager.__setstate__ = (
                    original_setstate
                )
        else:
            raise

    # # For new-style pickles or if the old AnnData was corrupted
    # if adata is not None:
    #     results.adata = adata

    return results


# def run_mapqc_analysis(

# ):
#     """
#     Perform preprocessing and run mapQC analysis on the given AnnData object.

#     Parameters:
#     - adata: AnnData object containing the data
#     - ref_q_key: Key for reference/query column in adata.obs
#     - ref_cat: Category name for reference cells
#     - query_cat: Category name for query cells
#     - condition_key: Key for condition column in adata.obs
#     - coi: Condition of interest
#     - emb_key: Key for embedding in adata.obsm
#     - dataset_key: Key for dataset information in adata.obs (this is used to exclude
#     same-dataset reference samples when calculating distances)
#     - sample_key: Key for sample information in adata.obs
#     - k: Number of nearest neighbors (if None, will be calculated)
#     - max_k: Maximum number of nearest neighbors for adaptive k (if None, will be calculated)
#     - n_nhoods: Number of neighborhoods to sample (if None, will be calculated)
#     - min_n_samples_q: Minimum number of query samples
#     - seed: Random seed for reproducibility

#     Returns:
#     - dict of mapQC_results objects, one for each condition
#     """
