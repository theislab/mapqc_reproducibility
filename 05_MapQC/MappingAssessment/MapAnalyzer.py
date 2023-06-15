import datetime
import itertools
import os
from collections import defaultdict
from typing import IO, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import pandas as pd
import scanpy as sc
import scipy
import statsmodels.stats.multitest as mtc
from anndata import AnnData
from MappingAssessment import label_transfer
from numpy import ndarray
from tqdm import tqdm


class AdataHandler:

    """
    Preprocesses an `adata` object for MapQC pipeline.

    Parameters
    ----------
    adata
        AnnData object to compute the score on.
    ident
        Name of destination folder where to save the results.
    workspace
        Name of path to save the results.
    log
        Whether to enable logging.

    - setting standarized column names
    - selection of appropriate level labels
    - label transfer from reference to query
    """

    def __init__(
        self,
        adata: AnnData,
        ident: str = "test",
        workspace: str = "Data/MapAnalyzer_Results/",
        log: bool = True,
    ):
        self.log = log
        if log:
            directory = f"{ident}"
            path = os.path.join(workspace, directory)
            if not os.path.exists(path):
                os.mkdir(path)
            else:
                print("Logger directory already exists")
            self.logfile = open(f"{path}logfile.txt", "a")
        self.ident = ident
        self.workspace = workspace
        self.__adata = adata

    def standarize_columns(
        self,
        query_dataset: AnnData,
        condition_name: Union[str, None] = None,
        cell_type_col: str = "cell_label",
        dataset_col: str = "dataset",
        sample_col: str = "sample",
        condition_col: str = "condition",
        subject_col: str = "subject_ID",
        disease_exclude: List = [],
    ) -> None:
        """
        Function to make column names compliant with the MapQC nomenclature.

        Parameters
        ----------
        query_dataset
            Name of the query dataset to be analyzed.
        condition_name
            Name of the condition in query dataset. If all healthy default is None.
        cell_type_col
            Name of the column with working cell_type labels.
            Calculated in the pipeline. The creation of a custom label is left at the discretion of the user.
        dataset_col
            Name of the column with dataset of n of the cells. # {what is n?}
        sample_col
            Name of the column with sample name.
        condition_col
            Name of the column denoting disease/healthy.
            Only healthy and disease in question should be present.
        subject_col
            Name of the column denoting the subject/patient/donor/healthy volunteer ID.
        disease_exclude
            In case of multiple disease datasets, the other diseases can be exluded.

        """
        self.write("Standarizing columns...")

        # remove unwanted datasets
        adata = self.__adata[
            ~(self.__adata.obs[condition_col].isin(disease_exclude))
        ].copy()

        # get distinction between query and reference
        adata.obs["query_or_reference"] = "reference"
        adata.obs.loc[
            adata.obs[dataset_col] == query_dataset, "query_or_reference"
        ] = "query"

        # get binary health status
        adata.obs["disease_or_healthy"] = "healthy"
        if condition_name != None:
            adata.obs.loc[
                adata.obs[condition_col] == condition_name, "disease_or_healthy"
            ] = "disease"

        # either rename provided cell_label column or don't and warn the user that 'set_cell_type_levels' needs to be run then
        if isinstance(cell_type_col, str):
            adata.obs.rename(mapper={cell_type_col: "cell_label"}, inplace=True, axis=1)
        else:
            self.write(
                "No cell_type column set. Please make sure to run 'set_cell_type_levels' to obtain a unified cell type column 'cell_label'."
            )

        # rename the other columns accordingly
        adata.obs.rename(mapper={subject_col: "subject_ID"}, inplace=True, axis=1)
        adata.obs.rename(mapper={dataset_col: "dataset"}, inplace=True, axis=1)
        adata.obs.rename(mapper={sample_col: "sample"}, inplace=True, axis=1)
        self.write("Finished column standarizing\n")

        # Update to standardized `adata` attribute - FIXME adata is saved as class attribute, not variable.
        self.__adata = adata
        del adata

    # TODO: if not maybe rerun standardization pipeline again?
    def check_columns(self, adata):
        """
        Function that checks whether the columns in a
        provided adata (e.g saved) object already match the standards.

        Parameters
        ----------
        adata
            AnnData object.
        """
        required_columns = [
            "dataset",
            "sample",
            "query_or_reference",
            "disease_or_healthy",
            "subject_ID",
            "cell_label",
        ]
        for col in required_columns:
            if col not in adata.obs:
                return False
        return True

    def transfer_labels(
        self,
        embedding_column: str = "X",
        transfer_columns: List[str] = ["cell_label"],
        n_neighbors: int = 50,
        pickle: bool = False,
    ) -> None:
        """
        Function that transfers labels from reference onto query via weightedKNN transfer.
        Advisory to run 'set_cell_type_levels' first and transfer 'cell_label' column.

        Parameters
        ----------
        embedding_column
            Name of the column with the embedding (e.g ScANVI) representation.
            If embedding is in main matrix X write 'X'.
        transfer_columns
            Columns from which to transfer labels onto query.
            Can be multiple (e.g for multiple annotation levels) but using default 'cell_label' column is advisable.
        pickle
            If transfer_labels has been previously ran on e.g another column,
            the process of transfer can be sped up by using a pickle of column independent precalculations (same query).
        """

        if "query_or_reference" not in self.__adata.obs.columns:
            raise Exception(
                "Column 'query_or_reference' not found in adata. Please make sure to run 'standarize_columns'"
            )
        else:
            self.write("Transferring labels...")
            ref_adata = self.__adata[self.__adata.obs.query_or_reference == "reference"]

            self.write("Train KNNModel")
            self.knn_model = label_transfer.weighted_knn_trainer(
                train_adata=ref_adata,
                train_adata_emb=f"{embedding_column}",
                label_key="cell_label",
                n_neighbors=n_neighbors,
            )

            ref_obs = ref_adata.obs
            del ref_adata

            query_adata = self.__adata[self.__adata.obs.query_or_reference == "query"]
            pickle = pickle

            # transferring labels column by column. Using pickle after first column!
            for label in transfer_columns:
                self.write(f"transferring {label}")
                labels, uncert = label_transfer.weighted_knn_transfer(
                    query_adata=query_adata,
                    query_adata_emb=f"{embedding_column}",
                    ref_adata_obs=ref_obs,
                    label_keys=label,
                    knn_model=self.knn_model,
                    numpy_pickle=pickle,
                )

                # setting new columns with transferred labels
                if label != "cell_label":
                    self.__adata.obs[f"{label}_transf"] = np.nan
                    self.__adata.obs.loc[
                        labels.index, f"{label}_transf"
                    ] = labels.to_numpy()
                else:
                    self.__adata.obs.loc[labels.index, f"{label}"] = labels.to_numpy()

                # setting new columns witn label transfer uncertainty:
                # how sure or unsure the label is.
                # If cell surrounded by hetergenous cell types: high uncertainty
                # homogenous: low uncertainty

                self.__adata.obs[f"{label}_uncert"] = np.nan
                self.__adata.obs.loc[
                    uncert.index, f"{label}_uncert"
                ] = uncert.to_numpy()

                pickle = True

            del query_adata

    def set_cell_type_levels(
        self,
        connect_threshold: float,
        annotation_columns: List[str],
        finest_level_column: str,  # TODO: How's is this supposed to be provided at runtime?
        unlabeled_tag: str,
        transfer_columns: Optional[Union[List[str], None]] = None,
        query: bool = False,
    ) -> None:
        """
        Sets the annotation *level* for cell type annotations.
        Takes in all columns for the different annotation levels
        and decides on the best levels for a cell type annotation branch based on the reference.
        Also can set the labels for the transfered query labels after deciding on appropriate levels based only on the reference.

        Parameters
        ----------
        connect_threshold
            Connection threshold (based on PAGA graph) between two cell types
            to merge into parent cell type. The lower the coarser the final labels will be.
        annotation_columns
            Columns of cell label annotation ordered by level (lowest to highest) e.g ["ann_level_1","ann_level_2",...].
        finest_level_column
            Column that annotates each cell by the finest label available.
        unlabeled_tag
            Annotation of cells that have no annotation e.g "unlabeled" or "None".
        query
            If True, labels will be set in the query to levels decided on previously.
            Usually wouldn't be used as label levels are advisory set before label transfer.
        transfer_columns
            If labels have been transferred before deciding on appropriate levels,
            this argument needs to be provided to set 'cell label columns'.
        """
        if "query_or_reference" not in self.__adata.obs.columns:
            raise Exception(
                "Column 'query_or_reference' not found in adata. Please make sure to run 'standarize_columns'"
            )

        logfile = self.logfile if hasattr(self, "logfile") else None
        label_leveling = LabelLeveling(
            self.__adata,
            annotation_columns,
            finest_level_column,
            unlabeled_tag,
            logfile,
        )

        if query == False:
            self.__adata.obs["cell_label"] = np.nan
            self.__adata.obs.loc[
                self.__adata.obs.query_or_reference == "reference", "cell_label"
            ] = label_leveling.set_cell_labels(connect_threshold)
        elif query == True and transfer_columns != None:
            self.__adata.obs.loc[
                self.__adata.obs.query_or_reference == "query", "cell_label"
            ] = label_leveling.set_transferred_cell_labels(
                transfer_columns  # type: ignore
            )
        else:
            self.__adata.obs.loc[
                self.__adata.obs.query_or_reference == "query", "cell_label"
            ] = label_leveling.set_transferred_cell_labels(annotation_columns)

    def get_adata(self) -> AnnData:
        return self.__adata

    def write(self, text) -> None:
        """
        Writes output in logger.

        Parameters
        ----------
        text
            String to stream out in logs.
        """
        if self.log:
            print(f"{datetime.datetime.now()}\t{text}")
            text = text.encode("utf-8")
            print(f"{datetime.datetime.now()}\t{text}", file=self.logfile)
            self.logfile.flush()
        else:
            print(f"{datetime.datetime.now()}\t{text}")


class LabelLeveling:
    """
    Helper class containing the label levelling algorithm and necessary helper functions.
    Sets levels based on reference. Can also apply changes to query columns if label transfer happened on all levels.

    Parameters
    ----------
    annotation_columns
        Columns of cell label annotation ordered by level (lowest to highest) e.g ["ann_level_1","ann_level_2",...].
    finest_level_column
        Column that annotates each cell by the finest label available.
    unlabeled_tag:
        Annotation of cells that have no annotation e.g "unlabeled" or "None".
    logfile
        File to write the logs on.
    """

    def __init__(
        self,
        adata: AnnData,
        annotation_columns: List[str],
        finest_level_column: str,
        unlabeled_tag: str,
        logfile: Union[None, IO[str]] = None,
    ):
        if logfile == None:  # if logfile passed logging will be enabled
            self.log = False
        else:
            self.log = True
            self.logfile = logfile

        self.__adata = adata
        self.annotation_columns = annotation_columns
        self.finest_level_column = finest_level_column
        self.dict_parents = self.get_dict_parents()
        self.dict_children = self.get_dict_children()
        self.unlabeled_tag = unlabeled_tag

    def get_dict_parents(self) -> Dict[str, str]:
        """
        Retrieves the immediate parent of a given annotation type in the levels hierarchy.

        Returns
        -------
        A dictionary of the form dict_parents[child] = immediate_parent.
        """
        dict_parents = {
            cat: self.__adata.obs.loc[
                self.__adata.obs[self.annotation_columns[i]] == cat,
                self.annotation_columns[i - 1],
            ][0]
            for i in range(len(self.annotation_columns) - 1, 1, -1)
            for cat in self.__adata.obs[self.annotation_columns[i]].cat.categories
        }
        dict_parents.pop("None", None)
        return dict_parents

    def get_dict_children(self) -> defaultdict[str, List[str]]:
        """
        Reverses the parents dictionary mapping each parent to its children.

        Returns
        -------
        A dictionary of the form dict_children[parent] = [immediate_child_1, ..., immediate_child_n].
        """
        dict_children = defaultdict(list)
        for child, parent in sorted(self.dict_parents.items()):
            dict_children[parent].append(child)
        return dict_children

    def get_level(self, cell_label: str):
        """
        Returns the level of the annotation from 0 to n_columns-1.

        Parameters
        ----------
        cell_label
            Tag of the cell type label to retrieve the level from.

        Returns
        -------
        Integer denoting the level of the annotation.
        """
        for i, column in enumerate(reversed(self.annotation_columns)):
            if cell_label in self.__adata.obs[column].cat.categories:
                return i

    def get_parent(self, cell_type_1: str, cell_type_2: str):
        """
        Determines if two cell types are related.

        Parameters
        ----------
        cell_type_1
            Tag of the first cell type.
        cell_type_2
            Tag of the second cell type.

        Returns
        -------
        The common, most immediate parent of cell_type_1 and cell_type_2.

        - Limitations -
        1. Does not handle the case when the two children share the same parent
           at > 1 level distance.
        2. Assumes that if two children have different levels, then the child with
           the lowest level (higher in the hierarchy) is parent of the other.
        """

        # Check if one cell type is None type -> defining ancestry not possible
        if cell_type_1 == self.unlabeled_tag or cell_type_2 == self.unlabeled_tag:
            return None

        # Get both cell types levels
        level1 = self.get_level(cell_type_1)
        level2 = self.get_level(cell_type_2)

        # If they are on the same level they can either share a parent or not
        if level1 == level2:
            if (
                level1 == 0 and level2 == 0
            ):  # if they are both of lowest level (e.g immune and epthelial), ancestry not possible
                return None
            if self.dict_parents[cell_type_1] == self.dict_parents[cell_type_2]:
                return self.dict_parents[cell_type_2]

        if level1 < level2:
            tmp = cell_type_2
            while tmp in self.dict_parents.keys():
                tmp = self.dict_parents[tmp]
                if tmp == cell_type_1:
                    return cell_type_1
        if level1 > level2:
            tmp = cell_type_1
            while tmp in self.dict_parents.keys():
                tmp = self.dict_parents[tmp]
                if tmp == cell_type_2:
                    return cell_type_2
        return None

        # TODO: l1 != l2 and sharing common parent that is not eihter of the ccell types
        # Tentative pc:
        # 1. Wlog assume ct_x and ct_y attaining levels l_x, l_y resp. with l_x < l_y;
        # 2. Decrease l_y to l_y' (corresp. to ct_y') s.t. l_y' = l_x;
        # 3. Iteratively get p(ct_x) and p(ct_y') and until a common parent is found.

    def get_edges(self, graph: np.matrix, groups: List[str]) -> List[Tuple[str, str]]:
        """
        Translates a binary adjacency graph of the cell types
        into a list of edges with names of the group elements (cell labels).

        Parameters
        ----------
        graph
            Binary adjacency graph as numpy.matrix.
        groups
            List of cell label categories.

        Returns
        -------
        A list of pairs of connected cell types.
        """
        # Remove below diagonal to avoid double edges
        graph = np.tril(graph)  # np.tril removes the *upper* diagonal

        # Get indices of non-zero elements in the graph
        indices = np.nonzero(graph)

        # Create a list of edges between cell types based on the indices
        edgelist = [(groups[i], groups[j]) for i, j in zip(indices[0], indices[1])]

        return edgelist

    def set_cell_labels(self, connect_threshold: float) -> pd.Series:
        """
        Core algorithm of the class. Iteratively connects edges with connectivity >= connect_threshold.
        Assumption: interconnected cell types are harder to distinguish when mapping leading to higher uncertainty.

        Parameters
        ----------
        connect_threshold
            Threshold for PAGA connectivity graph.

        Returns
        -------
        pd.Series of coarsed labels `labels` for each observation in the reference.
        """

        reference = self.__adata[
            self.__adata.obs["query_or_reference"] == "reference"
        ].copy()

        # Step 1: Set cells to their finest annotation (i.e. highest level).
        reference.obs["cell_label"] = reference.obs[self.finest_level_column]
        categories = (
            reference.obs["cell_label"].astype("category").cat.categories
        )  # FIXME: change to sensible variable name
        self.write(f"starting with cell labels:\n{list(categories)}\n")

        iteration = 1
        same_celltype_composition = False

        # Needed to run PAGA
        sc.pp.neighbors(reference, n_neighbors=15, n_pcs=30)

        while same_celltype_composition == False:
            # Step 2: Calculate PAGA graph and get the edges.
            self.write(f"Iteration {iteration}\n")
            reference.obs["cell_label"] = reference.obs["cell_label"].astype(str)
            sc.tl.paga(reference, groups="cell_label")
            paga_graph = reference.uns["paga"]["connectivities"].toarray()

            paga_graph[np.where(paga_graph >= connect_threshold)] = 1
            paga_graph[np.where(paga_graph < connect_threshold)] = 0

            edgelist = self.get_edges(
                paga_graph, reference.obs["cell_label"].cat.categories
            )

            # Step 3: Merge edges of related nodes with connectivity >= connect_threshold
            # and set all children to their respective parental cell type.
            for edge in edgelist:
                parent = self.get_parent(edge[0], edge[1])
                if parent != None:
                    self.write(
                        f"merging related cell types {self.dict_children[parent]} to parental cell type {parent}\n"
                    )
                    reference.obs["cell_label"] = reference.obs["cell_label"].astype(
                        str
                    )
                    reference.obs.loc[
                        reference.obs["cell_label"].isin(self.dict_children[parent]),
                        "cell_label",
                    ] = parent
                    reference.obs["cell_label"] = reference.obs["cell_label"].astype(
                        "category"
                    )

            # cleanup for cell types that were not merged
            # "or set to their parental cell type because more than one generation away:"
            # FIXME: why would it be the case that cts more than one generation away can't be assigned to a parent cell type?

            # they get identified and set to their ancestral cell type if it is present in the new cell label list
            for i in itertools.combinations(
                reference.obs["cell_label"].cat.categories, 2
            ):
                if self.get_level(i[0]) == self.get_level(i[1]):  # FIXME: why?
                    pass
                else:
                    parent = self.get_parent(i[0], i[1])
                    if parent != None:
                        self.write(
                            f"merging related cell types {i[0]} and {i[1]} to parental cell type {parent}\n"
                        )
                        if parent == i[0]:
                            reference.obs["cell_label"] = reference.obs[
                                "cell_label"
                            ].astype(str)
                            reference.obs.loc[
                                reference.obs["cell_label"] == i[1], "cell_label"
                            ] = parent
                            reference.obs["cell_label"] = reference.obs[
                                "cell_label"
                            ].astype("category")
                        else:
                            reference.obs["cell_label"] = reference.obs[
                                "cell_label"
                            ].astype(str)
                            reference.obs.loc[
                                reference.obs["cell_label"] == i[0], "cell_label"
                            ] = parent
                            reference.obs["cell_label"] = reference.obs[
                                "cell_label"
                            ].astype("category")

            # Step 4: Repeat steps 2-3 on new cell labels until no changes occur in cell type composition
            if len(reference.obs["cell_label"].cat.categories) != len(categories):
                same_celltype_composition = False
                categories = reference.obs["cell_label"].cat.categories
                self.write(f"current cell labels: {list(categories)}\n")
            else:
                same_celltype_composition = all(
                    reference.obs["cell_label"].cat.categories == categories
                )
                if same_celltype_composition:
                    self.write(f"final cell labels: {list(categories)}\n")
            iteration += 1

        labels = reference.obs["cell_label"]
        del reference
        return labels

    def set_transferred_cell_labels(self, transfer_columns: List[str]) -> pd.Series:
        """
        If label transfer has been performed on query before setting the appropriate cell type levels,
        reannotates query cells with the appropriate levels.

        Parameters
        ----------
        transfer_columns
            List of columns sorted by level lowest to highest ["ann_level_1_transf","ann_level_2_transf",...].

        Returns
        -------
        pd.Series of coarsed labels `labels` for each observation in the query.
        """
        query = self.__adata[self.__adata.obs.query_or_reference == "query"].copy()

        if "cell_label" not in self.__adata.obs.columns:
            raise Exception(
                "Column 'query_or_reference' not found in adata. Please make sure to run 'set_cell_labels'"
            )

        # if transfer_columns is not None:
        if len(transfer_columns) != len(self.annotation_columns):
            raise Exception(
                "Number of transfered columns must match annotation columns"
            )

        # For each cell type present in calculated cell_label column,
        # get level and set query cells that match this label on this level to the label
        labels = list(self.__adata.obs.cell_label.astype("category").cat.categories)
        for label in labels:
            lvl = self.get_level(label)  # 0 - len(columns-1)
            column = transfer_columns[lvl]
            query.obs.loc[query.obs[column] == label, "cell_label"] = label

        labels = query.obs["cell_label"]
        del query
        return labels

    def write(self, text):
        """
        Writes output in logger.

        Parameters
        ----------
        text
            String to stream out in logs.
        """
        if self.log:
            print(f"{datetime.datetime.now()}\t{text}")
            text = text.encode("utf-8")
            print(f"{datetime.datetime.now()}\t{text}", file=self.logfile)
            self.logfile.flush()
        else:
            print(f"{datetime.datetime.now()}\t{text}")


class MappingScoring:

    """
    Central class of the package. Computation of the mixing scores.

    Parameters
    ----------
    adata_handler
        Object of class AdataHandler.
    log
        Whether to enable logging.
    """

    def __init__(self, adata_handler: AdataHandler, log: bool = True):
        self.log = log
        self.ident = adata_handler.ident
        self.workspace = adata_handler.workspace

        if log:
            today = datetime.date.today()
            # date = today.strftime("%d_%m_%Y")
            directory = f"{self.ident}/"
            path = os.path.join(self.workspace, directory)
            if not os.path.exists(path):
                os.mkdir(path)
            else:
                print("Logger directory already exists")
            self.path = path
            self.logfile = open(f"{path}logfile.txt", "a")
            self.baseline = (
                False  # Whether or not a baseline has been calculated or added
            )

        self.transformation = None  # Parameters for sample size correction
        adata = adata_handler.get_adata()

        # Check if columns of adata are standarized
        if adata_handler.check_columns(adata):
            self.__adata = adata
        else:
            raise Exception(
                "Not all columns are present in .obs. Please run AdataHandler.standarize_columns"
            )

    def get_subsampled_adata(
        self,
        sub_adata,
        sample: str,
        n_neighbors: int,
        compute_neighbors: bool = True,
        exclude_dataset: bool = True,
        embedding: str = "X",
    ):
        """
        Subsamples an `AnnData` object.

        Parameters
        ----------
        sub_adata
            `AnnData` object to subset (e.g only of certain cell type).
        sample
            Name identifier of the sample to retain (e.g query sample).
        n_neighbors
            Number of neighbors in the k-NN graph.
        compute_neighbors
            Wheter to compute the neighborhood graph.
        exclude_dataset
            Whether to exclude the dataset where the sample come from.
        embedding
            Embedding column to use.

        Returns
        -------
        A subsampled instance of an AnnData object
        """

        if exclude_dataset is True:
            # Extract `sample` from dataset
            dataset = sub_adata.obs.loc[
                sub_adata.obs["sample"] == sample, "dataset"
            ].to_numpy()[0]

            # FIXME: Subset by either `sample` or ?
            sub = sub_adata[
                (sub_adata.obs["sample"] == sample)
                | ~(sub_adata.obs["dataset"] == dataset)
            ].copy()
        else:
            sub = sub_adata

        # TODO: discuss a reduction to possibly:
        # Subset `adata` by retaining only `sample` and its neighbors in a k-NN graph
        # sub_adata = adata[adata.obs["sample"] == sample].copy()

        if compute_neighbors:
            sc.pp.neighbors(
                sub, n_neighbors=n_neighbors, n_pcs=30, use_rep=f"{embedding}"
            )

        return sub

    def count_subjects(self, adata: AnnData) -> int:
        """
        Tallies the number of subjects with at least 15 cells per cell label.

        Parameters
        ----------
        adata
            Subsampled adata (e.g per cell_label).

        Returns
        -------
        The count of subject with >= 15 cells per cell type label.
        """
        # Isolate reference samples
        sub_adata = adata[adata.obs.query_or_reference == "reference"].copy()

        table = (
            sub_adata.obs.groupby("subject_ID")  # Group by subject identifier
            .agg(
                {"subject_ID": "count"}
            )  # Count the number of cells for each subject ID
            .rename(
                columns={"subject_ID": "count"}
            )  # Substitute the subject ID with the its many instances
        )

        subjects = table.loc[table["count"] >= 15].index.tolist()
        return len(subjects)

    def get_adata(self):
        return self.__adata

    def set_adata(self, adata):
        self.__adata = adata

    def calculate_mixing_scores(
        self,
        adata: AnnData = None,
        n_neighbors: int = 15,
        min_subjects: int = 40,
        samples: Union[List[str], None] = None,
        reinitialize: bool = True,
        embedding: str = "X",
        verbose: bool = False,
        overwrite: bool = True,
        save: bool = False,
    ) -> AnnData:
        """
        Computes the mixing scores of query or reference.

        Parameters
        ----------
        adata
            `AnnData` object to calculate the mixing score on.
            Passed e.g when calculating the null distribution (baseline).
        n_neighbors
            Size of neighborhood graph considered for mixing score calculation.
        min_subjects
            Number of subjects to be sufficently (>= n_neighbors) represented in a cell type.
        samples
            Batch of samples to analyze, if `None` all query samples are analyzed.
        reinitialize
            Whether to reset the score columns or not. Not resetting when baseline is calculated.
        embedding
            Embedding column name. Use 'X' when embedding in the count matrix.
        verbose
            If `True`, log is printed in output stream.
        overwrite
            Whether or not to overwrite the `self.__adata` class attribute.
        save
            Whether or not to save the scores to the workspace.

        Returns
        -------
        The initial `AnnData` object with mixing scores saved in "score".
        """
        if adata is None:
            adata = self.__adata

        if reinitialize:
            adata.obs["score"] = np.nan

        # stats_cell_type = []
        self.cell_types = []
        min_cells_sample = n_neighbors
        if samples is None:
            # Include all samples in the query
            samples = (
                adata[adata.obs.query_or_reference == "query"]
                .obs["sample"]
                .cat.categories
            )

        for cell_type in tqdm(
            adata.obs["cell_label"].astype("category").cat.categories
        ):
            self.write(f"Analyzing {cell_type}", verbose)

            # subset anndata to cell_type
            sub_adata = adata[(adata.obs["cell_label"] == cell_type)].copy()

            # Get the number of subject for a given cell types with >= 15 cells
            count_sub = self.count_subjects(sub_adata)
            # Check if enough subjects are in reference for that cell type
            if count_sub < min_subjects:
                self.write(f"skipping {cell_type} with {count_sub} subjects", verbose)
                continue

            # add cell type to list of cell types that made it through the selection for later use
            self.cell_types.append(cell_type)

            for sample in samples:
                # TODO
                # [ consider dropping this double filtering for counts in each sample
                # A modified `count_subjects` could handle it

                # check if sample has enough cells in cell type
                count_sample_cells = len(sub_adata[sub_adata.obs["sample"] == sample])

                if (
                    count_sample_cells < min_cells_sample
                ):  # check if sample has enough cells to analyze
                    self.write(
                        f"Skipping {sample} with {count_sample_cells} cells", verbose
                    )
                    continue
                else:
                    self.write(f"Analysing {sample}", verbose)
                # ]
                # TODO

                # Get adata with query_sample + reference excluding dataset of sample origin
                sub = self.get_subsampled_adata(
                    sub_adata,
                    sample=sample,
                    n_neighbors=n_neighbors,
                    exclude_dataset=True,  # FIXME: unnecessary if only one sample from the given dataset is extracted
                    embedding=embedding,
                )

                # Assign temporary query and reference labels to not intefere with original ones (especially for baseline calculation)
                sub.obs[
                    "tmp_qr"
                ] = "reference"  # FIXME (1): if only a sample from a Dataset is extracted this line is not needed
                sub.obs.loc[
                    sub.obs["sample"] == sample, "tmp_qr"
                ] = "query"  # FIXME (2): and this would be changed to `sub.obs.loc["tmp_qr"] = "query"`

                # Get indices of reference and query cells
                pandas_table = (
                    sub.obs.tmp_qr.copy().reset_index()
                )  # Double check as possible source of BUG
                query_idx = pandas_table.loc[
                    pandas_table.tmp_qr == "query"
                ].index.tolist()
                ref_idx = pandas_table.loc[
                    pandas_table.tmp_qr == "reference"
                ].index.tolist()

                # score calculation:
                query_connect = sub.obsp["connectivities"][query_idx]
                weights_to_reference = query_connect[:, ref_idx]
                # Sum of query edge weights to each reference cell
                ref_sum = np.sum(weights_to_reference, axis=1)
                # Sum of all reference connectivities per cell / sum of all connectivities per cell
                scores = ref_sum / np.sum(query_connect, axis=1)

                # Save scores in input adata
                adata.obs.loc[
                    sub.obs.loc[sub.obs.tmp_qr == "query"].index, "score"
                ] = np.ravel(scores)

        if save:
            adata.obs["score"].to_csv(f"{self.workspace}/{self.ident}/scores.csv")

        if overwrite:
            self.__adata = adata
        else:
            return adata

    def calculate_baseline(
        self,
        n_neighbors: int = 15,
        min_subjects: int = 40,
        embedding: str = "X",
        overwrite: bool = True,
        verbose: bool = False,
        save: bool = False,
    ):
        """
        Computes the mixing scores null distribution (i.e., the baseline).
        Samples from the same dataset are excluded from the calculation.

        Parameters
        ----------
        n_neighbors
            Size of neighborhood graph considered for mixing score calculation.
        min_subjects
            Number of subjects to be sufficently (>= n_neighbors) represented in a cell type.
        embedding
            Embedding column name. use 'X' when embedding in the count matrix.
        overwrite
            Whether or not to overwrite the `self.__adata` class attribute.
        verbose
            If `True`, log is printed in output stream.
        save
            Whether or not to save the baseline scores to workspace.

        Returns
        -------
        An Overwritten `self.__adata` class attribute or a copy of `self.__adata` with baseline mixing scores.
        """

        # Subsample to reference and copy to temporary adata
        adata = self.__adata[self.__adata.obs.query_or_reference == "reference"].copy()
        grouped_samples = (
            adata.obs.groupby("dataset")["sample"].apply(set).apply(list)
        )  # What is the data type of `grouped_samples`?

        sample_list = []
        for i in itertools.zip_longest(
            *grouped_samples.tolist()
        ):  # Why would you convert `grouped_samples` to a list again?
            x = np.asarray(i)
            sample_list.append(x[np.where(x != None)])
        samples = adata.obs["sample"].cat.categories  # (*)

        for samples in sample_list:
            adata = self.calculate_mixing_scores(
                adata=adata,
                n_neighbors=n_neighbors,
                min_subjects=min_subjects,
                samples=samples,  # BUG: is `samples` an element of `sample_list` or (*)? if the former, then (*) is unused code
                reinitialize=False,
                embedding=embedding,
                verbose=verbose,
                overwrite=False,
            )

        # Log that baseline has been calculated
        self.baseline = True

        if save:
            adata.obs["score"].to_csv(
                f"{self.workspace}/baseline.csv"
            )  # FIXME: change `["score"]` to "baseline_score"
            group = self.group_adata(adata)
            file = open(f"{self.workspace}/cell_types.txt", "w")
            for ct in group.cell_label.cat.categories:
                print(ct, file=file)
            file.close()

        if overwrite:
            # self.__adata[adata.obs.index].obs = adata.obs
            self.__adata.obs.update(adata.obs)
        else:
            return adata

    def calculate_transformation(self, save: bool = True) -> None:
        """
        Computes the parameters of the regression model used to transform the mixing scores.

        Parameters
        ----------
        save
            Save the regression model parameters to the workspace.
        """
        # baseline needs to be calculated
        if self.baseline is False:
            raise Exception("You need to add or calculate the baseline first")
        else:
            # subsample adata to reference
            baseline = self.__adata[
                self.__adata.obs["query_or_reference"] == "reference"
            ]

            # group adata by sample and cell label + log_count = log(sample size)
            baseline_g = self.group_adata(adata=baseline)
            transformation = []

            for ct in self.cell_types:
                # linear regression for all cell types
                x = baseline_g[baseline_g.cell_label == ct]["log_count"]
                y = baseline_g[baseline_g.cell_label == ct]["score"]
                slope, intercept, r, p, std_err = scipy.stats.linregress(x, y)
                transformation.append((ct, slope, intercept, r, p, std_err))

            # Save the transformation as pandas column
            transformation = pd.DataFrame(
                transformation,
                columns=("cell_label", "slope", "intercept", "r", "p", "std_err"),
            )
            transformation.set_index("cell_label", inplace=True)  # type: ignore
            self.transformation = transformation  # type: ignore

            if save:
                transformation.to_csv(f"{self.workspace}/transformation.csv")  # type: ignore

    def add_transformation(self, transformation_path: str) -> None:
        """
        Adds transformation from disc.

        Parameters
        ----------
        transformation_path
            File path where the transformation has been saved.
        """
        self.transformation = pd.read_csv(transformation_path, index_col="cell_label")

    def add_baseline(self, baseline_path: str, cell_label_path: str) -> None:
        """
        Adds baseline from disc.

        Parameters
        ----------
        baseline_path
            Path to baseline.
        cell_label_path
            Path to list of cell labels with sufficent subject count.
        """
        adata = self.__adata
        baseline_data = pd.read_csv(baseline_path, index_col=0)
        # baseline_data.set_index("Unnamed: 0", inplace=True)
        baseline_data = baseline_data.loc[
            adata[adata.obs.query_or_reference == "reference"].obs.index.tolist()
        ]
        self.__adata.obs.loc[
            adata.obs.query_or_reference == "reference",
            ["query_prop", "query_count", "deviation", "score"],
        ] = baseline_data

        cell_type_file = open(cell_label_path, "r")
        cell_types = []

        self.write("Baseline consists of following cell types:\n")

        for ct in cell_type_file.readlines():
            self.write(f"{ct}\n")
            cell_types.append(ct.strip("\n"))
        self.cell_types = cell_types
        self.baseline = True

    def add_scores(self, score_path: str) -> None:
        """
        Adds precalculated scores.

        Parameters
        ----------
        score_path
            Path to scores.
        """
        score_data = pd.read_csv(score_path, index_col=0)
        # score_data.set_index("Unnamed: 0", inplace=True)
        self.__adata.obs.loc[
            (self.__adata.obs.query_or_reference == "query"), "score"
        ] = score_data

    # TODO: as of the purposes of this function consider changing its name
    def filter_sample_mean(
        self, adata: AnnData, sample_name: str, cutoff: float = 5, verbose: bool = False
    ) -> tuple[List, List[str]]:
        """
        For each cell type compares the score to the 5th percentile of the cell type baseline distribution.
        Documents the cell types that are present in the query sample for later steps.

        Parameters
        ----------
        adata
            Adata passed from wrapper function.
        sample_name
            Name identifier of the query sample.
        cutoff
            Percentile at which a sample is rejected for a given cell type.
        verbose
            Whether to stream the result of the above operations in the logs.

        Returns
        -------
        feedback
            List of sample-specific statistics including cell type name identifiers, sample mean,
            health/disease status and whether the sample was rejected or accepted.
        cell_type_list
            List of cell types present in the sample.
        """
        feedback = []  # collects the rejection/acceptance data for each cell type
        aggr_data = self.__preprocess_adata(
            adata, only_reference=False, transform=True
        )  # observations grouped by cell label and sample with mean scores
        cell_type_list = []

        for ct in self.cell_types:
            # Obtain reference sample scores for cell type
            aggr_data = self.__preprocess_adata(
                adata, only_reference=False, transform=False
            )

            # Group reference scores by cell type
            baseline_scores = aggr_data[
                (aggr_data["cell_label"] == ct)
                & (aggr_data["query_or_reference"] == "reference")
            ].score.to_numpy()

            # Obtain sample score mean for query sample and cell type
            sample_mean = aggr_data[
                (aggr_data["cell_label"] == ct) & (aggr_data["sample"] == sample_name)
            ].score.to_numpy()

            # Check if cell type present in sample
            if len(sample_mean) == 0:
                continue
            sample_mean = sample_mean[
                0
            ]  # FIXME: merge with `sample_mean` instantiation above

            # Obtain sample health status
            sample_status = aggr_data[
                (aggr_data["cell_label"] == ct) & (aggr_data["sample"] == sample_name)
            ].disease_or_healthy.to_numpy()[0]

            # Add cell type to list
            cell_type_list.append(ct)

            # Obtain cutoff value for cell type amongus based on null distribution/baseline
            cutoff_val = np.percentile(baseline_scores, cutoff)

            if verbose:
                self.write(f"{ct}")
                self.write(
                    f"cutoff_val = {cutoff_val} % - rejected: {sample_mean < cutoff_val}"
                )
                self.write(f"sample_mean = {sample_mean}")

            # reject if sample mean below cutoff value
            if sample_mean < cutoff_val:
                rejected = True
                label = (f"{sample_status} rejected",)
            else:
                rejected = False
                label = (f"{sample_status} accepted",)

            feedback.append(
                (
                    ct,
                    sample_name,
                    sample_mean,
                    sample_status,
                    rejected,
                    label,  # FIXME: What for? Isn't `rejected` enough?
                )
            )

        return feedback, cell_type_list

    def sample_rejection(
        self, cutoff: float = 5, verbose: bool = False
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Function that manages the sample rejection.
        First rejection status for each cell type is obtained via 'filter_sample_mean'.

        Then for each sample, a custom threshold based on the cell types present in the query sample are obtained via 'get_threshold'.

        Based on that threshold the overall sample is either rejected or accepted.

        Parmeters
        ---------
        cutoff
            Percentile at which a sample is rejected for a given cell type.
        verbose
            Whether to stream the result of the above operations in the logs.

        Returns
        -------
        rejection_report
            pd.DataFrame contining acceptance/rejection status for each sample in `self.__adata`.
        sample_report
            Same as `rejection_report` but grouped by sample.
        """
        adata = self.__adata

        # Obtain samples from query
        samples = (
            adata[adata.obs.query_or_reference == "query"].obs["sample"].cat.categories
        )

        feedback_data = []  # Collects feedback from `filter_sample_mean`
        sample_thresholds = {}  # Collecting sample custom thresholds

        for sample_id in samples:
            sample_feedback, cell_type_list = self.filter_sample_mean(
                adata, sample_id, cutoff, verbose=verbose
            )
            self.write(f"{sample_id}: {cell_type_list}")
            if len(cell_type_list) > 0:
                feedback_data.extend(sample_feedback)
                sample_thresholds[sample_id] = self.get_threshold(cell_type_list)

        # create dataframe from feedback list
        rejection_report = pd.DataFrame(
            feedback_data,
            columns=[
                "cell_label",
                "sample",
                "mean",
                "health_status",
                "excluded",
                "label",
            ],
        )

        # Create a sample report
        sample_report = rejection_report.groupby("sample").agg(
            {"health_status": "first", "excluded": ["sum", "count"]}
        )
        # Get the fraction of rejected samples
        sample_report["fraction"] = (
            sample_report["excluded"]["sum"] / sample_report["excluded"]["count"]
        )

        # add rejection to obs variable
        samples = list(sample_report.index)
        adata.obs["rejection_status"] = np.nan
        adata.obs.cell_label = adata.obs.cell_label.astype("category")
        sample_report["accepted"] = True

        # Adding sample rejection status to sample report based on threshold
        for sample_id in samples:
            if (
                sample_report.loc[sample_id, "fraction"].to_numpy()
                >= sample_thresholds[sample_id]
            ):
                adata.obs.loc[
                    (adata.obs["sample"] == sample_id), "rejection_status"
                ] = "rejected"
                sample_report.loc[sample_id, "accepted"] = False
                # FIXME: Why do we need both a "rejection_status" and an "accepted" field?
            else:
                adata.obs.loc[
                    (adata.obs["sample"] == sample_id), "rejection_status"
                ] = "accepted"

        self.__adata = adata
        self.report = rejection_report
        self.sample_report = sample_report

        return rejection_report, sample_report

    def filter_data_by_disease_status(
        self,
        adata: AnnData,
        disease_status: Literal["healthy", "disease"],
        cell_type_id: str,
    ) -> AnnData:
        """
        Subsets `adata` based on the desired `disease_status`.

        Parameters
        ----------
        adata
            Object of class `AnnData`.
        disease_status
            ID of disease status. Either "healthy" or "disease".
        cell_type_id
            ID of a given cell type.

        Returns
        -------
        Subsetted `adata` input to `disease_status`.
        """
        if disease_status == "disease":
            return adata[
                (adata.obs.cell_label == cell_type_id)
                & (
                    ~(adata.obs["sample"].isin(self.rejected))
                    | (adata.obs["disease_or_healthy"] == "disease")
                )
            ].copy()
        else:
            return adata[
                (adata.obs.cell_label == cell_type_id)
                & ~(adata.obs["sample"].isin(self.rejected))
            ].copy()

    def perform_one_sided_ks_test(
        self, query_scores: ndarray, reference_scores: ndarray
    ) -> Union[ndarray, None]:
        """
        Computes the p-value of one-sided KS test.

        Parameters
        ----------
        query_scores
            np.ndarray of mixing scores for the query.
        reference_scores
            np.ndarray of mixing scores for the reference.

        Returns
        -------
        pvalues
            np.ndarray of uncorrected p-values from the KS test.
        """
        if min(len(query_scores), len(reference_scores)) > 0:
            pvalues = scipy.stats.ks_2samp(
                reference_scores, query_scores, alternative="less"
            ).pvalue
            return pvalues
        return None

    def multiple_testing_correction(
        self, tested_cell_types: pd.DataFrame, cutoff_pval: float
    ) -> pd.Series:
        """
        Performs multiple testing correction for the KS statistic.

        Parameters
        ----------
        tested_cell_types
            pd.DataFrame of uncorrected p-values for each cell type.
        cutoff_pval
            Cutoff threshold used for the KS test.

        Returns
        -------
        ks_positive_cts
            pd.Series of cell types in the query whose score distribution is significantly different from
            the baseline according to a one-sided KS test.
        """
        tested_cell_types["pvals_corr"] = mtc.multipletests(
            tested_cell_types["pvalue"], 0.05, "fdr_bh"
        )[1]
        ks_positive_cts = tested_cell_types[
            tested_cell_types["pvals_corr"] < cutoff_pval
        ].cell_label
        return ks_positive_cts

    # TODO: definitely change the name of the following function - not descriptive
    def cell_type_reporting(
        self,
        cutoff_pval: float = 0.05,
        disease_status: Literal["healthy", "disease"] = "healthy",
    ) -> Union[tuple[pd.Series, pd.DataFrame], None]:
        """
        Checks whether the distribution of healthy query samples in a cell type differs
        from the reference distribution via a one-sided Kolmogorov-smirnov-test.

        Parameters
        ----------
        cutoff_pval
            Pvalue at which a cell type is flagged

        Returns
        -------
        ks_positive_cts
            pd.Series of cell types in the query whose score distribution is significantly different from
            the baseline according to a one-sided KS test.
        tested_cell_types
            pd.DataFrame of the KS test statistics for each cell type.
        """
        adata = self.__adata
        if not hasattr(self, "rejected"):
            raise AttributeError(
                "calculate rejected samples before cell type level rejection!"
            )
        if not hasattr(self, "affected_cell_types") and disease_status == "disease":
            raise AttributeError(
                "Assess residual batch effect in healthy samples before accessing disease effect!"
            )

        tested_cell_types = []
        for ct in self.cell_types:
            if disease_status == "disease" and ct in self.affected_cell_types.tolist():  # type: ignore
                self.write(
                    f"skipping cell type {ct} for disease effect evaluation: confounded healthy"
                )
                continue

            adata_ct = self.filter_data_by_disease_status(adata, disease_status, ct)

            # Group by sample and cell type
            grouped = self.__preprocess_adata(
                adata=adata_ct, only_reference=False
            )  # FIXME: change to sensible variable name

            # Obtain healthy query sample scores
            query_scores = grouped[
                (grouped["query_or_reference"] == "query")
                & (grouped["disease_or_healthy"] == disease_status)
            ].score.to_numpy()

            # Obtain reference sample scores
            reference_scores = grouped[
                (grouped["query_or_reference"] == "reference")
                & (grouped["disease_or_healthy"] == "healthy")
            ].score.to_numpy()

            # Perform one-sided KS-test
            pval = self.perform_one_sided_ks_test(query_scores, reference_scores)
            if pval is not None:
                tested_cell_types.append((ct, pval))

        # Multiple testing correction
        tested_cell_types = pd.DataFrame(
            tested_cell_types, columns=["cell_label", "pvalue"]
        ).sort_values("pvalue")

        if disease_status == "disease":
            if len(tested_cell_types) == 0:
                print("no cell_types for disease effect assessment")
                self.disease_cell_types = []
                self.disease_report = tested_cell_types
                return None

        ks_positive_cts = self.multiple_testing_correction(
            tested_cell_types, cutoff_pval
        )

        if disease_status == "healthy":
            self.affected_cell_types = ks_positive_cts
            self.cell_type_report = tested_cell_types
        else:
            self.disease_cell_types = ks_positive_cts
            self.disease_report = tested_cell_types

        return ks_positive_cts, tested_cell_types

    def calculate_threshold_table(self, cutoff_sample: float = 5) -> pd.DataFrame:
        """
        Retrieves acceptance/rejection for all cell types in each reference sample.

        Parameters
        ----------
        cutoff_sample
            Cutoff at which a sample is rejected for a given cell type.

        Returns
        -------
        table
            pd.DataFrame indexed by sample IDs reporting, for each sample, whether a given cell type
            is either accepted, rejected or for which information is not available.
        """
        baseline = self.__adata[self.__adata.obs["query_or_reference"] == "reference"]
        # Group reference by cell type and sample
        group = (
            baseline.obs.groupby(["cell_label", "sample"])
            .agg({"score": "mean", "dataset": "count"})
            .reset_index(drop=False)
        )

        if self.baseline is False:
            raise Exception("Add or calculate baseline first!")

        ## add check for transformation TODO: clarify
        transform = self.transformation.reset_index(drop=False, inplace=False)  # type: ignore
        perc_5 = {}
        # Calculate the transformed scores for each cell type and derive threshold per cell type
        for ct in self.cell_types:
            # Retrieve cell type-specific regression parameters
            slope = transform.loc[transform.cell_label == ct, "slope"].to_numpy()
            intercept = transform.loc[
                transform.cell_label == ct, "intercept"
            ].to_numpy()

            # BUG: Why is the fitted score computed at the dataset level? Why is this not a log-count?
            group.loc[group.cell_label == ct, "score"] = group.loc[
                group.cell_label == ct, "score"
            ] - (group.loc[group.cell_label == ct, "dataset"] * slope + intercept)
            perc_5[ct] = np.nanpercentile(
                group.loc[group.cell_label == ct, "score"].to_numpy(), cutoff_sample
            )

        # Create table
        list_cat = self.cell_types.copy()
        list_cat.append("sample")
        table = pd.DataFrame(data=[], columns=list_cat)

        table["sample"] = list(group["sample"].cat.categories)
        table.set_index("sample", verify_integrity=True, inplace=True)

        # Set values in table
        # 0: rejected
        # 1: accepted
        # -1: nan
        for ct in self.cell_types:
            th = perc_5[ct]
            sub = group[
                group.cell_label == ct
            ].score.to_numpy()  # transformed score for cell type `ct` in reference
            sub_no_nan = (sub >= th).astype(int)
            sub_no_nan[np.argwhere(np.isnan(sub))] = -1
            table[ct] = sub_no_nan

        table.reset_index(drop=False, inplace=True)
        return table

    def get_threshold(
        self, cell_type_list_query: List[str], weighted: bool = True
    ) -> ndarray:
        """
        Computes a custom threshold based on the cell types present in the query.
        Step 1:
            collects the counts of rejected cell types for reference samples.
        Step 2:
            The 95th percentile of counts is obtained. (samples in reference that have more rejected cell types than 95% of the reference)
            This then the threshold for the query sample.

        Parameters
        ----------
        cell_type_list_query
            List of cell types present in query.
        weighted
            Whether to weight samples by how many cell types from cell_type_list are present in that reference sample.

        Returns
        -------
        query_thresholds
            ndarray of thresholds (95 percentiles) for cell types in cell_type_list.
        """
        cell_type_list_query.append(
            "sample"
        )  # FIXME: change to more sensible variable name (e.g. `cell_type_list_query`)
        # subsample precalculated table
        table = self.lookup_table[cell_type_list_query].copy()
        table.set_index("sample", inplace=True)
        cell_type_list_query.remove("sample")
        table = table.T

        count_list = []
        for sample in table.columns:
            # 0: rejected; 1: accepted; -1: undefined
            counts = table[sample].value_counts().reindex([0, 1, -1], fill_value=0)
            # weight: number of defined cell types (0 and 1)
            weight = len(cell_type_list_query) - counts[-1]
            if weight > 0:
                value = counts[0] / (
                    counts[0] + counts[1]
                )  # fraction of rejected cell types
                if weighted:
                    count_list.extend(
                        [value] * weight
                    )  # BUG: how's the weighting work exactly here?
                else:
                    count_list.extend([value])

        query_thresholds = np.percentile(count_list, 95)
        return query_thresholds

    def binomial_rejection(
        self, n_rejected: int, n_healthy: int, cutoff_probability: float = 0.05
    ) -> float:
        """
        Computes the probabilty that at least n_rejected samples are rejected
        given the probabilty that a sample is rejected randomly at p = 0.05.

        Parameters
        ----------
        n_rejected
            Number of rejected samples.
        n_healthy
            Number of all healthy samples.
        cutoff_probability
            Cutoff probablity at which the dataset is rejected.

        Returns
        -------
        rejection_prob
            Probabilty that at least n_rejected samples are rejected given p = 0.05.
        """
        # 1-BINOM.DIST(O2-1,N2,0.042,TRUE)

        n_rej = max([n_rejected - 1, 0])
        rejection_prob = 1 - scipy.stats.binom.cdf(n_rej, n_healthy, cutoff_probability)
        return rejection_prob

    def analyze_mapping(
        self,
        cutoff_sample: float = 5,
        cutoff_cell_type: float = 0.05,
        cutoff_disease: float = 5,
    ) -> None:
        """
        Core function: performs the entire analysis pipeline of the dataset.

        Parameters
        ----------
        cutoff_sample
            Cutoff quantile at which a sample is rejected for a given cell type.
        cutoff_cell_type
            Cutoff p-value at which a cell type is flagged as batch effect affected.
        cutoff_disease
            Cutoff p-value at which a cell type is flagged as possibly affected by disease.
        """
        # adata = self.__adata # FIXME: not used
        # failed = False # FIXME: not used
        report_file = open(f"{self.path}report.txt", "w")

        # if no score transformation is added, recalculate
        if self.transformation is None:
            self.calculate_transformation()

        print("Parameters:", file=report_file)
        print(f"Rejection of cell_type per sample:\t{cutoff_sample}", file=report_file)

        # Construct lookup table
        self.lookup_table = self.calculate_threshold_table(cutoff_sample)

        # Calculate sample rejection
        self.sample_rejection(
            cutoff=cutoff_sample
        )  # TODO: rewrite this to work with table
        self.rejected = list(
            self.sample_report[self.sample_report.accepted == False].index
        )  # all rejected samples

        # Filter out only healthy/control rejected
        rej_healthy = []
        for rejected_ID in self.rejected:
            if (
                "healthy"
                in self.sample_report.loc[rejected_ID, "health_status"].tolist()
            ):
                rej_healthy.append(rejected_ID)
        sample_report = self.sample_report

        # Calculate probabilty that at least n_rejeted samples are rejected given the p = 0.05
        n_healthy = sum((sample_report.health_status == "healthy").to_numpy())[0]
        n_rejected = len(rej_healthy)
        rej_probability = self.binomial_rejection(
            n_rejected, n_healthy, cutoff_probability=0.05
        )

        # If it is unlikely (<= 0.05) that at least number of healthy samples is rejected at random we reject the dataset
        if rej_probability <= 0.05:  # (*)
            print(
                f"{n_rejected} of healthy({n_healthy}) samples rejected (p = {np.round(rej_probability,3)}):\n please remap with different parameters,a different method or reintegrate",
                file=report_file,
            )
            for rejected_ID in rej_healthy:  # FIXME: sensible variable name
                print(
                    f"{rejected_ID}:\n{self.sample_report.loc[rejected_ID]}\n",
                    file=report_file,
                )
            report_file.flush()
            # return #removed experimentally
        else:
            print(
                f"{n_rejected} of healthy({n_healthy}) samples rejected (p = {np.round(rej_probability,3)}):\ndataset is good to go",
                file=report_file,
            )
            # Otherwise we report the samples that got rejected and move on
            for rejected_ID in rej_healthy:
                print(
                    f"{rejected_ID}:\n{self.sample_report.loc[rejected_ID]}",
                    file=report_file,
                )
            report_file.flush()

        print("\n\n", file=report_file)
        print(
            f"After exclusion of rejected samples cell_types\nthat show signs of batch effect are filtered at p = {cutoff_cell_type}",
            file=report_file,
        )

        # assessment which cell types can be affected by residual batch effect overall
        self.cell_type_reporting(cutoff_pval=cutoff_cell_type)

        print(
            "Following cell_types show signs of strong batch effect in healthy samples",
            file=report_file,
        )
        print("Excluding from disease effect candidate cell_types:", file=report_file)
        print("\n", file=report_file)
        print(f"Cell_type\tbh - corrected p-value", file=report_file)

        for (
            ct
        ) in (
            self.affected_cell_types
        ):  # TODO: `affected_cell_types` is not descriptive enough (refers to batch-affected cell types)
            pval = self.cell_type_report[
                self.cell_type_report.cell_label == ct  # type: ignore
            ].pvals_corr.to_numpy()[
                0
            ]  # type: ignore
            if pval < 0.001:
                pval = " < 0.001"
            else:
                pval = np.round(pval, 3)
            print(f"{ct}\t{pval}", file=report_file)
            report_file.flush()

        print("\n\n", file=report_file)
        print(
            f"After exclusion of rejected samples and batch-effect affected cell_types\ndisease effect is assumed at at p = {cutoff_disease}",
            file=report_file,
        )

        # check for cell types with possible disease effect excluding batch-affected cell types
        self.cell_type_reporting(cutoff_pval=cutoff_disease, disease_status="disease")

        print(
            "Following cell_types show signs possible disease effect:", file=report_file
        )
        print("\n", file=report_file)
        print(f"Cell_type\tbh - corrected p-value", file=report_file)

        for ct in self.disease_cell_types:
            pval = self.disease_report[
                self.disease_report.cell_label == ct  # type: ignore
            ].pvals_corr.to_numpy()[
                0
            ]  # type: ignore
            if pval < 0.001:
                pval = " < 0.001"
            else:
                pval = np.round(pval, 3)
            print(f"{ct}\t{pval}", file=report_file)
        report_file.flush()

    # Bring up for enhanced interpretability of more detailed analyses using this function
    def group_adata(self, adata: AnnData) -> AnnData:
        """
        Refactors and extends adata fields.

        Parameters
        ----------
        adata
            AnnData object to group.

        Returns
        -------
        group_adata
            Extended AnnData object.
        """
        group_adata = (
            adata.obs.groupby(["cell_label", "sample"])
            .agg({"score": "mean", "dataset": "first", "subject_ID": "count"})
            .dropna()
            .reset_index(drop=False)
        )
        group_adata["cell_label"] = group_adata["cell_label"].astype("category")
        group_adata["cell_label"] = group_adata[
            "cell_label"
        ].cat.remove_unused_categories()
        group_adata = group_adata.rename(columns={"subject_ID": "count"})
        group_adata["log_count"] = np.log(group_adata["count"])
        return group_adata

    def __preprocess_adata(
        self, adata: AnnData, only_reference: bool = True, transform: bool = True
    ) -> AnnData:
        """
        Groups `adata` by cell type and sample and applies scores correction.

        Parmeters
        ---------
        only_reference
            Whether to exclude query from aggregated table.
        transform
            Whether to apply sample size-correction to sample means.

        Returns
        -------
        aggr_data
            Preprocessed `adata`.
        """

        group = adata.obs.groupby(["cell_label", "sample"]).agg(
            {
                "score": "mean",
                "query_or_reference": "first",
                "dataset": "first",
                "disease_or_healthy": "first",
                "subject_ID": "count",
            }
        )
        group = group.rename(columns={"subject_ID": "count"})
        aggr_data = group.reset_index(drop=False)
        aggr_data.loc[~aggr_data.cell_label.isin(self.cell_types), "score"] = np.nan
        aggr_data.dropna(inplace=True)

        aggr_data["cell_label"] = aggr_data["cell_label"].astype("category")
        aggr_data["cell_label"] = aggr_data["cell_label"].cat.remove_unused_categories()
        categories = aggr_data["cell_label"].cat.categories.tolist()
        categories.sort()
        aggr_data["cell_label"] = aggr_data["cell_label"].cat.reorder_categories(
            categories
        )

        if transform:
            self.write("Correcting the for sample size...")
            aggr_data["score_untrans"] = aggr_data["score"]
            for ct in aggr_data.cell_label.cat.categories:
                trans = self.transformation.loc[ct]  # type: ignore

                aggr_data.loc[aggr_data.cell_label == ct, "score"] = aggr_data.loc[
                    aggr_data.cell_label == ct, "score"
                ] - (
                    np.log(aggr_data.loc[aggr_data.cell_label == ct, "count"])
                    * trans.slope
                    + trans.intercept
                )

        if only_reference:
            aggr_data = aggr_data[aggr_data["query_or_reference"] == "reference"]
        return aggr_data

    def write(self, text, verbose: bool = True):
        """
        Writes output in logger.

        Parameters
        ----------
        text
            String to stream out in logs.
        verbose
            Whether to stream additional information.
        """
        if self.log:
            if verbose:
                print(f"{datetime.datetime.now()}\t{text}")
            text = text.encode("utf-8")
            print(f"{datetime.datetime.now()}\t{text}", file=self.logfile)
            self.logfile.flush()
        else:
            if verbose:
                print(f"{datetime.datetime.now()}\t{text}")
