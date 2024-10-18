import scanpy as sc
import pandas as pd
import numpy as np
from .mapqc import MapqcResults


def sample_query_cells(
    mapqc_results: MapqcResults,
    n_cells: int,
    grouping_key: str,
):
    """
    Sample n_cells from the query category.
    """
    # check that grouping key is in adata
    if not grouping_key in mapqc_results.adata.obs.columns:
        raise ValueError(f"{grouping_key} not found in adata.obs.columns")
    # check that grouping key has values for all cells:
    if (
        pd.isnull(mapqc_results.adata.obs[grouping_key]).any()
        or "None" in mapqc_results.adata.obs[grouping_key]
    ):
        raise ValueError(
            f"grouping_key {grouping_key} has np.nan or None values for some cells."
        )
    # Now calculate the number of query cells in each group
    query_mask = (
        mapqc_results.adata.obs[mapqc_results.ref_q_key] == mapqc_results.query_cat
    )
    query_group_counts = mapqc_results.adata.obs[query_mask][
        grouping_key
    ].value_counts()
    # Now calculate the proportion of cells (both reference and query) in each group.
    # take their average, and normalize to sum to 1.
    ref_group_counts = mapqc_results.adata.obs[~query_mask][grouping_key].value_counts()
    group_proportions_reference = ref_group_counts / ref_group_counts.sum()
    group_proportions_query = query_group_counts / query_group_counts.sum()
    group_proportions = (group_proportions_reference + group_proportions_query) / 2

    # Calculate the number of cells to take from each group
    n_cells_per_group = (group_proportions * n_cells).round().astype(int)

    # Clip n cells per group and calculate how many cells need to be redistributed
    clipped_n_cells_per_group = n_cells_per_group.clip(upper=query_group_counts)
    cells_to_redistribute = n_cells - clipped_n_cells_per_group.sum()

    # Redistribute cells if necessary
    while cells_to_redistribute > 0:
        # Calculate remaining capacity for each group
        remaining_capacity = query_group_counts - clipped_n_cells_per_group
        groups_with_capacity = remaining_capacity[remaining_capacity > 0]

        if len(groups_with_capacity) == 0:
            break  # No more capacity in any group

        # Calculate proportions for groups with remaining capacity
        redistribution_proportions = group_proportions.loc[groups_with_capacity.index]

        # Redistribute cells
        additional_cells = np.ceil(
            redistribution_proportions * cells_to_redistribute
        ).astype(
            int
        )  # round up to prevent infinite loop
        clipped_additional_cells = additional_cells.clip(
            upper=remaining_capacity.loc[additional_cells.index]
        )
        # If, due to rounding up, we sample too many cells now, subtract cells
        # from largest groups:
        n_cells_extra = (clipped_additional_cells.sum() - cells_to_redistribute).astype(
            int
        )
        if n_cells_extra > 0:
            for i in range(n_cells_extra):
                group = clipped_additional_cells.index[i]
                clipped_additional_cells[group] = clipped_additional_cells[group] - 1

        clipped_n_cells_per_group.loc[additional_cells.index] = (
            clipped_n_cells_per_group.loc[additional_cells.index]
            + clipped_additional_cells
        )
        cells_to_redistribute = n_cells - clipped_n_cells_per_group.sum()

    # Sample cells from each group, store their index number and name
    sampled_cells_index_numbers = []
    for group, n in clipped_n_cells_per_group.items():
        group_query_cell_index_numbers = np.arange(mapqc_results.adata.n_obs)[
            (mapqc_results.adata.obs[grouping_key] == group) & query_mask
        ]
        sampled_cells_index_numbers.extend(
            np.random.choice(group_query_cell_index_numbers, size=n, replace=False)
        )
    sampled_cells_names = mapqc_results.adata.obs_names[sampled_cells_index_numbers]
    return sampled_cells_index_numbers, sampled_cells_names
